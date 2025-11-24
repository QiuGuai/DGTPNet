import torch
import torch.nn as nn
import os
import json
import numpy as np
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import open3d as o3d
from plyfile import PlyData, PlyElement

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                              builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial_points = 2048
                partial = misc.generate_viewpoint_fog_cloud(gt, partial_points,fog_beta_range=(0, 0.2),fog_noise_ratio_range=(0, 0.3))
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
            
            ret = base_model(partial)
            
            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)
        
            _loss = sparse_loss + dense_loss 
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.7f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save checkpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)      
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial_points = 2048
                partial = misc.generate_viewpoint_fog_cloud(gt, partial_points,fog_beta_range=(0, 0.2),fog_noise_ratio_range=(0, 0.3))
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])


            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)


            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

            #     sparse = coarse_points.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                                
            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                           ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
        
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


fog_levels = {
    'none': {'beta': 0.0, 'noise_ratio': 0.0}, 'easy': {'beta': 0.05, 'noise_ratio': 0.1},
    'median': {'beta': 0.1, 'noise_ratio': 0.2}, 'hard': {'beta': 0.2, 'noise_ratio': 0.3},
}

def save_point_cloud(points, filename):
    """
    Saves point cloud data to a .ply file.
    :param points: PyTorch Tensor or NumPy array, shape (N, 3).
    :param filename: The file path to save to.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")

def save_full_dgs_ply(dgs_params, filename):
    """
    Saves the complete 3DGS data output from the PC3DGSA model to a .ply file.

    :param dgs_params: A tuple from the model output (means, scales, rotations, colors, opacities).
    :param filename: The output .ply filename.
    """
    if dgs_params is None:
        print(f"Skipping saving 3DGS file because dgs_params is None.")
        return
    # 1. Unpack the tuple to get the individual attribute tensors
    # The shape of these tensors should all be (1, num_gaussians, D), because the batch size during testing is 1
    means, scales, rotations, colors, opacities = dgs_params

    # 2. Remove the batch dimension (B=1) and convert to NumPy
    means_np = means[0].detach().cpu().numpy()
    scales_np = scales[0].detach().cpu().numpy()
    rotations_np = rotations[0].detach().cpu().numpy()
    colors_np = colors[0].detach().cpu().numpy()
    opacities_np = opacities[0].detach().cpu().numpy()

    # 3. Define the element structure for the .ply file, matching the standard format for 3DGS viewers
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    # 4. Create an empty structured array
    num_points = means_np.shape[0]
    elements = np.empty(num_points, dtype=dtype)
    
    # 5. Merge all attribute data and fill it into the structured array
    #    Note that the order here must strictly correspond to the dtype definition above
    attributes = (
        means_np[:, 0], means_np[:, 1], means_np[:, 2],
        colors_np[:, 0], colors_np[:, 1], colors_np[:, 2],
        opacities_np[:, 0],
        scales_np[:, 0], scales_np[:, 1], scales_np[:, 2],
        rotations_np[:, 0], rotations_np[:, 1], rotations_np[:, 2], rotations_np[:, 3] # rot_0,1,2,3 correspond to w,x,y,z
    )
    for i, name in enumerate(dtype):
        elements[name[0]] = attributes[i]

    # 6. Create a PlyData object and write it to the file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element], text=True) # text=True can increase readability
    
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ply_data.write(filename)
    print(f"Saved Full 3D Gaussians to {filename}")

def load_and_preprocess_pcd(pcd_path, npoints):
    """
    Loads and preprocesses a single point cloud file.
    This version is modified to robustly load formats like .txt, .xyz, .ply, etc.
    """
    # Get the file extension
    file_extension = os.path.splitext(pcd_path)[1].lower()

    points = None
    # If it's a text format like .txt or .xyz, use NumPy to load
    if file_extension in ['.txt', '.xyz','csv']:
        try:
            # np.loadtxt is the standard method for reading this type of text file
            points = np.loadtxt(pcd_path, usecols=(0, 1, 2)) # Only read the first three columns as X, Y, Z
        except Exception as e:
            raise IOError(f"Failed to load {pcd_path} using NumPy: {e}")
    # For other standard formats, continue using open3d
    else:
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            if pcd.has_points():
                points = np.asarray(pcd.points)
            else:
                raise IOError(f"Open3D could not read any points from {pcd_path}.")
        except Exception as e:
            raise IOError(f"Failed to load {pcd_path} using Open3D: {e}")


    # Ensure points were loaded successfully
    if points is None or points.size == 0:
        raise ValueError(f"Could not load any point cloud data from file {pcd_path}, please check if the file is valid or the path is correct.")

    # --- Subsequent preprocessing code remains unchanged ---

    # Normalization
    points = points - np.mean(points, axis=0)
    # Add a small epsilon value to prevent division by zero
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if dist < 1e-8:
        dist = 1.0 # If the point cloud is already close to the origin, do not scale
    points = points / dist

    # Random sampling
    if len(points) >= npoints:
        p_idx = np.random.choice(len(points), npoints, replace=False)
        points = points[p_idx]
    else: # If the number of points is insufficient, sample with replacement
        p_idx = np.random.choice(len(points), npoints, replace=True)
        points = points[p_idx]
    
    # Convert to a Torch tensor and add a batch dimension
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)
    return points_tensor

def test_single(base_model, args, config, logger=None):
    """
    A function specifically for testing a single external point cloud file.
    This version has been updated to allow specifying a custom output directory via command-line arguments.
    """
    base_model.eval()  # Set to evaluation mode

    # Check if there is a new 'output_dir' parameter in 'args'.
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        vis_dir = args.output_dir
        print_log(f"Using custom output directory: {vis_dir}", logger=logger)
    else:
        vis_dir = os.path.join(args.experiment_path, 'single_file_results')
        print_log(f"Using default output directory: {vis_dir}", logger=logger)
    
    # Ensure the save directory exists
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    # ==================== End of Main Modifications ====================
    
    # Get the filename to use for saving results
    pcd_filename = os.path.splitext(os.path.basename(args.pcd_path))[0]

    with torch.no_grad():
        # Load and preprocess the point cloud
        npoints = 2048
        partial = load_and_preprocess_pcd(args.pcd_path, npoints)
        
        if args.use_gpu:
            partial = partial.cuda()

        # ==================== Timing Function (Start) ====================
        print_log("Running model inference...", logger=logger)
        start_time = time.time()
        ret = base_model(partial)
        end_time = time.time()
        inference_time = end_time - start_time
        print_log(f"Inference time for single file: {inference_time:.4f} seconds", logger=logger)
        # ==================== Timing Function (End) ====================
        
        # Safely unpack the model output
        coarse_points = ret[0]
        dense_points = ret[-1]
        
        # Only treat ret[1] as 3DGS parameters if the model is PC3DGSA
        dgs_params = None
        if config.model.NAME == 'PC3DGSA':
            if len(ret) > 2:
                dgs_params = ret[1]
            else:
                print_log(f"Warning: Model is PC3DGSA but did not return 3+ elements for file {pcd_filename}.", logger=logger)
        
        # --- Save all results ---
        # 1. Save the preprocessed input point cloud
        processed_input_path = os.path.join(vis_dir, f'{pcd_filename}_input_processed.ply')
        save_point_cloud(partial[0], processed_input_path)

        # 2. Save the model's intermediate 3DGS representation (if it exists)
        if dgs_params is not None:
            dgs_path = os.path.join(vis_dir, f'{pcd_filename}_3dgs_full.ply')
            save_full_dgs_ply(dgs_params, dgs_path)
        
        # 3. Save the dense point cloud generated by the model (final output)
        output_path = os.path.join(vis_dir, f'{pcd_filename}_output.ply')
        save_point_cloud(dense_points[0], output_path)

    print_log(f"Processing complete. Results saved in: {vis_dir}", logger=logger)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)

    # Added: Check if processing a single file
    if hasattr(args, 'pcd_path') and args.pcd_path is not None:
        print_log(f"Processing single point cloud file: {args.pcd_path}", logger=logger)
        # Load the model
        base_model = builder.model_builder(config.model)
        builder.load_model(base_model, args.ckpts, logger=logger)
        if args.use_gpu:
            base_model.to(args.local_rank)
        
        # Call the single file processing function
        test_single(base_model, args, config, logger)

    else:
        # Original logic: Process the test dataset
        print_log("Processing test dataset...", logger=logger)
        _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
        base_model = builder.model_builder(config.model)
        # load checkpoints
        builder.load_model(base_model, args.ckpts, logger = logger)
        if args.use_gpu:
            base_model.to(args.local_rank)

        #   DDP       
        if args.distributed:
            raise NotImplementedError()

        # Criterion
        ChamferDisL1 = ChamferDistanceL1()
        ChamferDisL2 = ChamferDistanceL2()

        test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)


def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    # ==================== Timing Function (Start) ====================
    # Create a new AverageMeter for inference time
    inference_time_meter = AverageMeter(['InferenceTime'])
    # ==================== Timing Function (End) ====================

    # Create a folder to store all visualization results
    vis_dir = os.path.join(args.experiment_path, 'visualization_results')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()

                # ==================== Timing Function (Start) ====================
                start_time = time.time()
                ret = base_model(partial)
                end_time = time.time()
                inference_time_meter.update([end_time - start_time])
                # ==================== Timing Function (End) ====================
                
                # Safely unpack the model output
                coarse_points = ret[0]
                dense_points = ret[-1]
                
                # Only treat ret[1] as 3DGS parameters if the model is PC3DGSA
                dgs_params = None
                if config.model.NAME == 'PC3DGSA':
                    if len(ret) > 2:
                        dgs_params = ret[1]
                    else:
                        print_log(f"Warning: Model is PC3DGS but did not return 3+ elements for sample {model_id}.", logger=logger)

                save_point_cloud(
                    partial[0],
                    os.path.join(vis_dir, f'{idx:04d}_{model_id}_input.ply')
                )

                # Only save if dgs_params exists
                if dgs_params is not None:
                    save_full_dgs_ply(
                        dgs_params,
                        os.path.join(vis_dir, f'{idx:04d}_{model_id}_3dgs_full.ply')
                    )
                
                # Save the dense point cloud generated by the model (Output)
                save_point_cloud(
                    dense_points[0],
                    os.path.join(vis_dir, f'{idx:04d}_{model_id}_output.ply')
                )
                
                # Save the ground truth complete point cloud (Ground Truth)
                save_point_cloud(
                    gt[0], 
                    os.path.join(vis_dir, f'{idx:04d}_{model_id}_gt.ply')
                )

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                # test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                npoints = 2048
                choice = [np.array([1,1,1]), np.array([1,1,-1]), np.array([1,-1,1]), np.array([-1,1,1]),
                          np.array([-1,-1,1]), np.array([-1,1,-1]), np.array([1,-1,-1]), np.array([-1,-1,-1])]
                fixed_fog = fog_levels[args.mode]
                beta = fixed_fog['beta']
                noise_ratio = fixed_fog['noise_ratio']
                for item in choice:    
                    partial = misc.generate_viewpoint_fog_cloud(
                                    gt, 
                                    npoints,
                                    fog_beta_range=(beta, beta),  # Convert fixed value to a range
                                    fog_noise_ratio_range=(noise_ratio, noise_ratio), # Convert fixed value to a range
                                    fixed_direction=item, 
                                    fixed_radius_scale=2.5
                                )
                    partial = partial.to('cuda')
                    
                    # ==================== Timing Function (Start) ====================
                    start_time = time.time()
                    ret = base_model(partial)
                    end_time = time.time()
                    inference_time_meter.update([end_time - start_time])
                    # ==================== Timing Function (End) ====================
                    
                    coarse_points = ret[0]
                    dense_points = ret[-1]

                    dgs_params = None
                    if config.model.NAME == 'DGTPNet':
                        if len(ret) > 2:
                            dgs_params = ret[1]
                        else:
                            print_log(f"Warning: Model is PC3DGS but did not return 3+ elements for sample {model_id}.", logger=logger)

                    dir_suffix = "_".join(str(int(x)) for x in item.tolist())
                    save_point_cloud(
                        partial[0],
                        os.path.join(vis_dir, f'{idx:04d}_{dir_suffix}_input.ply')
                    )
                    if dgs_params is not None:
                        save_full_dgs_ply(
                            dgs_params,
                            os.path.join(vis_dir, f'{idx:04d}_{dir_suffix}_3dgs_full.ply')
                        )
                    save_point_cloud(
                        dense_points[0],
                        os.path.join(vis_dir, f'{idx:04d}_{dir_suffix}_output.ply')
                    )
                    save_point_cloud(
                        gt[0],
                        os.path.join(vis_dir, f'{idx:04d}_{dir_suffix}_gt.ply')
                    )

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s InferenceTime = %.4f' %
                          (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                           ['%.4f' % m for m in _metrics], inference_time_meter.val(0)), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)
    
        print_log(f'[TEST] Average Inference Time: {inference_time_meter.avg(0):.4f} seconds', logger=logger)

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return
