import numpy as np
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import open3d as o3d
from plyfile import PlyData, PlyElement
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def run_net(args, config, train_writer=None):
    logger = get_logger(args.log_name)
    # Build Dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)
    # Build Model
    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)
        
    # Parameter Setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Resume Ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)
        if not os.path.exists(args.start_ckpts):
                print_log(f'[RESUME INFO] no checkpoint file from path {args.start_ckpts}...', logger = logger)
                return 0, 0, 0
        print_log(f'[RESUME INFO] Loading optimizer from {args.start_ckpts}...', logger = logger )
        state_dict = torch.load(args.start_ckpts, map_location='cpu')
        start_epoch = state_dict['epoch'] + 1

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, \
                                                         device_ids=[args.local_rank % torch.cuda.device_count()], \
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
        
    # Optimizer & Scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    elif args.start_ckpts is not None:
        # if not os.path.exists(args.start_ckpts):
        #     print_log(f'[RESUME INFO] no checkpoint file from path {args.start_ckpts}...', logger = logger)
        #     return 0, 0, 0
        # print_log(f'[RESUME INFO] Loading optimizer from {args.start_ckpts}...', logger = logger )
        # state_dict = torch.load(args.start_ckpts, map_location='cpu')
        # optimizer
        optimizer.load_state_dict(state_dict['optimizer'])

    # Training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss', 'SparsePenalty', 'DensePenalty'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        dataset_name = config.dataset.train._base_.NAME
        npoints = config.dataset.train._base_.N_POINTS
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            
            
            if dataset_name == 'PCN':
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

            elif dataset_name == 'MVP':
                gt = data[1].cuda()
                partial = data[0].cuda()
                
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
            
            ret = base_model(partial)
            #print(ret[3].size())
            loss_pcd, sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)
            orth_cons1, orth_cons2 = base_model.module.get_constrain(ret)
            #sparse_loss = config.loss.sparse_loss_weight * sparse_loss
            #dense_loss = config.loss.dense_loss_weight * dense_loss
            
            loss_cons = config.loss.orth_weight * (orth_cons1 + orth_cons2)

            _loss = loss_pcd + loss_cons
            _loss.backward()

            # Forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                orth_cons1 = dist_utils.reduce_tensor(orth_cons1, args)
                orth_cons2 = dist_utils.reduce_tensor(orth_cons2, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, \
                               orth_cons1.item() * 1000, orth_cons2.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, \
                               orth_cons1.item() * 1000, orth_cons2.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            # if train_writer is not None:
            #     train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
            #     train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
            #     train_writer.add_scalar('LR/training', optimizer.param_groups[0]['lr'], n_itr)
            #     train_writer.add_scalar('Penalty/Batch/cons1', orth_cons1.item() * 1000, n_itr)
            #     train_writer.add_scalar('Penalty/Batch/cons2', orth_cons2.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                print_log('[Memory: %f GB Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                          (mem, epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
            train_writer.add_scalar('Penalty/Epoch/orth1', losses.avg(2), epoch)
            train_writer.add_scalar('Penalty/Epoch/orth2', losses.avg(3), epoch)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
        print_log('[Training] Memory: %f GB EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (mem, epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, train_writer, args, config, logger=logger)

            # Save checkpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger) 
        torch.cuda.empty_cache()
            
    train_writer.close()
    #val_writer.close()

    
def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial_points = 2048
                partial = misc.generate_viewpoint_fog_cloud(gt, partial_points,fog_beta_range=(0, 0.2),fog_noise_ratio_range=(0, 0.3))
                partial = partial.cuda()
            elif dataset_name == 'MVP':
                gt = data[1].cuda()
                partial = data[0].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, \
                                dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt) 

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % args.val_interval == 0:
                # input_pc = partial.squeeze().detach().cpu().numpy()
                # input_pc = misc.get_ptcloud_img(input_pc)
                # val_writer.add_image('Model%02d-%d/Input'% (idx, epoch) , input_pc, epoch, dataformats='HWC')

                # sparse = coarse_points.squeeze().cpu().numpy()
                # sparse_img = misc.get_ptcloud_img(sparse)
                # val_writer.add_image('Model%02d-%d/Sparse' % (idx, epoch), sparse_img, epoch, dataformats='HWC')
                # pred_sparse_img = misc.get_ordered_ptcloud_img(sparse[0:224,:])
                # val_writer.add_image('Model%02d-%d/PredSparse' % (idx, epoch), pred_sparse_img, epoch, dataformats='HWC')

                # dense = dense_points.squeeze().cpu().numpy()
                # dense_img = misc.get_ptcloud_img(dense)
                # val_writer.add_image('Model%02d-%d/Dense' % (idx, epoch), dense_img, epoch, dataformats='HWC')
                
                # gt_ptcloud = gt.squeeze().cpu().numpy()
                # gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                # val_writer.add_image('Model%02d-%d/DenseGT' % (idx, epoch), gt_ptcloud_img, epoch, dataformats='HWC')
        
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
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
        msg += (str(taxonomy_id) + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        if taxonomy_id in shapenet_dict:
            msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Test/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Test/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())

def save_point_cloud(points, filename):
    """
    将点云数据保存为 .ply 文件。
    :param points: PyTorch Tensor 或 NumPy array, 形状为 (N, 3)。
    :param filename: 保存的文件路径。
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")

def save_full_dgs_ply(dgs_params, filename):
    """
    将PC3DGS392模型输出的完整3DGS数据保存到.ply文件。

    :param dgs_params: 从模型输出的元组 (means, scales, rotations, colors, opacities)。
    :param filename: 输出的 .ply 文件名。
    """
    # 1. 解包元组，获取各个属性张量
    # 这些张量的形状都应该是 (1, num_gaussians, D)，因为测试时的批大小为1
    means, scales, rotations, colors, opacities = dgs_params

    # 2. 移除批次维度 (B=1)，并转换到 NumPy
    means_np = means[0].detach().cpu().numpy()
    scales_np = scales[0].detach().cpu().numpy()
    rotations_np = rotations[0].detach().cpu().numpy()
    colors_np = colors[0].detach().cpu().numpy()
    opacities_np = opacities[0].detach().cpu().numpy()

    # 3. 定义.ply文件的元素结构，这与3DGS查看器的标准格式匹配
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    # 4. 创建一个空的结构化数组
    num_points = means_np.shape[0]
    elements = np.empty(num_points, dtype=dtype)
    
    # 5. 将所有属性数据合并并填充到结构化数组中
    #    请注意，这里的顺序必须与上面的 dtype 定义严格对应
    attributes = (
        means_np[:, 0], means_np[:, 1], means_np[:, 2],
        colors_np[:, 0], colors_np[:, 1], colors_np[:, 2],
        opacities_np[:, 0],
        scales_np[:, 0], scales_np[:, 1], scales_np[:, 2],
        rotations_np[:, 0], rotations_np[:, 1], rotations_np[:, 2], rotations_np[:, 3] # rot_0,1,2,3 对应 w,x,y,z
    )
    for i, name in enumerate(dtype):
        elements[name[0]] = attributes[i]

    # 6. 创建PlyData对象并写入文件
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element], text=True) # text=True可以增加可读性
    
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ply_data.write(filename)
    print(f"Saved Full 3D Gaussians to {filename}") 

fog_levels = {
    'none': {'beta': 0.0, 'noise_ratio': 0.0}, 'easy': {'beta': 0.05, 'noise_ratio': 0.1},
    'median': {'beta': 0.1, 'noise_ratio': 0.2}, 'hard': {'beta': 0.2, 'noise_ratio': 0.3},
}

def load_and_preprocess_pcd(pcd_path, npoints):
    """
    加载并预处理单个点云文件。
    此版本经过修改，可以稳健地加载 .txt, .xyz, .ply 等格式。
    """
    # 获取文件扩展名
    file_extension = os.path.splitext(pcd_path)[1].lower()

    points = None
    # 如果是.txt或.xyz等文本格式，使用NumPy加载
    if file_extension in ['.txt', '.xyz','csv']:
        try:
            # np.loadtxt 是读取这类文本文件的标准方法
            points = np.loadtxt(pcd_path, usecols=(0, 1, 2)) # 只读取前三列作为X,Y,Z
        except Exception as e:
            raise IOError(f"使用 NumPy 加载 {pcd_path} 文件失败: {e}")
    # 对于其他标准格式，继续使用open3d
    else:
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            if pcd.has_points():
                points = np.asarray(pcd.points)
            else:
                 raise IOError(f"Open3D 无法从 {pcd_path} 中读取任何点。")
        except Exception as e:
            raise IOError(f"使用 Open3D 加载 {pcd_path} 文件失败: {e}")


    # 确保成功加载了点
    if points is None or points.size == 0:
        raise ValueError(f"无法从文件 {pcd_path} 加载任何点云数据，请检查文件是否有效或路径是否正确。")

    # --- 后续的预处理代码保持不变 ---

    # 归一化
    points = points - np.mean(points, axis=0)
    # 增加一个小的epsilon值防止除以零
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if dist < 1e-8:
        dist = 1.0 # 如果点云已经接近原点，则不缩放
    points = points / dist

    # 随机采样
    if len(points) >= npoints:
        p_idx = np.random.choice(len(points), npoints, replace=False)
        points = points[p_idx]
    else: # 如果点数不足，则重复采样
        p_idx = np.random.choice(len(points), npoints, replace=True)
        points = points[p_idx]
    
    # 转换为Torch张量并添加批次维度
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)
    return points_tensor

def test_single(base_model, args, config, logger=None):
    """
    专门用于测试单个外部点云文件的函数。
    此版本已更新，允许通过命令行参数指定自定义的输出目录。
    """
    base_model.eval()  # 设置为评估模式

    # 检查 'args' 中是否有新的 'output_dir' 参数。
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        vis_dir = args.output_dir
        print_log(f"Using custom output directory: {vis_dir}", logger=logger)
    else:
        vis_dir = os.path.join(args.experiment_path, 'single_file_results')
        print_log(f"Using default output directory: {vis_dir}", logger=logger)
    
    # 确保保存目录存在
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    # ==================== 主要修改点 (结束) ====================
    
    # 获取文件名，用于保存结果
    pcd_filename = os.path.splitext(os.path.basename(args.pcd_path))[0]

    with torch.no_grad():
        # 加载并预处理点云
        npoints = 2048
        partial = load_and_preprocess_pcd(args.pcd_path, npoints)
        
        if args.use_gpu:
            partial = partial.cuda()

        # ==================== 计时功能 (开始) ====================
        print_log("Running model inference...", logger=logger)
        start_time = time.time()
        ret = base_model(partial)
        end_time = time.time()
        inference_time = end_time - start_time
        print_log(f"Inference time for single file: {inference_time:.4f} seconds", logger=logger)
        # ==================== 计时功能 (结束) ====================
        
        # 安全地解包模型输出
        coarse_points = ret[0]
        dense_points = ret[1]
        
        # --- 保存所有结果 ---
        # 1. 保存预处理后的输入点云
        processed_input_path = os.path.join(vis_dir, f'{pcd_filename}_input_processed.ply')
        save_point_cloud(partial[0], processed_input_path)
        
        # 2. 保存模型生成的密集点云 (最终输出)
        output_path = os.path.join(vis_dir, f'{pcd_filename}_output.ply')
        save_point_cloud(dense_points[0], output_path)

    print_log(f"Processing complete. Results saved in: {vis_dir}", logger=logger)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)

    # 新增：检查是否是处理单个文件
    if hasattr(args, 'pcd_path') and args.pcd_path is not None:
        print_log(f"Processing single point cloud file: {args.pcd_path}", logger=logger)
        # 加载模型
        base_model = builder.model_builder(config.model)
        builder.load_model(base_model, args.ckpts, logger=logger)
        if args.use_gpu:
            base_model.to(args.local_rank)
        
        # 调用单个文件处理函数
        test_single(base_model, args, config, logger)

    else:
        # 原始逻辑：处理测试数据集
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

    # ==================== 计时功能 (开始) ====================
    # 为推理时间创建一个新的 AverageMeter
    inference_time_meter = AverageMeter(['InferenceTime'])
    # ==================== 计时功能 (结束) ====================

    # 创建一个用于存放所有可视化结果的文件夹
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

                # ==================== 计时功能 (开始) ====================
                start_time = time.time()
                ret = base_model(partial)
                end_time = time.time()
                inference_time_meter.update([end_time - start_time])
                # ==================== 计时功能 (结束) ====================
                
                # 安全地解包模型输出
                coarse_points = ret[0]
                dense_points = ret[1]

                save_point_cloud(
                    partial[0],
                    os.path.join(vis_dir, f'{idx:04d}_{model_id}_input.ply')
                )
                
                # 保存模型生成的密集点云 (Output)
                save_point_cloud(
                    dense_points[0],
                    os.path.join(vis_dir, f'{idx:04d}_{model_id}_output.ply')
                )
                
                # 保存真实完整的点云 (Ground Truth)
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
                                    fog_beta_range=(beta, beta),  # 将固定值转为范围
                                    fog_noise_ratio_range=(noise_ratio, noise_ratio), # 将固定值转为范围
                                    fixed_direction=item, 
                                    fixed_radius_scale=2.5
                                )
                    partial = partial.to('cuda')
                    
                    # ==================== 计时功能 (开始) ====================
                    start_time = time.time()
                    ret = base_model(partial)
                    end_time = time.time()
                    inference_time_meter.update([end_time - start_time])
                    # ==================== 计时功能 (结束) ====================
                    
                    coarse_points = ret[0]
                    dense_points = ret[1]

                    dir_suffix = "_".join(str(int(x)) for x in item.tolist())
                    # 保存输入的部分点云
                    save_point_cloud(
                        partial[0],
                        os.path.join(vis_dir, f'{idx:04d}_{dir_suffix}_input.ply')
                    )
                    # 保存模型生成的密集点云 (Output)
                    save_point_cloud(
                        dense_points[0],
                        os.path.join(vis_dir, f'{idx:04d}_{dir_suffix}_output.ply')
                    )
                    # 保存真实完整的点云 (Ground Truth)
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
    
        # ==================== 计时功能 (开始) ====================
        # 在最终结果中打印平均推理时间
        print_log(f'[TEST] Average Inference Time: {inference_time_meter.avg(0):.4f} seconds', logger=logger)
        # ==================== 计时功能 (结束) ====================

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
