import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
from pointnet2_ops import pointnet2_utils
import open3d as o3d

def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc

def random_sample(data, number):
    '''
        data B N 3
        number int
    '''
    assert data.size(1) > number
    assert len(data.shape) == 3
    ind = torch.multinomial(torch.rand(data.size()[:2]).float(), number).to(data.device)
    data = torch.gather(data, 1, ind.unsqueeze(-1).expand(-1, -1, data.size(-1)))
    return data

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_lambda_sche(opti, config, last_epoch=-1):
    if config.get('decay_step') is not None:
        # lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        warming_up_t = getattr(config, 'warmingup_e', 0)
        lr_lbmd = lambda e: max(config.lr_decay ** ((e - warming_up_t) / config.decay_step), config.lowest_decay) if e >= warming_up_t else max(e / warming_up_t, 0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return scheduler

def build_lambda_bnsche(model, config, last_epoch=-1):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return bnm_scheduler
    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)
    

def farthest_point_sample_indices(point_cloud, n_points):
    N, _ = point_cloud.shape
    if N == 0:
        return np.array([], dtype=np.int32)

    centroids_indices = np.zeros(n_points, dtype=np.int32)
    distance = np.full(N, 1e10)
    
    farthest_idx = np.random.randint(0, N)
    
    for i in range(n_points):
        centroids_indices[i] = farthest_idx
        centroid = point_cloud[farthest_idx, :]
        
        dist = np.sum((point_cloud - centroid) ** 2, axis=-1)
        
        mask = dist < distance
        distance[mask] = dist[mask]
        
        farthest_idx = np.argmax(distance)
        
    return centroids_indices

def create_single_view_point_cloud(pcd, viewpoint):
    _, pt_map = pcd.hidden_point_removal(viewpoint, radius=100)
    partial_pcd = pcd.select_by_index(pt_map)
    return partial_pcd

def generate_viewpoint_partial_cloud(gt_batch, n_points_partial):

    gt_batch_np = gt_batch.cpu().numpy()
    partial_clouds_list = []
    
    for i in range(gt_batch_np.shape[0]):
        gt_points = gt_batch_np[i] # Shape: [N, 3]
        
        gt_pcd_o3d = o3d.geometry.PointCloud()
        gt_pcd_o3d.points = o3d.utility.Vector3dVector(gt_points)
        
        center = gt_pcd_o3d.get_center()
        bounding_box = gt_pcd_o3d.get_axis_aligned_bounding_box()
        radius_scale = 3.6
        distance_from_center = np.linalg.norm(bounding_box.get_max_bound() - bounding_box.get_min_bound()) * radius_scale
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        viewpoint_np = center + direction * distance_from_center
        
        partial_pcd_o3d = create_single_view_point_cloud(gt_pcd_o3d, viewpoint_np)
        
        partial_points_np = np.asarray(partial_pcd_o3d.points)
        
        if partial_points_np.shape[0] == 0:
            indices = np.random.choice(gt_points.shape[0], n_points_partial, replace=True)
            resampled_points = gt_points[indices, :]
        elif partial_points_np.shape[0] < n_points_partial:
            indices = np.random.choice(partial_points_np.shape[0], n_points_partial, replace=True)
            resampled_points = partial_points_np[indices, :]
        else:
            indices = farthest_point_sample_indices(partial_points_np, n_points_partial)
            resampled_points = partial_points_np[indices, :]

        partial_clouds_list.append(resampled_points)
        
    partial_batch_np = np.stack(partial_clouds_list, axis=0)
    partial_batch_tensor = torch.from_numpy(partial_batch_np).float()
    
    return partial_batch_tensor

def simulate_fog(point_cloud, viewpoint, beta=0.05, noise_ratio=0.15):

    num_original_points = point_cloud.shape[0]
    if num_original_points == 0:
        return np.array([], dtype=np.float64).reshape(0, 3)

    distances_to_viewpoint = np.linalg.norm(point_cloud - viewpoint, axis=1)
    keep_probabilities = np.exp(-beta * distances_to_viewpoint)
    mask = np.random.rand(num_original_points) < keep_probabilities
    surviving_points = point_cloud[mask]
    
    num_removed_points = num_original_points - surviving_points.shape[0]
    num_noise_points = int(num_removed_points * noise_ratio)
    
    noise_points = np.array([], dtype=np.float64).reshape(0, 3)
    if num_noise_points > 0 and point_cloud.shape[0] > 0:
        cloud_center = np.mean(point_cloud, axis=0)
        cloud_radius = np.max(np.linalg.norm(point_cloud - cloud_center, axis=1)) if point_cloud.shape[0] > 1 else 0.1
        
        random_unit_vectors = np.random.randn(num_noise_points, 3)
        random_unit_vectors /= np.linalg.norm(random_unit_vectors, axis=1, keepdims=True)
        sphere_target_points = cloud_center + random_unit_vectors * cloud_radius
        
        direction_vectors = sphere_target_points - viewpoint
        t_values = np.random.rand(num_noise_points, 1)
        noise_points = viewpoint + t_values * direction_vectors
    
    return np.vstack((surviving_points, noise_points)) if noise_points.shape[0] > 0 else surviving_points

def generate_viewpoint_fog_cloud(
    gt_batch, 
    n_points_partial, 
    fog_beta_range=(0.0, 0.2), 
    fog_noise_ratio_range=(0.0, 0.3), 
    fixed_direction=None, 
    fixed_radius_scale=None
):

    gt_batch_np = gt_batch.cpu().numpy()
    partial_clouds_list = []
    
    for i in range(gt_batch_np.shape[0]):
        gt_points = gt_batch_np[i]
        
        gt_pcd_o3d = o3d.geometry.PointCloud()
        gt_pcd_o3d.points = o3d.utility.Vector3dVector(gt_points)
        
        center = gt_pcd_o3d.get_center()
        bounding_box = gt_pcd_o3d.get_axis_aligned_bounding_box()
        
        if fixed_direction is not None and fixed_radius_scale is not None:
            direction = fixed_direction / np.linalg.norm(fixed_direction)
            radius_scale = fixed_radius_scale
        else:
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            radius_scale = np.random.uniform(2.0, 3.5)
            
        distance = np.linalg.norm(bounding_box.get_max_bound() - bounding_box.get_min_bound()) * radius_scale
        viewpoint_np = center + direction * distance
        
        partial_pcd_o3d = create_single_view_point_cloud(gt_pcd_o3d, viewpoint_np)
        partial_points_np = np.asarray(partial_pcd_o3d.points)

        current_beta = np.random.uniform(fog_beta_range[0], fog_beta_range[1])
        current_noise_ratio = np.random.uniform(fog_noise_ratio_range[0], fog_noise_ratio_range[1])
        
        foggy_points_np = simulate_fog(partial_points_np, viewpoint_np, beta=current_beta, noise_ratio=current_noise_ratio)
        
        num_foggy_points = foggy_points_np.shape[0]
        if num_foggy_points == 0:
            indices = np.random.choice(gt_points.shape[0], n_points_partial, replace=True)
            resampled_points = gt_points[indices, :]
        elif num_foggy_points < n_points_partial:
            indices = np.random.choice(num_foggy_points, n_points_partial, replace=True)
            resampled_points = foggy_points_np[indices, :]
        else:
            indices = farthest_point_sample_indices(foggy_points_np, n_points_partial)
            resampled_points = foggy_points_np[indices, :]

        partial_clouds_list.append(resampled_points)
        
    partial_batch_np = np.stack(partial_clouds_list, axis=0)
    return torch.from_numpy(partial_batch_np).float()


def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    try:
        ax = fig.gca(projection=Axes3D.name, adjustable='box')
    except:
        ax = fig.add_subplot(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(30, 45)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img



def visualize_KITTI(path, data_list, titles = ['input','pred'], cmap=['bwr','autumn'], zdir='y', 
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1) ):
    fig = plt.figure(figsize=(6*len(data_list),6))
    cmax = data_list[-1][:,0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:,0] /cmax
        ax = fig.add_subplot(1, len(data_list) , i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color,vmin=-1,vmax=1 ,cmap = cmap[0],s=4,linewidth=0.05, edgecolors = 'black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + '.png'
    fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e//50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1,1))[0,0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim = 1)
    return pc
    

def random_scale(partial, gt, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale, gt * scale



from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
