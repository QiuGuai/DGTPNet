import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from models.SeedFormerUtils import vTransformer, PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, get_nearest_index, indexing_neighbor
import math
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2,ChamferDistanceL1


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024, n_knn=20):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = vTransformer(256, dim=64, n_knn=n_knn)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, partial_cloud):
        """
        Args:
             partial_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = partial_cloud
        l0_points = partial_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points, l2_xyz, l2_points


class SeedGenerator(nn.Module):
    def __init__(self, feat_dim=512, seed_dim=128, n_knn=20, factor=2, attn_channel=True):
        super(SeedGenerator, self).__init__()
        self.uptrans = UpTransformer(256, 128, dim=64, n_knn=n_knn, use_upfeat=False, attn_channel=attn_channel, up_factor=factor, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor (B, feat_dim, 1)
            patch_xyz: (B, 3, 128)
            patch_feat: (B, seed_dim, 128)
        """
        x1 = self.uptrans(patch_xyz, patch_feat, patch_feat, upfeat=None)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        completion = self.mlp_4(x3)  # (B, 3, 256)
        return completion, x3


class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True, 
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor,1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = self.mlp_v(torch.cat([key, query], 1)) # (B, dim, N)
        identity = value
        key = self.conv_key(key) # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        # upfeat embedding
        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat) # (B, dim, N)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn) # (B, dim, N, k)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel) # (B, dim, N*up_factor, k)

        # softmax function
        attention = self.scale(attention)

        # knn value is correct
        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel # (B, dim, N, k)
        value = self.upsample1(value) # (B, dim, N*up_factor, k)

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg) # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity) # (B, out_dim, N)
        identity = self.upsample2(identity) # (B, out_dim, N*up_factor)

        return y+identity
        

class UpLayer(nn.Module):
    """
    Upsample Layer with upsample transformers
    """
    def __init__(self, dim, seed_dim, up_factor=2, i=0, radius=1, n_knn=20, interpolate='three', attn_channel=True):
        super(UpLayer, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.n_knn = n_knn
        self.interpolate = interpolate

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + seed_dim, layer_dims=[256, dim])

        self.uptrans1 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, up_factor=None)
        self.uptrans2 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, attn_channel=attn_channel, up_factor=self.up_factor)

        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim*2, hidden_dim=dim, out_dim=dim)

        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, seed, seed_feat, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, feat_dim, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_new: Tensor, upsampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape

        # Collect seedfeature
        if self.interpolate == 'nearest':
            idx = get_nearest_index(pcd_prev, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3) # (B, seed_dim, N_prev)
        elif self.interpolate == 'three':
            # three interpolate
            idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True) # (B, N_prev, 3), (B, N_prev, 3)
            dist_recip = 1.0 / (dis + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True) # (B, N_prev, 1)
            weight = dist_recip / norm # (B, N_prev, 3)
            feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1), dim=-1) # (B, seed_dim, N_prev)
        else:
            raise ValueError('Unknown Interpolation: {}'.format(self.interpolate))

        # Query mlps
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_upsample], 1)
        Q = self.mlp_2(feat_1)

        # Upsample Transformers
        H = self.uptrans1(pcd_prev, K_prev if K_prev is not None else Q, Q, upfeat=feat_upsample) # (B, 128, N_prev)
        feat_child = self.uptrans2(pcd_prev, K_prev if K_prev is not None else H, H, upfeat=feat_upsample) # (B, 128, N_prev*up_factor)

        # Get current features K
        H_up = self.upsample(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        # New point cloud
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_new = self.upsample(pcd_prev)
        pcd_new = pcd_new + delta

        return pcd_new, K_curr

@MODELS.register_module()
class SeedFormer(nn.Module):
    """
    SeedFormer 点云补全模型。
    __init__ 方法已被重构为 PC3DGSA 的配置驱动风格。
    """
    def __init__(self, config):
        """
        Args:
            config (object): 一个包含所有模型超参数的配置对象或类。
                             参数通过 getattr(config, 'param_name', default_value) 的方式读取。
        """
        super(SeedFormer, self).__init__()

        # --- 从配置对象中读取参数 ---
        # 这种方式使得所有超参数都由配置文件控制，增加了灵活性。
        self.num_p0 = getattr(config, 'num_p0', 512)
        feat_dim = getattr(config, 'feat_dim', 512)
        embed_dim = getattr(config, 'embed_dim', 128)
        n_knn = getattr(config, 'n_knn', 20)
        radius = getattr(config, 'radius', 1.0)
        up_factors = getattr(config, 'up_factors', [1, 4, 4])
        seed_factor = getattr(config, 'seed_factor', 2)
        interpolate = getattr(config, 'interpolate', 'three')
        attn_channel = getattr(config, 'attn_channel', True)

        # --- 使用读取的参数实例化子模块 ---

        # 编码器部分: 特征提取器和种子生成器
        self.feat_extractor = FeatureExtractor(out_dim=feat_dim, n_knn=n_knn)
        self.seed_generator = SeedGenerator(feat_dim=feat_dim, seed_dim=embed_dim, n_knn=n_knn, 
                                            factor=seed_factor, attn_channel=attn_channel)

        # 解码器部分: 上采样层
        up_layers = []
        for i, factor in enumerate(up_factors):
            up_layers.append(
                UpLayer(dim=embed_dim, seed_dim=embed_dim, up_factor=factor, i=i, 
                        n_knn=n_knn, radius=radius, interpolate=interpolate, 
                        attn_channel=attn_channel)
            )
        self.up_layers = nn.ModuleList(up_layers)

        self.use_L2 = getattr(config, 'l2_loss', True) # 是否使用L2损失 (ChamferDistanceL2)
        self.build_loss_func()

    def forward(self, partial_cloud):
        """
        Args:
            partial_cloud: (B, N, 3)
        """
        # Encoder
        feat, patch_xyz, patch_feat = self.forward_encoder(partial_cloud)

        # Decoder
        pred_pcds = self.forward_decoder(feat, partial_cloud, patch_xyz, patch_feat)

        return pred_pcds

    def forward_encoder(self, partial_cloud):
        # feature extraction
        # 将 (B, N, 3) 转换为 (B, 3, N) 以匹配卷积层的输入格式
        partial_cloud_transposed = partial_cloud.permute(0, 2, 1).contiguous()
        feat, patch_xyz, patch_feat = self.feat_extractor(partial_cloud_transposed) # feat: (B, feat_dim, 1)

        return feat, patch_xyz, patch_feat

    def forward_decoder(self, feat, partial_cloud, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor, (B, feat_dim, 1) - 全局特征
            partial_cloud: Tensor, (B, N, 3) - 原始输入点云
            patch_xyz: Tensor, (B, 3, num_patches) - 局部块的中心点坐标
            patch_feat: Tensor, (B, embed_dim, num_patches) - 局部块的特征
        """
        pred_pcds = []

        # 1. 生成种子点云
        seed, seed_feat = self.seed_generator(feat, patch_xyz, patch_feat)
        seed = seed.permute(0, 2, 1).contiguous() # -> (B, num_seed, 3)
        pred_pcds.append(seed)

        # 2. 准备上采样的初始点云
        # 合并生成的种子点和原始输入点，然后下采样到一个固定的粗点云数量 (num_p0)
        pcd = fps_subsample(torch.cat([seed, partial_cloud], 1), self.num_p0) # -> (B, num_p0, 3)
        
        # 3. 逐层上采样
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous() # -> (B, 3, num_p0)
        
        # 将种子点云的维度也准备好以供上采样层使用
        seed_transposed = seed.permute(0, 2, 1).contiguous() # -> (B, 3, num_seed)

        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, seed_transposed, seed_feat, K_prev)
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())

        return pred_pcds
    
    def build_loss_func(self):
        if not self.use_L2: # False 表示 L1
            self.loss_func = ChamferDistanceL1()
        else: # True 表示 L2
            self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt, epoch=0): # 计算损失

        P0, P1, P2, P3 = ret

        cd0 = self.loss_func(P0, gt)
        cd1 = self.loss_func(P1, gt)
        cd2 = self.loss_func(P2, gt)
        cd3 = self.loss_func(P3, gt)

        loss1 = cd0 + cd1 + cd2
        loss2 = cd3

        return loss1, loss2 # 返回损失和用于记录的损失值