import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2,ChamferDistanceL1
from pointnet2_ops.pointnet2_utils import furthest_point_sample


# Helper functions:
def knn(x, k):
    """Calculate k-nearest neighbors for a batch of data"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """Construct graph features from k-nearest neighbors"""
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    device = x.device

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x_transposed = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x_transposed.view(B * N, -1)[idx, :]  # (B*N*k, C)
    feature = feature.view(B, N, k, C)
    x = x_transposed.view(B, N, 1, C).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (B, 2*C, N, k)
    return feature

def quat_to_rotmat(quat):
    """Convert quaternion to rotation matrix"""
    quat = F.normalize(quat, p=2, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    r00 = 1 - 2 * (yy + zz); r01 = 2 * (xy - wz); r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz); r11 = 1 - 2 * (xx + zz); r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy); r21 = 2 * (yz + wx); r22 = 1 - 2 * (xx + yy)
    r00, r01, r02 = r00.unsqueeze(-1), r01.unsqueeze(-1), r02.unsqueeze(-1)
    r10, r11, r12 = r10.unsqueeze(-1), r11.unsqueeze(-1), r12.unsqueeze(-1)
    r20, r21, r22 = r20.unsqueeze(-1), r21.unsqueeze(-1), r22.unsqueeze(-1)
    row0 = torch.cat([r00, r01, r02], dim=-1)
    row1 = torch.cat([r10, r11, r12], dim=-1)
    row2 = torch.cat([r20, r21, r22], dim=-1)
    rot_mat = torch.cat([row0.unsqueeze(-2), row1.unsqueeze(-2), row2.unsqueeze(-2)], dim=-2)
    return rot_mat


class DGCNN_Feature_Extractor(nn.Module):
    """Extract local geometric features using EdgeConv"""
    def __init__(self, input_dim=3, embed_dim=512, k=20):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim

        c1, c2, c3, c4 = 64, 128, 256, self.embed_dim

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(c1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn3 = nn.BatchNorm2d(c3)
        self.bn4 = nn.BatchNorm1d(c4)

        # Convolutional layers (EdgeConv)
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim * 2, c1, kernel_size=1, bias=False),
                                     self.bn1,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(c1 * 2, c2, kernel_size=1, bias=False),
                                     self.bn2,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(c2 * 2, c3, kernel_size=1, bias=False),
                                     self.bn3,
                                     nn.LeakyReLU(negative_slope=0.2))
        # Concatenate features from all preceding layers
        self.conv4 = nn.Sequential(nn.Conv1d(c1 + c2 + c3, c4, kernel_size=1, bias=False),
                                     self.bn4,
                                     nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # x: (B, N, C_in)
        x = x.permute(0, 2, 1) # (B, C_in, N)
        
        # First EdgeConv
        graph_feature_1 = get_graph_feature(x, k=self.k) # (B, C_in*2, N, k)
        x1 = self.conv1(graph_feature_1)                 # (B, c1, N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]           # (B, c1, N)

        # Second EdgeConv
        graph_feature_2 = get_graph_feature(x1, k=self.k) # (B, c1*2, N, k)
        x2 = self.conv2(graph_feature_2)                  # (B, c2, N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]             # (B, c2, N)

        # Third EdgeConv
        graph_feature_3 = get_graph_feature(x2, k=self.k) # (B, c2*2, N, k)
        x3 = self.conv3(graph_feature_3)                  # (B, c3, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0]             # (B, c3, N)

        # Concatenate all intermediate features
        features_concat = torch.cat((x1, x2, x3), dim=1) # (B, c1+c2+c3, N)

        feature = self.conv4(features_concat) # (B, embed_dim, N)
        
        # Reshape back to (B, N, embed_dim)
        return feature.permute(0, 2, 1) # (B, N, embed_dim)


class PointTransformerEncoder(nn.Module):
    """Transformer with DGCNN feature extractor and positional encoding"""
    def __init__(self, input_dim=3, d_model=512, nhead=4, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.1, k_neighbors=20):
        super().__init__()
        self.d_model = d_model

        self.feature_extractor = DGCNN_Feature_Extractor(
            input_dim=input_dim,
            embed_dim=self.d_model,
            k=k_neighbors
        )

        self.positional_encoding_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),      # 3 -> 128
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),          # 128 -> 256
            nn.ReLU(inplace=True),
            nn.Linear(256, self.d_model)      # 256 -> 512 (d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src):
        # src: (B, N, C_in)
        
        local_features = self.feature_extractor(src) # (B, N, d_model)

        pos_encoding = self.positional_encoding_mlp(src) # (B, N, d_model)
        
        features_with_pos = local_features + pos_encoding
        
        memory = self.transformer_encoder(features_with_pos) # (B, N, d_model)
        return memory

class GaussianRefinementLayer(nn.Module):
    """A single Gaussian parameter refinement layer, containing cross-attention and FFN"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, queries, point_features_memory):
        attn_output, _ = self.cross_attention(query=queries, key=point_features_memory, value=point_features_memory)
        queries = self.norm1(queries + attn_output)
        ffn_output = self.ffn(queries)
        queries = self.norm2(queries + ffn_output)
        return queries


class Encoder3DGS_Transformer(nn.Module):
    def __init__(self, input_dim=3, num_gaussians=512, d_model=256, nhead=4,
                 num_encoder_layers=4, dim_feedforward=512, dropout=0.1,
                 num_refinement_layers=2, k_neighbors=20):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.d_model = d_model

        self.point_transformer = PointTransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, k_neighbors=k_neighbors
        )

        self.refinement_layers = nn.ModuleList([
            GaussianRefinementLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_refinement_layers)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 3 + 3 + 4 + 3 + 1) # 14 parameters
        )

    def forward(self, x):
        point_features_memory = self.point_transformer(x) # (B, N, d_model)
        xyz_coords = x[..., :3].contiguous()
        fps_indices = furthest_point_sample(xyz_coords, self.num_gaussians) # (B, G)
        
        queries = torch.gather(
            point_features_memory, 1,
            fps_indices.long().unsqueeze(-1).expand(-1, -1, self.d_model)
        ) # (B, G, d_model)

        for layer in self.refinement_layers:
            queries = layer(queries, point_features_memory)

        predicted_params = self.mlp_head(queries) # (B, G, 14)

        # Split and process the parameters
        means = predicted_params[..., 0:3]
        log_scales = predicted_params[..., 3:6]
        rotations = predicted_params[..., 6:10]
        colors = predicted_params[..., 10:13]
        opacities = predicted_params[..., 13:14]

        scales = torch.exp(log_scales)
        rotations_normalized = F.normalize(rotations, p=2, dim=-1)
        colors = torch.sigmoid(colors)
        opacities = torch.sigmoid(opacities)

        return (means, scales, rotations_normalized, colors, opacities)

class Decoder3DGSampler(nn.Module):
    def __init__(self, num_output_points=2048, use_opacity_weighted_sampling=True):
        super().__init__()
        self.num_output_points = num_output_points
        self.use_opacity_weighted_sampling = use_opacity_weighted_sampling
    
    def forward(self, dgs_params):
        means, scales, rotations, colors, opacities = dgs_params
        B, G, _ = means.shape
        N_out = self.num_output_points
        device = means.device

        if G == 0:
            return torch.zeros(B, N_out, 3, device=device)

        if self.use_opacity_weighted_sampling and opacities is not None and G > 0:
            opacities_flat = opacities.squeeze(-1) + 1e-8
            probs_sum = opacities_flat.sum(dim=1, keepdim=True)
            probs_are_zero = (probs_sum <= 1e-7)
            uniform_probs = torch.ones_like(opacities_flat) / G
            probs = torch.where(probs_are_zero, uniform_probs, opacities_flat / probs_sum)
            if N_out == 0: return torch.zeros(B, 0, 3, device=device)
            sampled_gaussian_indices = torch.multinomial(probs, num_samples=N_out, replacement=True)
            idx_expanded_3 = sampled_gaussian_indices.unsqueeze(-1).expand(-1, -1, 3)
            idx_expanded_4 = sampled_gaussian_indices.unsqueeze(-1).expand(-1, -1, 4)
            sampled_means = torch.gather(means, 1, idx_expanded_3)
            sampled_scales = torch.gather(scales, 1, idx_expanded_3)
            sampled_rotations = torch.gather(rotations, 1, idx_expanded_4)
            z = torch.randn(B, N_out, 3, device=device)
            p_scaled = sampled_scales * z
            rot_mats = quat_to_rotmat(sampled_rotations.reshape(-1,4)).reshape(B, N_out, 3, 3)
            p_rotated = torch.matmul(rot_mats, p_scaled.unsqueeze(-1)).squeeze(-1)
            output_points = sampled_means + p_rotated
        else:
            if G == 0 or N_out == 0:
                return torch.zeros(B, N_out if N_out > 0 else 0, 3, device=device)
            if N_out <= G:
                k, eff_G = 1, N_out
                means_eff, scales_eff, rotations_eff = means[:, :eff_G, :], scales[:, :eff_G, :], rotations[:, :eff_G, :]
            else:
                k, eff_G = math.ceil(N_out / G), G
                means_eff, scales_eff, rotations_eff = means, scales, rotations
            z = torch.randn(B, eff_G, k, 3, device=device)
            p_scaled = scales_eff.unsqueeze(2) * z
            rot_mats = quat_to_rotmat(rotations_eff.reshape(-1, 4)).reshape(B, eff_G, 3, 3)
            p_rotated = torch.einsum('bgij,bgkj->bgki', rot_mats, p_scaled)
            p_world = means_eff.unsqueeze(2) + p_rotated
            all_samples = p_world.reshape(B, eff_G * k, 3)
            output_points = all_samples[:, :N_out, :]
        return output_points

@MODELS.register_module()
class DGTPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = getattr(config, 'input_dim', 3)
        self.num_gaussians = config.num_gaussians
        self.d_model = getattr(config, 'd_model', 256)
        self.nhead = getattr(config, 'nhead', 4)
        self.num_encoder_layers = getattr(config, 'num_encoder_layers', 4)
        self.dim_feedforward = getattr(config, 'dim_feedforward', 512)
        self.dropout = getattr(config, 'dropout', 0.1)

        self.k_neighbors = getattr(config, 'k_neighbors', 20)

        self.num_refinement_layers = getattr(config, 'num_refinement_layers', 2)

        self.num_coarse_points = config.num_coarse_points
        self.num_fine_points = config.num_fine_points

        self.use_opacity_weighted_sampling = getattr(config, 'use_opacity_weighted_sampling', True)

        self.encoder = Encoder3DGS_Transformer(
            input_dim=self.input_dim,
            num_gaussians=self.num_gaussians,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            num_refinement_layers=self.num_refinement_layers,
            k_neighbors=self.k_neighbors
        )

        # Create two decoder instances: one for fine and one for coarse point clouds
        self.fine_decoder = Decoder3DGSampler( 
            num_output_points=self.num_fine_points,
            use_opacity_weighted_sampling=self.use_opacity_weighted_sampling
        )
        self.coarse_decoder = Decoder3DGSampler(
            num_output_points=self.num_coarse_points,
            use_opacity_weighted_sampling=self.use_opacity_weighted_sampling
        )

        self.use_L2 = getattr(config, 'l2_loss', True) # Whether to use L2 loss (ChamferDistanceL2)
        self.build_loss_func()

    def build_loss_func(self):
        if not self.use_L2:
            self.loss_func = ChamferDistanceL1()
        else:
            self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt, epoch=0):
        loss_coarse = self.loss_func(ret[-1], gt)
        loss_fine = self.loss_func(ret[0], gt)
        
        return loss_coarse, loss_fine # Return the losses

    def forward(self, xyz):
        if xyz.dim() == 3 and xyz.shape[1] == self.input_dim and xyz.shape[1] != xyz.shape[2] :
             xyz = xyz.transpose(1, 2)
        elif xyz.dim() == 3 and xyz.shape[2] != self.input_dim: 
             xyz = xyz[..., :self.input_dim]
        
        if xyz.shape[1] == 0: # Input point cloud is empty
             B = xyz.shape[0]
             dummy_output_pc = torch.zeros(B, self.num_output_points, 3, device=xyz.device) # Directly return empty output
             return dummy_output_pc, dummy_output_pc

        dgs_params = self.encoder(xyz) # Encoder: point cloud -> 3DGS parameters

        coarse_pc = self.coarse_decoder(dgs_params) # Decoder: 3DGS parameters -> coarse point cloud
        fine_pc = self.fine_decoder(dgs_params) # Decoder: 3DGS parameters -> fine point cloud

        return coarse_pc, dgs_params ,fine_pc