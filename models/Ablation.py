import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2, ChamferDistanceL1


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x_transposed = x.transpose(2, 1).contiguous()
    feature = x_transposed.view(B * N, -1)[idx, :]
    feature = feature.view(B, N, k, C)
    x = x_transposed.view(B, N, 1, C).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNN_Feature_Extractor(nn.Module):
    def __init__(self, input_dim=3, embed_dim=512, k=20):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim

        c1, c2, c3, c4 = 64, 128, 256, self.embed_dim

        self.bn1 = nn.BatchNorm2d(c1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn3 = nn.BatchNorm2d(c3)
        self.bn4 = nn.BatchNorm1d(c4)

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim * 2, c1, kernel_size=1, bias=False),
                                     self.bn1,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(c1 * 2, c2, kernel_size=1, bias=False),
                                     self.bn2,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(c2 * 2, c3, kernel_size=1, bias=False),
                                     self.bn3,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(c1 + c2 + c3, c4, kernel_size=1, bias=False),
                                     self.bn4,
                                     nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        graph_feature_1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(graph_feature_1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        graph_feature_2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(graph_feature_2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        graph_feature_3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(graph_feature_3)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        features_concat = torch.cat((x1, x2, x3), dim=1)
        feature = self.conv4(features_concat)
        
        return feature.permute(0, 2, 1)


class PointTransformerEncoder(nn.Module):
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
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src):
        local_features = self.feature_extractor(src)
        pos_encoding = self.positional_encoding_mlp(src)
        features_with_pos = local_features + pos_encoding
        
        memory = self.transformer_encoder(features_with_pos)
        return memory


class PointTransformerDecoder(nn.Module):
    def __init__(self, num_coarse, num_fine, d_model, nhead, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_coarse = num_coarse
        self.num_fine = num_fine

        self.coarse_query_embed = nn.Embedding(num_coarse, d_model)
        coarse_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.coarse_transformer_decoder = nn.TransformerDecoder(coarse_decoder_layer, num_layers=num_decoder_layers)
        self.coarse_output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True), nn.Linear(d_model // 2, 3)
        )

        self.fine_query_embed = nn.Embedding(num_fine, d_model)
        fine_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.fine_transformer_decoder = nn.TransformerDecoder(fine_decoder_layer, num_layers=num_decoder_layers)
        self.fine_output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True), nn.Linear(d_model // 2, 3)
        )

    def forward(self, memory):
        B = memory.shape[0]

        coarse_queries = self.coarse_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        coarse_output = self.coarse_transformer_decoder(tgt=coarse_queries, memory=memory)
        coarse_pc = self.coarse_output_mlp(coarse_output)

        fine_queries = self.fine_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        fine_output = self.fine_transformer_decoder(tgt=fine_queries, memory=memory)
        fine_pc = self.fine_output_mlp(fine_output)
        
        return coarse_pc, fine_pc


@MODELS.register_module()
class Ablation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = getattr(config, 'input_dim', 3)
        self.d_model = getattr(config, 'd_model', 256)
        self.nhead = getattr(config, 'nhead', 4)
        self.num_encoder_layers = getattr(config, 'num_encoder_layers', 4)
        self.num_decoder_layers = getattr(config, 'num_decoder_layers', 6)
        self.dim_feedforward = getattr(config, 'dim_feedforward', 512)
        self.dropout = getattr(config, 'dropout', 0.1)
        self.k_neighbors = getattr(config, 'k_neighbors', 20)
        self.num_coarse_points = config.num_coarse_points
        self.num_fine_points = config.num_fine_points

        self.encoder = PointTransformerEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            k_neighbors=self.k_neighbors
        )

        self.decoder = PointTransformerDecoder(
            num_coarse=self.num_coarse_points,
            num_fine=self.num_fine_points,
            d_model=self.d_model,
            nhead=self.nhead,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )

        self.use_L2 = getattr(config, 'l2_loss', True)
        self.build_loss_func()

    def build_loss_func(self):
        if not self.use_L2:
            self.loss_func = ChamferDistanceL1()
        else:
            self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        fine_pc, coarse_pc = ret
        loss_coarse = self.loss_func(coarse_pc, gt)
        loss_fine = self.loss_func(fine_pc, gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        if xyz.dim() == 3 and xyz.shape[1] == self.input_dim and xyz.shape[1] != xyz.shape[2]:
            xyz = xyz.transpose(1, 2)
        elif xyz.dim() == 3 and xyz.shape[2] != self.input_dim:
            xyz = xyz[..., :self.input_dim]
        
        if xyz.shape[1] == 0:
            B = xyz.shape[0]
            dummy_coarse = torch.zeros(B, self.num_coarse_points, 3, device=xyz.device)
            dummy_fine = torch.zeros(B, self.num_fine_points, 3, device=xyz.device)
            return dummy_fine, dummy_coarse

        point_features_memory = self.encoder(xyz)

        coarse_pc, fine_pc = self.decoder(point_features_memory)

        return fine_pc, coarse_pc
