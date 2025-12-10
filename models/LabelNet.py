import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


class LabelNetworkModule(nn.Module):
    def __init__(self, label_feature_dim, gnn_hidden_dim, gnn_output_dim):
        super(LabelNetworkModule, self).__init__()
        self.gnn_layer1 = SAGEConv(label_feature_dim, gnn_hidden_dim)
        self.gnn_layer2 = SAGEConv(gnn_hidden_dim, gnn_output_dim)

        self.weight_generator = nn.Sequential(
            nn.Linear(gnn_output_dim, 1),
            nn.Tanh()
        )
        self.omega = None  # 用于存储计算出的权重

    def normalize_weights(self, w):
        w_min = torch.min(w)
        w_max = torch.max(w)
        if (w_max - w_min).abs() < 1e-8:
            return torch.full_like(w, 0.5)
        return (w - w_min) / (w_max - w_min)

    def forward(self, l_t, y_all_labels):
        # 1. 构建标签共现图
        co_occurrence = torch.matmul(l_t.T.float(), l_t.float()).bool().float()
        label_edge_index = co_occurrence.nonzero().t().contiguous()

        # 2. GNN 增强
        y_star = F.relu(self.gnn_layer1(y_all_labels, label_edge_index))
        y_star = self.gnn_layer2(y_star, label_edge_index)

        # 3. 计算并保存标签权重 omega
        raw_weights = self.weight_generator(y_star)
        self.omega = self.normalize_weights(raw_weights).squeeze()

        # 4. 计算 Affinity 矩阵 S
        l_weighted = l_t.float() * self.omega
        sum_weights_per_image = l_weighted.sum(dim=1)

        numerator = torch.matmul(l_weighted, l_t.T.float())
        denominator = sum_weights_per_image.view(-1, 1) + sum_weights_per_image.view(1, -1)

        denominator[denominator.abs() < 1e-8] = 1  # 防止除以零

        s_t = numerator / denominator
        return s_t