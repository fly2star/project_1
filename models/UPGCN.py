# models/UPGCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyPrunedGCN(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.5):
        """
        不确定性剪枝图卷积网络 (Uncertainty-Pruned GCN)
        :param in_features: 输入特征维度 (通常等于 bit 数)
        :param hidden_features: 输出特征维度
        """
        super(UncertaintyPrunedGCN, self).__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # 🌟 核心安全机制：可学习的 alpha 参数
        # 初始化为 0.0，确保训练初期 GCN 输出为 0，完全不影响原模型性能
        self.alpha = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, x, u):
        """
        :param x: [Batch, Bit] - 原始哈希码 (建议是 tanh 之前的 logits，或者是 tanh 之后的也行)
        :param u: [Batch, 1] - 不确定性 (0~1之间)
        """
        # 1. 构建构图特征 (归一化，防止模长影响相似度)
        x_norm = F.normalize(x, p=2, dim=1)
        
        # 2. 基础相似度矩阵 (Cosine Similarity) -> [Batch, Batch]
        adj = torch.mm(x_norm, x_norm.t())
        
        # 3. 不确定性剪枝 (Soft Pruning)
        # reliability: [Batch, 1], 值越大越可靠
        reliability = 1.0 - u 
        
        # 只有两个节点都可靠时，边权重才高
        # mask[i, j] = rel[i] * rel[j]
        pruning_mask = torch.mm(reliability, reliability.t())

        # debug
        if adj.shape != pruning_mask.shape:
            print(f"!!! SHAPE MISMATCH ERROR !!!")
            print(f"x shape: {x.shape}")
            print(f"u shape: {u.shape}")
            print(f"adj shape: {adj.shape}")
            print(f"pruning_mask shape: {pruning_mask.shape}")
        
        # 4. 最终邻接矩阵
        # 加上单位矩阵 I (Self-loop)，保留自身信息
        A_final = adj * pruning_mask + torch.eye(adj.shape[0], device=x.device)
        
        # 5. 归一化 (Row Normalization)
        # 避免度大的节点特征数值爆炸
        D_inv = A_final.sum(dim=1, keepdim=True).pow(-1)
        # 处理除以0的情况 (虽然加了eye不太可能为0，但为了稳健)
        D_inv[torch.isinf(D_inv)] = 0.0
        A_norm = A_final * D_inv
        
        # 6. 图卷积运算: A * ReLU(Dropout(Wx))
        h = self.fc(x) 
        h = self.dropout(h)
        h = self.relu(h)
        h_gcn = torch.mm(A_norm, h)
        
        return h_gcn