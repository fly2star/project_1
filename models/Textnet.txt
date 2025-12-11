import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
import torch.nn.init as init


# TextNet(  (fc): Sequential(    (0): Linear(in_features=1386, out_features=8192, bias=True)    (1): ReLU(inplace=True)    (2): Linear(in_features=8192, out_features=8192, bias=True)    (3): ReLU(inplace=True)    (4): Linear(in_features=8192, out_features=128, bias=True)  ))

class TextNet(nn.Module):
    def __init__(self, y_dim, bit, embed_dim=1024, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2, trainable=True):
        """
        :param y_dim: 输入特征维度 (如 1386)
        :param bit: 哈希码长度
        :param embed_dim: 统一嵌入空间维度（用于特征对齐）
        :param trainable: 是否允许训练（默认 True；用于冻结特征提取阶段）
        """
        super(TextNet, self).__init__()
        self.module_name = "txt_model"

        self.embed = nn.Sequential(
            nn.Linear(y_dim, embed_dim),
            nn.ReLU(inplace=True)
        )

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(embed_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)

        self.norm = norm

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward_to_embed(self, x):
        """仅映射到共享嵌入空间（用于相似度计算）"""
        with torch.no_grad():
            return self.embed(x)

    def forward_embed(self, x):
        """仅映射到共享嵌入空间（用于相似度计算）"""
        return self.embed(x)

    def forward(self, x):
        """标准哈希输出路径"""
        out = self.embed(x)
        out = self.fc(out).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out