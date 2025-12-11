import torch
from torch import nn
from torch.nn import functional as F
# from utils.config import args

class ImageNet(nn.Module):
    def __init__(self, y_dim, bit, num_classes, embed_dim=1024, mid_num1=2048*8, hiden_layer=2, trainable=True, shared_W=None):
        super(ImageNet, self).__init__()
        
        # --- 1. 定义可学习的类别中心 W ---
        # W 将特征空间 [bit] 映射到隶-属度空间 [num_classes]
        # 如果外部提供了 shared_W，则使用它（共享）；否则在模块内创建新的可训练参数
        if shared_W is None:
            self.W = nn.Parameter(torch.Tensor(bit, num_classes))
            nn.init.orthogonal_(self.W, gain=1)
        else:
            if tuple(shared_W.shape) != (bit, num_classes):
                raise ValueError(f"shared_W shape {shared_W.shape} does not match (bit,num_classes)=({bit},{num_classes})")
            self.W = shared_W

        # --- 2. (新增) 定义从隶-属度空间“提取”哈希码的层 ---
        # 这是一个逆向工程：将类别隶-属度信息压缩回哈希码
        self.hash_projector = nn.Sequential(
            nn.Linear(num_classes, bit), # 从 num_classes 维压缩回 bit 维
            nn.ReLU(),
            nn.Linear(bit, bit)
        )
         
        # --- 新增：定义对数方差的层 ---1205
        self.var_layer = nn.Sequential(
            nn.Linear(bit, bit),
            nn.Tanh() # 限制一下范围，防止方差过大
        )

        # === 3. 原有的特征提取部分保持不变 ===
        self.embed = nn.Sequential(
            nn.Linear(y_dim, embed_dim),
            nn.ReLU(inplace=True)
        )
        # 注意：fc 层的最终输出维度是 bit，这仍然是我们的核心“特征空间”维度
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(embed_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                # 原始代码 mid_num2 可能未定义，这里修正一下
                modules += [nn.Linear(pre_num, pre_num), nn.ReLU(inplace=True)]
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        # =======================================    
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    # 加入互信息 1205
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧: z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # 在测试/评估时，直接使用均值，这是确定性的
            return mu

    def forward_to_embed(self, x):
        """仅映射到共享嵌入空间（用于相似度计算）"""
        with torch.no_grad():
            return self.embed(x)

    def forward_embed(self, x):
        """仅映射到共享嵌入空间（用于相似度计算）"""
        return self.embed(x)

    # def forward(self, x):
    #     """
    #     新的 forward 逻辑，同时输出隶-属度和哈希码
    #     """
    #     # a. 首先，计算出核心的、连续的内部特征 a (shape: [bs, bit])
    #     # 注意：我们在这里去掉 tanh 和 norm，得到最原始的特征
    #     internal_feature = self.embed(x)
    #     internal_feature = self.fc(internal_feature)

    #     # b. (新增) 将内部特征映射到隶-属度空间
    #     #    W 需要在使用前进行单位化
    #     W_normalized = self.W / torch.norm(self.W, p=2, dim=0, keepdim=True)
    #     membership_degree = internal_feature @ W_normalized

    #     # c. (新增) 从隶-属度向量中“提取”哈希码代理
    #     #    这个哈希码现在是基于类别信息的，理论上更具语义
    #     hash_proxy_from_membership = self.hash_projector(membership_degree)
        
    #     # d. (新增) 也可以结合原始特征，使用残差连接
    #     final_hash_proxy = internal_feature + hash_proxy_from_membership

    #     # e. 对最终的哈希码代理进行 tanh 和 norm，以匹配 DECH 的要求
    #     final_hash_proxy = torch.tanh(final_hash_proxy)
    #     norm_val = torch.norm(final_hash_proxy, dim=1, keepdim=True)
    #     final_hash_code = final_hash_proxy / norm_val

    #     # f. 返回一个字典，包含所有我们需要的信息
    #     return {
    #         "membership_degree": membership_degree, # -> 用于 loss_fml
    #         "hash_code": final_hash_code            # -> 用于 loss_dech_original
    #     }
    
    def forward(self, x):
        # 1. 计算内部特征 (Feature Extraction)
        internal_feature = self.embed(x)
        internal_feature = self.fc(internal_feature)

        # 2. 计算隶属度 (FUME Logic)
        W_normalized = self.W / torch.norm(self.W, p=2, dim=0, keepdim=True)
        membership_degree = internal_feature @ W_normalized

        # 3. 计算均值 mu (即之前的 final_hash_proxy)
        hash_proxy_from_membership = self.hash_projector(membership_degree)
        mu = internal_feature + hash_proxy_from_membership
        
        # --- 新增：计算对数方差 logvar ---
        logvar = self.var_layer(internal_feature)

        # --- 新增：采样潜在变量 z ---
        # z 是“带噪声”的哈希码代理
        z = self.reparameterize(mu, logvar)

        # 4. 生成最终哈希码 (Hash Code Generation)
        # 注意：这里我们对 z 进行 tanh，而不是对 mu 进行 tanh
        # 这样训练时模型能学会适应噪声
        final_hash_proxy = torch.tanh(z)
        norm_val = torch.norm(final_hash_proxy, dim=1, keepdim=True)
        final_hash_code = final_hash_proxy / norm_val

        # 返回字典增加 mu 和 logvar，用于计算 KL loss
        return {
            "membership_degree": membership_degree,
            "hash_code": final_hash_code,
            "mu": mu,          # 用于 KL Loss
            "logvar": logvar   # 用于 KL Loss
        }
