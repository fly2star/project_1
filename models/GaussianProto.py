import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianPrototype(nn.Module):
    def __init__(self, bit, num_classes, device='cuda'):
        """
        高斯概率原型模块 (共享均值版)
        """
        super(GaussianPrototype, self).__init__()
        self.bit = bit
        self.num_classes = num_classes
        self.device = device
        
        # ❌ 删除：self.class_mu = nn.Parameter(...) 
        # 我们不再自己维护均值，而是借用 shared_W
        
        # ✅ 保留：类别方差原型 (Learnable)
        # 初始化为 0 (即方差为 1)
        self.class_logvar = nn.Parameter(torch.zeros(num_classes, bit))

    def forward(self, mu_sample, logvar_sample, labels, shared_W):
        """
        :param shared_W: [bit, num_classes] - 来自主模型的共享权重
        """
        # shared_W 通常是 [bit, num_classes]，我们需要 [num_classes, bit] 作为均值
        # 所以进行转置
        class_mu = shared_W.t() 
        
        # 1. 获取样本对应的类别原型
        labels_norm = F.normalize(labels, p=1, dim=1)
        
        # target_mu: [Batch, bit]
        # 使用传入的 shared_W (转置后) 计算目标均值
        target_mu = torch.mm(labels_norm, class_mu)
        
        # target_logvar: [Batch, bit]
        target_logvar = torch.mm(labels_norm, self.class_logvar)
        
        # 2. 计算 KL Divergence
        # term1 = var_s / var_c
        term1 = (logvar_sample - target_logvar).exp() 
        # term2 = (mu_c - mu_s)^2 / var_c
        term2 = (target_mu - mu_sample).pow(2) / target_logvar.exp() 
        # term3 = log(var_c) - log(var_s)
        term3 = target_logvar - logvar_sample 
        
        kl_loss = 0.5 * torch.sum(term1 + term2 + term3 - 1.0, dim=1)
        
        return kl_loss.mean()