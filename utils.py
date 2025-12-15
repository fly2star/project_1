"""
一些辅助函数
"""
def compute_uncertainty_diag(evidence, num_classes=2):
    """
    输入 Evidence [Batch, Class]，返回对角线上的不确定性 [Batch, 1]
    """
    # 1. 计算 alpha, S
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # 2. 计算 u_all (包含了 Batch 内所有 Pair 的不确定性)
    # 注意：这里的 evidence 可能是 i2t 或 t2i 的输出，通常维度是 [Batch, Class]
    # 但根据您之前的逻辑，evidence_model 输出的可能是 [Batch, Batch] 的全连接结果？
    # 让我们回顾一下之前的 bug：S 出来是 16384 (128*128)。
    # 所以这里的 evidence 输入应该是已经经过 .view 之前的原始输出
    
    u_all = num_classes / S # [Batch*Batch, 1]
    
    # 3. 提取对角线
    # 动态获取 Batch Size，假设 total elements = bs * bs
    total_len = u_all.size(0)
    bs = int(total_len ** 0.5) 
    
    u_mat = u_all.view(bs, bs)
    u_diag = u_mat.diag().view(-1, 1) # [Batch, 1]
    
    return u_diag