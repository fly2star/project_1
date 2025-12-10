import torch
import torch.nn.functional as F
# from helpers import get_device
from CONFIG import args
from hypll.tensors import ManifoldTensor


def get_device():
    return f'cuda:{args.gpuIdx}'


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = 'cuda:1'
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / 10, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)

    return torch.mean(A + kl_div)


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = output
    alpha = evidence + 1

    TrueEvidence = torch.sum(target * evidence, dim=1).view(-1)

    loss = edl_loss(
        torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
    )

    if args.L3idx == 5:
        l3Loss = 1 / (TrueEvidence + 1e-5)
    elif args.L3idx == 1:
        l3Loss = -torch.log((TrueEvidence).tanh())
    elif args.L3idx == 2:
        l3Loss = torch.log(1 + 1 / TrueEvidence)

    return loss + l3Loss.mean()


def calculate_similarity_preserving_loss(image_hash_codes, text_hash_codes, s_matrix, k, device=None):
    """
    计算基于 S 矩阵的跨模态相似性保持损失
    """
    if not device:
        device = get_device()

    # 确保所有张量在同一设备上
    image_hash_codes = image_hash_codes.to(device)
    text_hash_codes = text_hash_codes.to(device)
    s_matrix = s_matrix.to(device)

    # 使用 tanh 对连续哈希码进行近似
    H_img = torch.tanh(image_hash_codes)
    H_txt = torch.tanh(text_hash_codes)

    # 目标矩阵
    target_S = k * s_matrix

    # 图-图损失
    loss_ii = F.mse_loss(H_img @ H_img.T, target_S)

    # 文-文损失
    loss_tt = F.mse_loss(H_txt @ H_txt.T, target_S)

    # 图-文损失 (核心)
    loss_it = F.mse_loss(H_img @ H_txt.T, target_S)

    # 将三者相加（可以给跨模态损失更高的权重，例如 * 2）
    total_similarity_loss = loss_ii + loss_tt + 2 * loss_it

    return total_similarity_loss


# file: LOSS.py
# ... (保留 calculate_similarity_preserving_loss 等函数) ...

def consistency_learning_loss(view1_feature, view2_feature, tau=1.0, device=None):
    """
    移植自 FUME 的 consistency_Learning, 用于自监督的对比学习。
    """
    if not device:
        device = get_device()
    
    view1_feature = view1_feature.to(device)
    view2_feature = view2_feature.to(device)

    # --- FUME 原始代码 ---
    n_view = 2
    batch_size = view1_feature.shape[0]
    all_fea = torch.cat([view1_feature, view2_feature])
    sim = all_fea.mm(all_fea.t())

    sim = (sim / tau).exp()
    sim = sim - sim.diag().diag() # 移除对角线上的自身相似度
    
    # 构造正样本对的 mask
    pos_mask = torch.zeros_like(sim)
    pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
    pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1
    
    # 计算损失 (InfoNCE 的一种变体)
    # 对于每个样本，其损失是 正样本相似度 / (所有样本相似度之和) 的负对数
    positive_pairs_sim = sim[pos_mask.bool()].view(batch_size * 2, 1)
    denominator = sim.sum(dim=1, keepdim=True)
    
    loss = -torch.log(positive_pairs_sim / denominator).mean()
    
    # FUME 的原始实现稍有不同，但目标一致。为了简化和鲁棒性，我们使用更标准的 InfoNCE 形式。
    # 如果您想完全复现，可以保留 FUME 的 loss1+loss2 计算方式。
    # 这里我们暂时使用 FUME 的原始版本：
    sim_sum1 = sim[:, :batch_size] + sim[:, batch_size:]
    diag1 = torch.cat([sim_sum1[:batch_size].diag(), sim_sum1[batch_size:].diag()])
    loss1 = -(diag1 / sim.sum(1)).log().mean()

    sim_sum2 = sim[:batch_size] + sim[batch_size:]
    diag2 = torch.cat([sim_sum2[:, :batch_size].diag(), sim_sum2[:, batch_size:].diag()])
    loss2 = -(diag2 / sim.sum(1)).log().mean()
    
    return loss1 + loss2

