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

# 新的量化损失函数
def calculate_quantization_loss(image_hash_codes, text_hash_codes, device=None):
    """
    计算量化损失 (Quantization Loss)。
    目标：最小化连续哈希码与二值哈希码之间的距离。
    公式: L_q = || H - sign(H) ||^2
    """
    # 1. 生成目标二值码 B
    # sign() 函数不可导，所以必须使用 .detach() 将其从计算图中剥离
    # 我们把 B 当作一个固定的“锚点”，让 H 去拟合它
    B_img = torch.sign(image_hash_codes).detach()
    B_txt = torch.sign(text_hash_codes).detach()

    # 2. 计算均方误差 (MSE)
    # 这会迫使 H_img 和 H_txt 的值向 -1 或 1 靠拢
    loss_q_img = F.mse_loss(image_hash_codes, B_img)
    loss_q_txt = F.mse_loss(text_hash_codes, B_txt)

    # 3. 返回总量化损失
    return loss_q_img + loss_q_txt

# file: LOSS.py
# ... (保留 calculate_similarity_preserving_loss 等函数) ...

# def consistency_learning_loss(view1_feature, view2_feature, tau=1.0, device=None):
#     """
#     移植自 FUME 的 consistency_Learning, 用于自监督的对比学习。
#     """
#     if not device:
#         device = get_device()
    
#     view1_feature = view1_feature.to(device)
#     view2_feature = view2_feature.to(device)

#     # --- FUME 原始代码 ---
#     n_view = 2
#     batch_size = view1_feature.shape[0]
#     all_fea = torch.cat([view1_feature, view2_feature])
#     sim = all_fea.mm(all_fea.t())

#     sim = (sim / tau).exp()
#     sim = sim - sim.diag().diag() # 移除对角线上的自身相似度
    
#     # 构造正样本对的 mask
#     pos_mask = torch.zeros_like(sim)
#     pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
#     pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1
    
#     # 计算损失 (InfoNCE 的一种变体)
#     # 对于每个样本，其损失是 正样本相似度 / (所有样本相似度之和) 的负对数
#     positive_pairs_sim = sim[pos_mask.bool()].view(batch_size * 2, 1)
#     denominator = sim.sum(dim=1, keepdim=True)
    
#     loss = -torch.log(positive_pairs_sim / denominator).mean()
    
#     # FUME 的原始实现稍有不同，但目标一致。为了简化和鲁棒性，我们使用更标准的 InfoNCE 形式。
#     # 如果您想完全复现，可以保留 FUME 的 loss1+loss2 计算方式。
#     # 这里我们暂时使用 FUME 的原始版本：
#     sim_sum1 = sim[:, :batch_size] + sim[:, batch_size:]
#     diag1 = torch.cat([sim_sum1[:batch_size].diag(), sim_sum1[batch_size:].diag()])
#     loss1 = -(diag1 / sim.sum(1)).log().mean()

#     sim_sum2 = sim[:batch_size] + sim[batch_size:]
#     diag2 = torch.cat([sim_sum2[:, :batch_size].diag(), sim_sum2[:, batch_size:].diag()])
#     loss2 = -(diag2 / sim.sum(1)).log().mean()
#     return loss1 + loss2

# ===1210===
def consistency_learning_loss(view1_feature, view2_feature, weights=None, tau=1.0, device=None):
    """
    移植自 FUME 的 consistency_Learning (支持不确定性加权)。
    """
    if not device:
        device = get_device()
    
    view1_feature = view1_feature.to(device)
    view2_feature = view2_feature.to(device)

    # --- FUME 原始代码逻辑开始 ---
    n_view = 2
    batch_size = view1_feature.shape[0]
    all_fea = torch.cat([view1_feature, view2_feature])
    sim = all_fea.mm(all_fea.t())

    sim = (sim / tau).exp()
    sim = sim - sim.diag().diag() # 移除对角线上的自身相似度
    
    # --- 准备权重 (新增) ---
    if weights is not None:
        # weights 形状是 [bs]，我们需要将其扩展为 [2*bs] 以匹配 all_fea
        # 这样前 bs 个权重对应 view1，后 bs 个权重对应 view2
        weights_expanded = torch.cat([weights, weights]).to(device)
    
    # --- 计算 Loss 1 ---
    # FUME 原逻辑: sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
    # 展开写更清晰:
    sim_sum1 = sim[:, :batch_size] + sim[:, batch_size:]
    
    # diag1 形状: [2*bs]
    diag1 = torch.cat([sim_sum1[:batch_size].diag(), sim_sum1[batch_size:].diag()])
    
    # 计算逐样本损失 (不取 mean)
    loss1_elementwise = -(diag1 / sim.sum(1)).log()
    
    # 加权并求平均
    if weights is not None:
        loss1 = (loss1_elementwise * weights_expanded).mean()
    else:
        loss1 = loss1_elementwise.mean()

    # --- 计算 Loss 2 ---
    # FUME 原逻辑: sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
    sim_sum2 = sim[:batch_size] + sim[batch_size:]
    
    # diag2 形状: [2*bs]
    diag2 = torch.cat([sim_sum2[:, :batch_size].diag(), sim_sum2[:, batch_size:].diag()])
    
    # 计算逐样本损失
    loss2_elementwise = -(diag2 / sim.sum(1)).log()
    
    # 加权并求平均
    if weights is not None:
        loss2 = (loss2_elementwise * weights_expanded).mean()
    else:
        loss2 = loss2_elementwise.mean()
    
    return loss1 + loss2


def get_train_category_credibility(predict, labels):
    """
    根据模型的预测（隶属度）和真实标签，计算训练时的类别可信度。
    移植自 FUME 论文的 processor.py。
    
    :param predict: 模型的原始预测，shape: [batch_size, num_classes]
    :param labels: 真实标签 (one-hot)，shape: [batch_size, num_classes]
    :return: 类别可信度 r, shape: [batch_size, num_classes]
    """
    # 确保 labels 是 float 类型以进行乘法操作
    labels = labels.float()
    
    # 找出对于真实为负的类别中，模型给出的最高分
    top1Possibility = (predict * (1 - labels)).max(1)[0].reshape([-1, 1])

    # 找出对于真实为正的类别中，模型给出的分数
    labelPossibility = (predict * labels).max(1)[0].reshape([-1, 1])

    # 计算“必要性 (neccessity)”
    neccessity = (1 - labelPossibility) * (1 - labels) + (1 - top1Possibility) * labels

    # 最终的可信度 = (模型原始预测 + 必要性) / 2
    r = (predict + neccessity) / 2
    return r


def bayesian_uncertainty_loss(loss_elementwise, uncertainty, lambda_reg=0.5):
    """
    基于贝叶斯风险最小化的损失函数。
    利用不确定性 u 对原始损失进行加权。
    
    公式: L = (Loss / (u + eps)) + lambda * log(u + eps)
    
    :param loss_elementwise: 每个样本的原始损失，shape [bs, 1] 或 [bs]
    :param uncertainty: 每个样本的不确定性 u，shape [bs, 1] 或 [bs]
    :param lambda_reg: 正则项的权重，防止不确定性无限增大
    """
    eps = 1e-6
    # 确保维度匹配
    uncertainty = uncertainty.view_as(loss_elementwise)
    
    # 1. 损失缩放项：不确定性越大，原本的损失权重越小
    #    这允许模型在困难样本上“减负”
    weighted_loss = loss_elementwise / (uncertainty + eps)
    
    # 2. 正则化项：惩罚过大的不确定性
    #    迫使模型在能学懂的地方尽量降低不确定性
    regularization = torch.log(uncertainty + eps)
    
    # 组合
    total_loss = weighted_loss + lambda_reg * regularization
    
    return total_loss.mean()

def vib_kl_loss(mu, logvar):
    """
    计算 VIB 的 KL 散度损失。
    KL(N(mu, sigma^2) || N(0, I))
    公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    # logvar = log(sigma^2)
    # 这里的求和是针对 bit 维度，最后对 batch 取平均
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_div.mean()