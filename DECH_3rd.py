from CONFIG import args
import torch
import torch.nn as nn
from tqdm import tqdm
from models import *
from LOSS import *
import CONFIG
import os

model_save_path = os.path.join(CONFIG.model_save_dirname,f'{args.data_name}_{args.bit}bit_last_{args.L3idx}')

import torch

import logging
os.makedirs('./DECH__NEW/LogFiles',exist_ok=True)
LogFilesPath = './DECH__NEW/LogFiles'
# 创建第一个logger和handler
logger1 = logging.getLogger("map_all")
logger1.setLevel(logging.INFO)
handler1 = logging.FileHandler(f'{LogFilesPath}/map_all.log')
formatter1 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler1.setFormatter(formatter1)
logger1.addHandler(handler1)

# 创建第二个logger和handler
logger2 = logging.getLogger("map_500")
logger2.setLevel(logging.INFO)
handler2 = logging.FileHandler(f'{LogFilesPath}/map_500.log')
formatter2 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler2.setFormatter(formatter2)
logger2.addHandler(handler2)

import numpy as np 
def i2t(npts, sims, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # rank = np.where((labels[inds].mm(labels[index].t()) > 0).sum(-1))[0][0]
        rank = np.where(inds == index)[0][0]
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
    
def generate_data_laoder():
    from DATASET import CMDataset
    train_dataset = CMDataset(
        data_name=args.data_name,
        partition='train'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    retrieval_dataset = CMDataset(
        data_name=args.data_name,
        partition='retrieval'
    )
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = CMDataset(
        data_name=args.data_name,
        partition='test'
    )
    query_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader,query_loader,retrieval_loader,4096,retrieval_dataset.text_dim

# 版本 1
# def generate_hash_code(data_loader,image_model,text_model):
#     imgs, txts, labs = [], [], []
#     # imgs_fea,txts_fea = [],[]
#     with torch.no_grad():
#         for batch_idx, (images, texts, targets) in enumerate(data_loader):
#             images_outputs = [image_model(images.cuda().float())]
#             texts_outputs = [text_model(texts.cuda().float())]
#             imgs += images_outputs
#             txts += texts_outputs
#             labs.append(targets.float())

#         imgs = torch.cat(imgs).sign_()#.cpu().numpy()
#         txts = torch.cat(txts).sign_()#.cpu().numpy()
#         labs = torch.cat(labs)#.cpu().numpy()
        
#     return imgs,txts,labs

# 版本 2
# def generate_hash_code(data_loader, image_model, text_model):
#     imgs_list = []
#     txts_list = []
#     labs_list = []
    
#     image_model.eval()
#     text_model.eval()

#     with torch.no_grad():
#         # 注意：这里可能需要 tqdm
#         for batch_idx, (images, texts, targets) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating Hash Codes for Eval"):
#             images = images.cuda().float()
#             texts = texts.cuda().float()
            
#             # 1. 获取字典输出
#             image_outputs_dict = image_model(images)
#             text_outputs_dict = text_model(texts)

#             # 2. 提取 hash_code 并转到 CPU
#             img_h = image_outputs_dict["hash_code"].cpu()
#             txt_h = text_outputs_dict["hash_code"].cpu()
            
#             imgs_list.append(img_h)
#             txts_list.append(txt_h)
#             labs_list.append(targets.float())

#     # 3. 循环外拼接
#     imgs = torch.cat(imgs_list).sign()
#     txts = torch.cat(txts_list).sign()
#     labs = torch.cat(labs_list)

#     return imgs, txts, labs

# 版本 3
def generate_hash_code(dataloader, image_model, text_model, evidence_model, gcn_img, gcn_txt):
    """
    修改后的哈希码生成函数，包含 GCN 微调逻辑。
    """
    # 切换到评估模式 (这就关闭了 Dropout)
    image_model.eval()
    text_model.eval()
    evidence_model.eval()
    gcn_img.eval()
    gcn_txt.eval()
    
    bs_img_list, bs_txt_list, bs_label_list = [], [], []

    with torch.no_grad(): # 评估时不需要计算梯度
        for img, txt, label in dataloader:
            img = img.cuda().float()
            txt = txt.cuda().float()
            
            # 1. 获取原始哈希码 (Raw Hash)
            img_out = image_model(img)
            txt_out = text_model(txt)
            img_hash_raw = img_out["hash_code"]
            txt_hash_raw = txt_out["hash_code"]
            
            bs = img.size(0)

            # 2. 计算不确定性 (用于 GCN 剪枝)
            # 必须使用 raw hash 计算，逻辑与训练阶段 1 一致
            evidencei2t = evidence_model(img_hash_raw, txt_hash_raw, 'i2t')
            evidencet2i = evidence_model(img_hash_raw, txt_hash_raw, 't2i')
            
            # 计算 u (假设 num_classes_evidence=2)
            alpha_i2t = evidencei2t + 1
            S_i2t = torch.sum(alpha_i2t, dim=1, keepdim=True)
            u_i2t_all = 2.0 / S_i2t
            # 提取对角线
            u_i2t_mat = u_i2t_all.view(bs,bs)
            u_i2t_diag = u_i2t_mat.diag()
            u_img = u_i2t_diag.view(-1, 1).detach() # [Batch, 1]
            
            alpha_t2i = evidencet2i + 1
            S_t2i = torch.sum(alpha_t2i, dim=1, keepdim=True)
            u_t2i_all = 2.0 / S_t2i
            # 提取对角线
            u_t2i_mat = u_t2i_all.view(bs,bs)
            u_t2i_diag = u_t2i_mat.diag()
            u_txt = u_t2i_diag.view(-1, 1).detach() # [Batch, 1]
            
            # 3. GCN 微调 (Refinement)
            delta_img = gcn_img(img_hash_raw, u_img)
            delta_txt = gcn_txt(txt_hash_raw, u_txt)
            
            # 4. 融合得到最终哈希码 (Final Hash)
            # 注意：eval 模式下 alpha 也是加载进来的训练好的值
            img_hash_final = img_hash_raw + gcn_img.alpha * delta_img
            txt_hash_final = txt_hash_raw + gcn_txt.alpha * delta_txt
            
            # 5. 生成二值码 (用于计算 MAP)
            # sign(H) -> {-1, 1}
            code_img = torch.sign(img_hash_final)
            code_txt = torch.sign(txt_hash_final)
            
            bs_img_list.append(code_img.cpu().data.numpy())
            bs_txt_list.append(code_txt.cpu().data.numpy())
            bs_label_list.append(label.cpu().data.numpy())

    return np.concatenate(bs_img_list), np.concatenate(bs_txt_list), np.concatenate(bs_label_list)

def getbdu(evidence):
    L = (2+evidence[:,0]+evidence[:,1])
    b = evidence[:,0] / L
    d = evidence[:,1] / L
    u = 2 / L
    return b,d,u
def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH
# from tqdm import tqdm

def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=None, metric='hamming'):
    retrieval, retrieval_labels, query, query_label = [i.cuda().to(dtype=torch.float64) for i in [retrieval, retrieval_labels, query, query_label]]
    # qB, rB, query_label, retrieval_label = [i.cuda().to(dtype=torch.float64) for i in [qB, rB, query_label, retrieval_label]]
    import scipy.spatial
    # dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    # ord = dist.argsort()
    # numcases = dist.shape[0]
    
    numcases = query.shape[0]
    if k == None:
        k = retrieval_labels.shape[0]
    res = []
    for i in (range(numcases)):
        hamm = calc_hamming_dist(query[i,:],retrieval).cuda()
        _, ord = torch.sort(hamm)
        order = ord.reshape(-1)[0: k]
        tmp_label =  (query_label[i,:]@retrieval_labels[order].T)>0
        tmp_label =tmp_label.float().cuda()
        if tmp_label.sum() > 0:
            prec = torch.cumsum(tmp_label,-1) / torch.arange(1.0, 1 + tmp_label.shape[0]).cuda()
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [tmp_label@prec/total_pos]
    return torch.mean(torch.tensor(res))

# def test(query_loader,retrieval_loader,image_model,text_model,evidence_model,epoch):
#     global pre_val,no_update_count,state_dict
#     # global state_dict
#     qX,qY,qL = generate_hash_code(query_loader,image_model,text_model)
#     rX,rY,rL = generate_hash_code(retrieval_loader,image_model,text_model)
#     MAPi2t = fx_calc_map_multilabel_k(rY,rL,qX,qL,None)
#     MAPt2i = fx_calc_map_multilabel_k(rX,rL,qY,qL,None)

#     # print(MAPi2t,MAPt2i)
#     # print(f"\nEval (epoch={epoch}): MAP(i→t)={MAPi2t:.4f}, MAP(t→i)={MAPt2i:.4f}")
#     return MAPi2t, MAPt2i

# 全新版本
def test(query_loader, retrieval_loader, image_model, text_model, evidence_model, gcn_img, gcn_txt, epoch):
    global pre_val, no_update_count, state_dict
    
    # 🌟 关键修改：将 GCN 传入 generate_hash_code
    # 确保 generate_hash_code 已经按照我们上一步的讨论修改过，支持接收 GCN
    qX, qY, qL = generate_hash_code(query_loader, image_model, text_model, evidence_model, gcn_img, gcn_txt)
    rX, rY, rL = generate_hash_code(retrieval_loader, image_model, text_model, evidence_model, gcn_img, gcn_txt)

    # 修复
    qX = torch.from_numpy(qX)
    qY = torch.from_numpy(qY)
    qL = torch.from_numpy(qL)
    
    rX = torch.from_numpy(rX)
    rY = torch.from_numpy(rY)
    rL = torch.from_numpy(rL)
    
    MAPi2t = fx_calc_map_multilabel_k(rY, rL, qX, qL, None)
    MAPt2i = fx_calc_map_multilabel_k(rX, rL, qY, qL, None)
    
    return MAPi2t, MAPt2i


def main():
    if args.data_name == 'nus_wide_deep':
        args.hiden_layer = 2
    
    if args.data_name == 'mirflickr25k_deep':
        num_classes = 24
    elif args.data_name == 'nus_wide_deep':
        num_classes = 10
    elif args.data_name == 'mscoco_deep':
        num_classes = 80
    
    train_loader, query_loader, retrieval_loader, image_dim, text_dim = generate_data_laoder()

    # 创建一个共享的类别原型 W，并传入 ImageNet 和 TextNet（两者共享同一对象）
    shared_W = nn.Parameter(torch.Tensor(args.bit, num_classes))
    nn.init.orthogonal_(shared_W, gain=1)

    evidence_model = EvidenceNet(args.bit, args.tau).cuda()
    image_model = ImageNet(image_dim, args.bit, num_classes=num_classes, hiden_layer=args.hiden_layer, shared_W=shared_W).cuda()
    text_model = TextNet(text_dim, args.bit, num_classes=num_classes, hiden_layer=args.hiden_layer, shared_W=shared_W).cuda()
    gcn_img = UncertaintyPrunedGCN(in_features=args.bit, hidden_features=args.bit).cuda()
    gcn_txt = UncertaintyPrunedGCN(in_features=args.bit, hidden_features=args.bit).cuda()

    # 聚合全部参数并去重（因为 shared_W 在 image_model 和 text_model 中是同一对象）
    all_params = list(image_model.parameters()) + list(text_model.parameters()) + \
                    list(evidence_model.parameters()) + \
                    list(gcn_img.parameters()) + list(gcn_txt.parameters())
    unique_params = list({id(p): p for p in all_params}.values())

    # 使用单个优化器管理所有参数，避免对同一参数进行多次更新
    optimizer = torch.optim.Adam(unique_params, 1e-4)

    # fuzzy_module = FuzzyLogicModule(feature_dim=args.bit, num_classes=num_classes, device=args.gpuIdx, use_relu=args.use_relu).cuda()
    # parameters = list(image_model.parameters()) + list(text_model.parameters())  + list(evidence_model.parameters()) + list(fuzzy_module.parameters())

    # 旧的按-module 优化器已合并为单个 `optimizer`，这里保留 unique 参数列表以供裁剪和调试
    parameters = unique_params

    # optimizerFuzzy = torch.optim.Adam(fuzzy_module.parameters(),1e-4)

    if args.continue_eval == 1:
        status_dict = torch.load(model_save_path, weights_only=True)
        image_model.load_state_dict(status_dict['image_model_state_dict'])
        text_model.load_state_dict(status_dict['text_model_state_dict'])
        evidence_model.load_state_dict(status_dict['evidence_model_state_dict'])
        # fuzzy_module.load_state_dict(status_dict['fuzzy_module_state_dict'])
        gcn_img.load_state_dict(status_dict['gcn_img_state_dict'])
        gcn_txt.load_state_dict(status_dict['gcn_txt_state_dict'])

    # print(f'start train')
    best_i2t_map = 0.0
    best_i2t_epoch = -1

    best_t2i_map = 0.0
    best_t2i_epoch = -1

    # optimizer = Lion(parameters, lr=1e-5)
    for epoch in range(0, args.max_epochs):
        train_loss=0.0

        # print(f'********{epoch+1} / {args.max_epochs}********')
        for batch_idx, (images, texts, label) in (enumerate(train_loader)):
                
                images = images.cuda().float()
                texts = texts.cuda().float()
                label = label.cuda().float()
                
                image_outputs_dict = image_model(images)
                text_outputs_dict = text_model(texts)
                
                # 提取隶-属度
                img_membership = image_outputs_dict["membership_degree"]
                txt_membership = text_outputs_dict["membership_degree"]

                # 提取哈希码
                img_hash_raw = image_outputs_dict["hash_code"]
                txt_hash_raw = text_outputs_dict["hash_code"]
                
                # # 1210新增: 量化损失
                # loss_q = calculate_quantization_loss(img_hash_raw, txt_hash_raw)
                
                # --- 新增：提取 mu 和 logvar 并计算 KL 损失 --- 1205
                mu_img, logvar_img = image_outputs_dict["mu"], image_outputs_dict["logvar"]
                mu_txt, logvar_txt = text_outputs_dict["mu"], text_outputs_dict["logvar"]
                
                # DECH 原始损失 (基于哈希码) ---
                evidencei2t = evidence_model(img_hash_raw, txt_hash_raw, 'i2t')
                evidencet2i = evidence_model(img_hash_raw, txt_hash_raw, 't2i')
                
                # 计算不确定性 u
                # u = K / S, 这里K = 2(相似/不相似), S = sum(alpha) = sum(evidence + 1)
                num_classes_evidence = 2
                # alpha = e + 1
                alpha_i2t = evidencei2t + 1
                alpha_t2i = evidencet2i + 1
                # S = sum(alpha)
                S_i2t = torch.sum(alpha_i2t, dim=1, keepdim=True)
                S_t2i = torch.sum(alpha_t2i, dim=1, keepdim=True)
                # u = K / S
                u_i2t = num_classes_evidence / S_i2t
                u_t2i = num_classes_evidence / S_t2i
                # 获取当前的 batch_size
                bs = img_hash_raw.shape[0]
                # 将 u 按照 bs 展开
                u_i2t_mat = u_i2t.view(bs, bs)
                u_t2i_mat = u_t2i.view(bs, bs)
                # 提取对角线元素 (正样本对的不确定性)
                # 结果形状变为 [bs]
                u_i2t_diag = u_i2t_mat.diag()
                u_t2i_diag = u_t2i_mat.diag() 
                # 简化，将两个方向不确定性的平均值，作为着对样本总的不确定性
                uncertaint_joint = (u_i2t_diag + u_t2i_diag) / 2

                # 为 GCN 准备不确定性输入 
                u_img_in = u_i2t_diag.view(-1, 1).detach()
                u_txt_in = u_t2i_diag.view(-1, 1).detach()

                # GCN 增强
                delta_img = gcn_img(img_hash_raw, u_img_in)
                delta_txt = gcn_txt(txt_hash_raw, u_txt_in)

                # 融合
                img_hash_code = img_hash_raw + delta_img * gcn_img.alpha
                txt_hash_code = txt_hash_raw + delta_txt * gcn_txt.alpha

                # (可选) 再次 Tanh 约束范围
                # 如果您的后续 Loss 对范围敏感，建议加上。如果用 MSE 则不强制。
                # 推荐加上，保持与原始输出分布一致
                # img_hash_code = torch.tanh(img_hash_code)
                # txt_hash_code = torch.tanh(txt_hash_code)


                # 使用更新后的哈希码
                evidencei2t = evidence_model(img_hash_code, txt_hash_code, 'i2t')
                evidencet2i = evidence_model(img_hash_code, txt_hash_code, 't2i')
                
                # 重复流程
                # ==================================================
                # 计算不确定性 u
                # u = K / S, 这里K = 2(相似/不相似), S = sum(alpha) = sum(evidence + 1)
                num_classes_evidence = 2
                # alpha = e + 1
                alpha_i2t = evidencei2t + 1
                alpha_t2i = evidencet2i + 1
                # S = sum(alpha)
                S_i2t = torch.sum(alpha_i2t, dim=1, keepdim=True)
                S_t2i = torch.sum(alpha_t2i, dim=1, keepdim=True)
                # u = K / S
                u_i2t = num_classes_evidence / S_i2t
                u_t2i = num_classes_evidence / S_t2i
                # 获取当前的 batch_size
                bs = img_hash_raw.shape[0]
                # 将 u 按照 bs 展开
                u_i2t_mat = u_i2t.view(bs, bs)
                u_t2i_mat = u_t2i.view(bs, bs)
                # 提取对角线元素 (正样本对的不确定性)
                # 结果形状变为 [bs]
                u_i2t_diag = u_i2t_mat.diag()
                u_t2i_diag = u_t2i_mat.diag() 
                # 简化，将两个方向不确定性的平均值，作为着对样本总的不确定性
                uncertaint_joint = (u_i2t_diag + u_t2i_diag) / 2
                # ==================================================

                # 量化损失 - 更新
                loss_q = calculate_quantization_loss(img_hash_code, txt_hash_code)

                # 准备标签
                GND = (label @ label.T > 0).float().view(-1, 1)
                target = torch.cat([GND, 1 - GND], dim=1)
                
                lossI2t = edl_log_loss(evidencei2t, target, epoch, 2, 42)
                lossT2i = edl_log_loss(evidencet2i, target, epoch, 2, 42)
                loss_dech = lossI2t + lossT2i
                
                # # 证据联合损失，与dech损失重复
                # # joint_evidence = evidencei2t + evidencet2i
                # # loss_joint = edl_log_loss(joint_evidence, target, epoch, 2, 42)
                
                # FUME 模糊损失 (基于隶属度) ---
                img_cred = get_train_category_credibility(img_membership, label)
                txt_cred = get_train_category_credibility(txt_membership, label)
    
                # # 修正后的 RMSE 计算（逐样本）
                loss_fml_image = ((img_cred - label)**2).sum(dim=1).sqrt()
                loss_fml_txt = ((txt_cred - label)**2).sum(dim=1).sqrt()


                # # 应用贝叶斯加权（需要逐样本输入），返回标量
                # loss_bayes_img = bayesian_uncertainty_loss(loss_fml_image, u_i2t_diag, lambda_reg=0.5) 
                # loss_bayes_txt = bayesian_uncertainty_loss(loss_fml_txt, u_t2i_diag, lambda_reg=0.5) 

                # loss_aleatoric = loss_bayes_img + loss_bayes_txt
                # # 非加权版本使用样本均值作为标量
                loss_excess = (loss_fml_image + loss_fml_txt).mean()
                # loss_aleatoric_scaled = loss_aleatoric * 0.1
                
                # kl_loss_img = vib_kl_loss(mu_img, logvar_img)
                # kl_loss_txt = vib_kl_loss(mu_txt, logvar_txt)
                # loss_vib = (kl_loss_img + kl_loss_txt) / 2
                
                #  # 对比损失 (可选) ---
                # if args.use_cl:
                #     # 1. 计算权重：不确定性越大，权重越小
                #     # 策略 A: 指数衰减 (推荐)
                #     cl_weights = torch.exp(-1.0 * uncertaint_joint)
                #     # 策略 B: 反比
                #     # cl_weights = 1.0 / (1.0 + uncertainty_joint)
                    
                #     # (可选) 归一化权重，保持总损失量级稳定
                #     # cl_weights = cl_weights / cl_weights.mean()
                    
                #     # 2. 调用修改后的函数
                #     loss_cl = consistency_learning_loss(img_hash_raw, txt_hash_raw, weights=cl_weights)
                # else:
                #     loss_cl = 0.0
                
            
                
                # if args.max_epochs <= 20:
                #     warm_up = 5
                # else:
                #     warm_up = int(0.1 * args.max_epochs)
                
                # if epoch < warm_up:
                #     t = float(epoch) / float(warm_up)
                # else:
                #     t = 1.0

                # loss_fml_mix = loss_excess + (t * loss_aleatoric_scaled)


                # loss = args.alpha * loss_dech + args.delta * loss_q + args.beta * loss_fml_mix + args.gamma * loss_cl + args.eta * loss_vib


                # loss = args.alpha * loss_dech + args.delta * loss_q + args.beta * loss_excess
                loss = args.alpha * loss_dech + args.delta * loss_q
                
                            
                # 使用单个优化器
                optimizer.zero_grad()
                loss.backward()
                from torch.nn.utils.clip_grad import clip_grad_norm_

                clip_grad_norm_(unique_params, max_norm=1.0)

                optimizer.step()

                # optimizerFuzzy.step()
                
                
                train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # test(query_loader,retrieval_loader,image_model,text_model,evidence_model,epoch)
        print(f'\nEvaluating after Epoch {epoch + 1} ...')
        # MAPi2t, MAPt2i = test(query_loader, retrieval_loader, image_model, text_model, evidence_model, epoch)
        MAPi2t, MAPt2i = test(query_loader, retrieval_loader, image_model, text_model, evidence_model, gcn_img, gcn_txt, epoch)

        if MAPi2t > best_i2t_map:
            best_i2t_map = MAPi2t
            best_i2t_epoch = epoch + 1

        if MAPt2i > best_t2i_map:
            best_t2i_map = MAPt2i
            best_t2i_epoch = epoch + 1

        print(f"Epoch [{epoch + 1}/{args.max_epochs}] Finished | "
              f"MAP(i→t): {MAPi2t:.4f} | MAP(t→i): {MAPt2i:.4f} | Avg Loss: {avg_loss:.4f} ")

        # ===== 在每个 epoch 结束后保存模型 =====
        save_dir = CONFIG.model_save_dirname  # 与 main() 中一致
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'{args.data_name}_{args.bit}bit_last_{args.L3idx}')

        # 构造保存字典，结构必须与 eval_res() 中加载时一致
        state = {
            'epoch': epoch,
            'image_model_state_dict': image_model.state_dict(),
            'text_model_state_dict': text_model.state_dict(),
            'evidence_model_state_dict': evidence_model.state_dict(),
            # 'fuzzy_module_state_dict': fuzzy_module.state_dict(),
            'gcn_img_state_dict': gcn_img.state_dict(),
            'gcn_txt_state_dict': gcn_txt.state_dict(),
        }
        if epoch + 1 == args.max_epochs:
            torch.save(state, save_path)
            print(f'\nModel checkpoint saved at: {save_path}')

        # torch.save(state, save_path)
        # print(f'Model checkpoint saved at: {save_path}')

    eval_res()
    final_message = (
        f"--- Experiment Summary ---\n"
        f"Run parameters: data_name={args.data_name}, bit={args.bit}, alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, delta={args.delta},eta={args.eta},use_relu={args.use_relu},use_cl={args.use_cl}\n\n"
        f"--- Peak Performance ---\n"
        f"  - Best MAP(i→t): {best_i2t_map:.4f} (achieved at Epoch {best_i2t_epoch})\n"
        f"  - Best MAP(t→i): {best_t2i_map:.4f} (achieved at Epoch {best_t2i_epoch})\n"
        f"--------------------------"
    )
    print(final_message)
    logger1.info(final_message)

def eval_res():
    if args.data_name == 'mirflickr25k_deep':
        num_classes = 24
    elif args.data_name == 'nus_wide_deep':
        num_classes = 10
    elif args.data_name == 'mscoco_deep':
        num_classes = 80
    train_loader,query_loader,retrieval_loader,image_dim,text_dim = generate_data_laoder()
    evidence_model = EvidenceNet(args.bit,args.tau).cuda().eval()
    image_model = ImageNet(image_dim,args.bit,num_classes).cuda().eval()

    text_model = TextNet(text_dim,args.bit,num_classes).cuda().eval()
    # 新增
    gcn_img = UncertaintyPrunedGCN(in_features=args.bit, hidden_features=args.bit).cuda().eval()
    gcn_txt = UncertaintyPrunedGCN(in_features=args.bit, hidden_features=args.bit).cuda().eval()

    status_dict = torch.load(model_save_path, weights_only=True)
    image_model.load_state_dict(status_dict['image_model_state_dict'])
    text_model.load_state_dict(status_dict['text_model_state_dict'])
    evidence_model.load_state_dict(status_dict['evidence_model_state_dict'])
    # 新增
    gcn_img.load_state_dict(status_dict['gcn_img_state_dict'])
    gcn_txt.load_state_dict(status_dict['gcn_txt_state_dict'])

    # qX,qY,qL = generate_hash_code(query_loader,image_model,text_model)
    # rX,rY,rL = generate_hash_code(retrieval_loader,image_model,text_model)
    qX, qY, qL = generate_hash_code(query_loader, image_model, text_model, evidence_model, gcn_img, gcn_txt)
    rX, rY, rL = generate_hash_code(retrieval_loader, image_model, text_model, evidence_model, gcn_img, gcn_txt)

    # 修复
    qX = torch.from_numpy(qX)
    qY = torch.from_numpy(qY)
    qL = torch.from_numpy(qL)
    
    rX = torch.from_numpy(rX)
    rY = torch.from_numpy(rY)
    rL = torch.from_numpy(rL)

    epoch=status_dict['epoch']
    
    MAPi2t =fx_calc_map_multilabel_k(rY,rL,qX,qL,2000)
    MAPt2i =fx_calc_map_multilabel_k(rX,rL,qY,qL,2000)
    logger2.info(f'i2t:{MAPi2t},t2i:{MAPt2i},data_name = {args.data_name}, bit = {args.bit},epoch={epoch},L3idx={args.L3idx},tau = {args.tau}, alpha={args.alpha}, beta={args.beta}, gamma={args.gamma},delta={args.delta},eta={args.eta},use_relu={args.use_relu},use_cl={args.use_cl}')

    hash_code = {"rI":rX, "rT":rY, "rL":rL,"qI":qX, "qT":qY, "qL":qL}
    MAPi2t = fx_calc_map_multilabel_k(rY,rL,qX,qL,None)
    MAPt2i = fx_calc_map_multilabel_k(rX,rL,qY,qL,None)
    logger1.info(f'i2t:{MAPi2t},t2i:{MAPt2i},data_name = {args.data_name}, bit = {args.bit},epoch={epoch},L3idx={args.L3idx},tau = {args.tau}, alpha={args.alpha}, beta={args.beta}, gamma={args.gamma},delta={args.delta},eta={args.eta},use_relu={args.use_relu},use_cl={args.use_cl}')
    file_path = os.path.join(os.path.dirname(__file__),f'checkpoints', f'MYMETHOD_{str(args.bit)}_{args.data_name}_{args.L3idx}')
    os.makedirs(os.path.join(os.path.dirname(__file__),f'checkpoints'),exist_ok=True)
    with open(file_path,'wb') as f:
        import pickle
        pickle.dump(hash_code,f)
    print(f'eval_res save at {file_path}')
    
if __name__ == '__main__':
    # print('ds')
    if args.is_eval == 1:
        eval_res()
    else:
        main()