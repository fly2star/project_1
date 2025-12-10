# file: models/FuzzyLogicModule.py

import torch
import torch.nn as nn

class FuzzyLogicModule(nn.Module):
    def __init__(self, feature_dim, num_classes, device, use_relu=False): # <-- 新增 use_relu 参数
        super(FuzzyLogicModule, self).__init__()
        # 1. 初始化可学习的类别中心 W
        W_tensor = torch.Tensor(feature_dim, num_classes)
        self.W = nn.Parameter(torch.nn.init.orthogonal_(W_tensor, gain=1).to(device))
        
        # 2. 保存 use_relu 选项
        self.use_relu = use_relu
        if self.use_relu:
            print("--- FuzzyLogicModule initialized WITH ReLU on membership degree. ---")
        else:
            print("--- FuzzyLogicModule initialized WITHOUT ReLU on membership degree. ---")

    def get_train_category_credibility(self, predict, labels):
        # 复制自 processor.py
        top1Possibility = (predict * (1 - labels)).max(1)[0].reshape([-1, 1])
        labelPossibility = (predict * labels).max(1)[0].reshape([-1, 1])
        neccessity = (1 - labelPossibility) * (1 - labels) + (1 - top1Possibility) * labels
        r = (predict + neccessity) / 2
        return r

    def forward(self, view1_feature, view2_feature, labels):
        # view1_feature, view2_feature 是来自 DECH 编码器的输出
        
        # 1. 动态单位化 W
        W_normalized = self.W / torch.norm(self.W, p=2, dim=0, keepdim=True)

        # 2. 计算原始内积 (predict)
        view1_predict = view1_feature.mm(W_normalized)
        view2_predict = view2_feature.mm(W_normalized)
        
        # --- 核心修改：根据 use_relu 选项决定是否使用 ReLU ---
        if self.use_relu:
            # 方案二：完全模仿 FUME，使用 ReLU
            predict_for_credibility_calc_v1 = torch.relu(view1_predict)
            predict_for_credibility_calc_v2 = torch.relu(view2_predict)
        else:
            # 方案一：理论更自洽，不使用 ReLU
            predict_for_credibility_calc_v1 = view1_predict
            predict_for_credibility_calc_v2 = view2_predict
        # --------------------------------------------------------

        # 3. 计算训练时的可信度
        view1_cred = self.get_train_category_credibility(predict_for_credibility_calc_v1, labels)
        view2_cred = self.get_train_category_credibility(predict_for_credibility_calc_v2, labels)

        # 4. 计算 loss_fml (模糊多模态学习损失)
        loss_fml = ((view1_cred - labels.float())**2).sum(1).sqrt().mean() + \
                   ((view2_cred - labels.float())**2).sum(1).sqrt().mean()

        return loss_fml