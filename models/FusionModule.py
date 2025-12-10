class SimpleFusionModule(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleFusionModule, self).__init__()
        # 使用一个简单的 MLP 来学习三个特征的融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2) # 输出融合后的 img 和 txt 特征
        )
    
    def forward(self, img_feat, txt_feat, prompt_feat):
        # 1. 拼接所有输入特征
        concatenated_features = torch.cat((img_feat, txt_feat, prompt_feat), dim=-1)
        
        # 2. 通过 MLP 进行深度融合
        fused_output = self.fusion_mlp(concatenated_features)
        
        # 3. 分割成增强后的图像和文本特征
        enhanced_img_feat, enhanced_txt_feat = torch.chunk(fused_output, 2, dim=-1)
        
        # 4. 使用残差连接
        final_img_feat = 0.5 * img_feat + 0.5 * enhanced_img_feat
        final_txt_feat = 0.5 * txt_feat + 0.5 * enhanced_txt_feat
        
        return final_img_feat, final_txt_feat