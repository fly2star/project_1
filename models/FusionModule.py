import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyFusion(nn.Module):
    def __init__(self, bit, dropout=0.1, num_heads=4):
        super(UncertaintyFusion, self).__init__()
        # 简单的多头注意力 (自注意力机制用于跨模态)
        # 这里为了轻量化，我们只用单层 Attention
        self.attn = nn.MultiheadAttention(embed_dim=bit, num_heads=num_heads, dropout=dropout, batch_first=True)
        # 归一化层
        self.norm = nn.LayerNorm(bit)

        # 门控网络
        self.gate_fc = nn.Linear(bit * 2 + 1, bit)
        self.sigmoid = nn.Sigmoid()

        nn.init.constant_(self.gate_fc.weight, 0)
        nn.init.constant_(self.gate_fc.bias, -5.0)

    def forward(self, x_my, x_other, u_my):
        """
        Args:
            x_my:    [Batch, Bit] 本模态特征 (e.g., Image Hash Raw)
            x_other: [Batch, Bit] 对方模态特征 (e.g., Text Hash Raw)
            u_my:    [Batch, 1]   本模态的不确定性/方差 (Proxy Var)
        """
        # 调整维度以适配 MultiheadAttention
        query = x_my.unsqueeze(1)        # [Batch, 1, Bit]
        key_value = x_other.unsqueeze(1) # [Batch, 1, Bit]

        # 执行注意力机制
        attn_output, _ = self.attn(query, key_value, key_value) # [Batch, 1, Bit]
        attn_output = attn_output.squeeze(1) # 压缩回 [Batch, Bit]

        # 准备门控输入
        # 确保 u_my 维度正确
        if u_my.dim() == 1:
            u_my = u_my.view(-1, 1) # [Batch, 1]
        
        # 拼接
        gate_input = torch.cat([x_my, attn_output, u_my], dim=1) # [Batch, 2*Bit + 1]

        # 计算融合系数 alpha
        alpha = self.sigmoid(self.gate_fc(gate_input)) # [Batch, Bit]

        # 残差融合
        fused = (1 - alpha) * x_my + alpha * attn_output # [Batch, Bit]

        # 归一化输出
        return self.norm(fused)

    
'''
class UncertaintyFusion(nn.Module):
    def __init__(self, bit, dropout=0.1):
        super(UncertaintyFusion, self).__init__()
        # 简单的多头注意力 (自注意力机制用于跨模态)
        # 这里为了轻量化，我们只用单层 Attention
        self.attn = nn.MultiheadAttention(embed_dim=bit, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(bit)
        
        # 融合门控 (Gate): 决定保留多少原始信息，接受多少跨模态信息
        self.gate = nn.Sequential(
            nn.Linear(bit * 2 + 1, bit), # 输入: [MyFeature, OtherFeature, Uncertainty]
            nn.Sigmoid()
        )

        # 门控网络
        self.gate_fc = nn.Linear(bit * 2 + 1, bit)
        self.sigmoid = nn.Sigmoid()

        nn.init.constant_(self.gate_fc.weight, 0)
        nn.init.constant_(self.gate_fc.bias, -5.0)

    def forward(self, x_my, x_other, u_my):
        """
        x_my: 当前模态特征 (e.g., Image Hash Raw)
        x_other: 另一模态特征 (e.g., Text Hash Raw)
        u_my: 当前模态的不确定性 (利用 VIB 的 logvar 计算)
        """
        # 1. 跨模态注意力 (Cross Attention)
        # Query = x_my, Key/Value = x_other
        # "我想从对方那里查询与我相关的信息"
        # x_my.unsqueeze(1) 变为 [Batch, 1, Bit] 以适配 Attention
        query = x_my.unsqueeze(1)
        key_value = x_other.unsqueeze(1)
        
        # attn_output: [Batch, 1, Bit]
        attn_output, _ = self.attn(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)
        
        # 2. 不确定性门控融合
        # 如果我不确定 (u_my 大)，我应该多听听对方的 (attn_output)
        # 如果我很确定 (u_my 小)，我应该坚持自己 (x_my)
        
        # 拼接特征用于计算门控权重
        # u_my 需要是 [Batch, 1]
        if u_my.dim() == 1:
            u_my = u_my.view(-1, 1)
            
        gate_input = torch.cat([x_my, attn_output, u_my], dim=1)
        alpha = self.gate(gate_input) # [Batch, Bit]
        
        # 3. 残差连接 + 归一化
        # Out = (1-alpha) * Self + alpha * Other
        fused = (1 - alpha) * x_my + alpha * attn_output
        return self.norm(fused)
'''