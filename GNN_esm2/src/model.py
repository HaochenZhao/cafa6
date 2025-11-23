"""
Pairwise评分模型 (修改版: 优化权重初始化)
"""

import torch
import torch.nn as nn
import math  # [新增]
from typing import List


class PairwiseScorer(nn.Module):
    """
    Pairwise评分模型
    
    输入: (protein_emb, go_emb)
    输出: Logits (范围 -inf 到 +inf)
    """
    
    def __init__(
        self,
        esm_dim: int = 320,
        go_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        fusion_type: str = "concat"
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # 融合层
        if fusion_type == "concat":
            fusion_dim = esm_dim + go_dim
        elif fusion_type == "attention":
            fusion_dim = esm_dim + go_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=esm_dim,
                num_heads=4,
                batch_first=True
            )
        elif fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(esm_dim, go_dim, hidden_dims[0])
            fusion_dim = hidden_dims[0]
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
        
        # MLP
        layers = []
        in_dim = fusion_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # 输出层
        # [注意] 最后一层是 Linear，直接输出 Logits，没有 Sigmoid
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

        # [新增] 执行自定义初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        自定义初始化策略：
        1. 隐藏层：Kaiming 初始化 (适合 ReLU)
        2. 输出层：Prior Bias 初始化 (适合极度不平衡数据)
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对所有 Linear 层使用 Kaiming Normal
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # === 特殊处理输出层 ===
        # 找到 MLP 的最后一层 (Linear)
        # 这里的 -1 是因为最后一层就是 Linear (没有 Sigmoid)
        last_layer = self.mlp[-1]
        
        if isinstance(last_layer, nn.Linear):
            # 设定先验概率 pi = 0.01 (即假设正样本只有 1%)
            # 对于 CAFA 这种 1:3000 的数据，0.01 是一个比较安全的起点，既压低了分数，又保留了梯度
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            
            # 初始化 Bias
            nn.init.constant_(last_layer.bias, bias_value)
            
            # 将权重初始化得非常小，确保初始输出主要由 Bias 决定
            nn.init.normal_(last_layer.weight, std=0.01)
            
            print(f"✓ 模型初始化完成: 输出层 Bias 设为 {bias_value:.4f} (Prior={prior_prob})")

    def forward(self, protein_embs: torch.Tensor, go_embs: torch.Tensor) -> torch.Tensor:
        # 融合
        if self.fusion_type == "concat":
            fused = torch.cat([protein_embs, go_embs], dim=-1)
        elif self.fusion_type == "attention":
            protein_attended, _ = self.attention(
                protein_embs.unsqueeze(1),
                go_embs.unsqueeze(1),
                go_embs.unsqueeze(1)
            )
            fused = torch.cat([protein_attended.squeeze(1), go_embs], dim=-1)
        elif self.fusion_type == "bilinear":
            fused = self.bilinear(protein_embs, go_embs)
        
        # MLP
        scores = self.mlp(fused).squeeze(-1)
        
        return scores

    def predict(self, protein_embs: torch.Tensor, go_embs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(protein_embs, go_embs)
        return torch.sigmoid(logits)