"""
Pairwise评分模型 (支持 LayerNorm + 可配置 Bias Initialization)
"""

import torch
import torch.nn as nn
import math
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
        fusion_type: str = "concat",
        use_bias_init: bool = True  # [Ablation] 控制 Bias Init
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.use_bias_init = use_bias_init
        
        # 1. 定义融合层
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
        
        # [关键] LayerNorm 层
        # 在送入MLP之前进行归一化，确保特征值分布稳定
        self.layernorm = nn.LayerNorm(fusion_dim)
        
        # 2. 构建 MLP
        layers = []
        in_dim = fusion_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # 输出层 (Linear, 无 Sigmoid)
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

        # 3. 执行初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        权重初始化策略
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 默认使用 Kaiming Normal
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # [Ablation] Bias Initialization
        if self.use_bias_init:
            last_layer = self.mlp[-1]
            if isinstance(last_layer, nn.Linear):
                # 设定先验概率 pi = 0.01 (即假设正样本只有 1%)
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                
                # 初始化 Bias 为负值
                nn.init.constant_(last_layer.bias, bias_value)
                # 初始化 Weight 为极小值
                nn.init.normal_(last_layer.weight, std=0.01)
                
                print(f"✓ [Model] Bias Initialization 启用: Bias={bias_value:.4f} (Prior={prior_prob})")
        else:
            print("✓ [Model] Bias Initialization 禁用 (使用标准初始化)")

    def forward(self, protein_embs: torch.Tensor, go_embs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        返回 Logits (未经过 Sigmoid)
        """
        # 1. 特征融合
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
        
        # 2. [关键] LayerNorm 归一化
        fused = self.layernorm(fused)
        
        # 3. MLP 预测
        scores = self.mlp(fused).squeeze(-1)
        
        return scores

    def predict(self, protein_embs: torch.Tensor, go_embs: torch.Tensor) -> torch.Tensor:
        """预测概率 (0-1)"""
        logits = self.forward(protein_embs, go_embs)
        return torch.sigmoid(logits)