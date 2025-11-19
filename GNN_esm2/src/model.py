"""
Pairwise评分模型
"""

import torch
import torch.nn as nn
from typing import List


class PairwiseScorer(nn.Module):
    """
    Pairwise评分模型
    
    输入: (protein_emb, go_emb)
    输出: score (0-1)
    """
    
    def __init__(
        self,
        esm_dim: int = 320,
        go_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        fusion_type: str = "concat"
    ):
        """
        参数:
            esm_dim: ESM2 embedding维度
            go_dim: GO embedding维度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比率
            fusion_type: 融合方式 (concat/attention/bilinear)
        """
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
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, protein_embs: torch.Tensor, go_embs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            protein_embs: [batch, esm_dim] 或 [batch, num_gos, esm_dim]
            go_embs: [batch, go_dim] 或 [batch, num_gos, go_dim]
        
        返回:
            scores: [batch] 或 [batch, num_gos]
        """
        # 融合
        if self.fusion_type == "concat":
            fused = torch.cat([protein_embs, go_embs], dim=-1)
        elif self.fusion_type == "attention":
            # 使用attention融合
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



