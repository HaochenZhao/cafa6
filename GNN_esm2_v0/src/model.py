"""
Pairwise Scoring模型
输入: (seq_embedding, go_embedding)
输出: score (0-1)
"""

import torch
import torch.nn as nn
from typing import Dict


class MLPFusion(nn.Module):
    """MLP融合模块"""
    
    def __init__(
        self,
        seq_dim: int,
        go_dim: int,
        hidden_dims: list,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        
        # 构建MLP
        layers = []
        input_dim = seq_dim + go_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, seq_emb: torch.Tensor, go_emb: torch.Tensor) -> torch.Tensor:
        """
        参数:
            seq_emb: [batch_size, seq_dim]
            go_emb: [batch_size, go_dim]
        
        返回:
            scores: [batch_size, 1]
        """
        concat = torch.cat([seq_emb, go_emb], dim=-1)
        return self.mlp(concat)


class BilinearFusion(nn.Module):
    """Bilinear融合模块"""
    
    def __init__(
        self,
        seq_dim: int,
        go_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 双线性层
        self.bilinear = nn.Bilinear(seq_dim, go_dim, hidden_dim)
        
        # 后续MLP
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, seq_emb: torch.Tensor, go_emb: torch.Tensor) -> torch.Tensor:
        """
        参数:
            seq_emb: [batch_size, seq_dim]
            go_emb: [batch_size, go_dim]
        
        返回:
            scores: [batch_size, 1]
        """
        bilinear_out = self.bilinear(seq_emb, go_emb)
        return self.mlp(bilinear_out)


class AttentionFusion(nn.Module):
    """Attention融合模块"""
    
    def __init__(
        self,
        seq_dim: int,
        go_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 投影到相同维度
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.go_proj = nn.Linear(go_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, seq_emb: torch.Tensor, go_emb: torch.Tensor) -> torch.Tensor:
        """
        参数:
            seq_emb: [batch_size, seq_dim]
            go_emb: [batch_size, go_dim]
        
        返回:
            scores: [batch_size, 1]
        """
        # 投影
        seq_proj = self.seq_proj(seq_emb).unsqueeze(1)  # [batch, 1, hidden]
        go_proj = self.go_proj(go_emb).unsqueeze(1)     # [batch, 1, hidden]
        
        # Attention (seq as query, go as key/value)
        attn_out, _ = self.attention(seq_proj, go_proj, go_proj)  # [batch, 1, hidden]
        attn_out = attn_out.squeeze(1)  # [batch, hidden]
        
        return self.mlp(attn_out)


class PairwiseScoringModel(nn.Module):
    """
    Pairwise Scoring模型
    对每个(sequence, go_term)对打分
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # 获取维度
        esm_model_name = config['esm2']['model_name']
        seq_dim = config['model']['esm_dim_map'][esm_model_name]
        go_dim = config['go_gnn']['embedding_dim']
        
        # 选择融合模块
        fusion_type = config['model']['fusion']['type']
        
        if fusion_type == 'mlp':
            self.fusion = MLPFusion(
                seq_dim=seq_dim,
                go_dim=go_dim,
                hidden_dims=config['model']['fusion']['hidden_dims'],
                dropout=config['model']['fusion']['dropout'],
                activation=config['model']['fusion']['activation']
            )
        elif fusion_type == 'bilinear':
            self.fusion = BilinearFusion(
                seq_dim=seq_dim,
                go_dim=go_dim,
                hidden_dim=config['model']['fusion']['hidden_dims'][0],
                dropout=config['model']['fusion']['dropout']
            )
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(
                seq_dim=seq_dim,
                go_dim=go_dim,
                hidden_dim=config['model']['fusion']['hidden_dims'][0],
                num_heads=4,
                dropout=config['model']['fusion']['dropout']
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        print(f"✓ 创建Pairwise模型:")
        print(f"  融合方式: {fusion_type}")
        print(f"  序列维度: {seq_dim}")
        print(f"  GO维度: {go_dim}")
    
    def forward(self, seq_embs: torch.Tensor, go_embs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            seq_embs: [batch_size, seq_dim]
            go_embs: [batch_size, go_dim]
        
        返回:
            scores: [batch_size, 1] (logits)
        """
        return self.fusion(seq_embs, go_embs)
    
    def predict(self, seq_embs: torch.Tensor, go_embs: torch.Tensor) -> torch.Tensor:
        """
        预测（带sigmoid）
        
        参数:
            seq_embs: [batch_size, seq_dim]
            go_embs: [batch_size, go_dim]
        
        返回:
            probs: [batch_size, 1] (probabilities)
        """
        logits = self.forward(seq_embs, go_embs)
        return torch.sigmoid(logits)
    
    def get_num_params(self) -> Dict[str, int]:
        """获取参数统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'fusion': sum(p.numel() for p in self.fusion.parameters())
        }


if __name__ == "__main__":
    # 测试模型
    import yaml
    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = PairwiseScoringModel(config)
    
    # 测试前向传播
    batch_size = 16
    seq_dim = config['model']['esm_dim_map'][config['esm2']['model_name']]
    go_dim = config['go_gnn']['embedding_dim']
    
    seq_embs = torch.randn(batch_size, seq_dim)
    go_embs = torch.randn(batch_size, go_dim)
    
    scores = model(seq_embs, go_embs)
    probs = model.predict(seq_embs, go_embs)
    
    print(f"\n测试:")
    print(f"  输入形状: seq {seq_embs.shape}, go {go_embs.shape}")
    print(f"  Logits形状: {scores.shape}")
    print(f"  Probs形状: {probs.shape}")
    print(f"  Probs范围: [{probs.min():.3f}, {probs.max():.3f}]")
    
    params = model.get_num_params()
    print(f"\n参数统计:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
