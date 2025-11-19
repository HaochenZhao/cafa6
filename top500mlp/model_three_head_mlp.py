"""
三头MLP模型：基于ESM2嵌入，每个aspect一个独立的MLP头
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class ThreeHeadMLP(nn.Module):
    """
    三头MLP模型架构

    结构：
        ESM2 Embeddings (frozen)
            -> 共享层 (可选)
            -> 三个独立的MLP头 (C/F/P)
    """

    def __init__(
        self,
        esm_dim: int = 320,  # ESM2-8M 的嵌入维度
        hidden_dims: list = [512, 256],  # MLP隐藏层维度
        num_labels_per_aspect: Dict[str, int] = None,  # 每个aspect的标签数
        dropout: float = 0.3,
        use_shared_layer: bool = True,  # 是否使用共享层
        activation: str = 'relu'
    ):
        super().__init__()

        if num_labels_per_aspect is None:
            num_labels_per_aspect = {'C': 500, 'F': 500, 'P': 500}

        self.esm_dim = esm_dim
        self.num_labels_per_aspect = num_labels_per_aspect
        self.use_shared_layer = use_shared_layer

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 可选的共享层
        if use_shared_layer:
            self.shared_layer = nn.Sequential(
                nn.Linear(esm_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                self.activation,
                nn.Dropout(dropout)
            )
            input_dim = hidden_dims[0]
        else:
            self.shared_layer = None
            input_dim = esm_dim

        # 三个独立的MLP头
        self.heads = nn.ModuleDict({
            'C': self._build_head(input_dim, hidden_dims, num_labels_per_aspect['C'], dropout),
            'F': self._build_head(input_dim, hidden_dims, num_labels_per_aspect['F'], dropout),
            'P': self._build_head(input_dim, hidden_dims, num_labels_per_aspect['P'], dropout)
        })

    def _build_head(
        self,
        input_dim: int,
        hidden_dims: list,
        num_labels: int,
        dropout: float
    ) -> nn.Module:
        """构建单个MLP头"""
        layers = []

        # 如果使用了共享层，跳过第一个hidden_dim
        start_idx = 1 if self.use_shared_layer else 0
        prev_dim = input_dim

        # 隐藏层
        for dim in hidden_dims[start_idx:]:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_labels))

        return nn.Sequential(*layers)

    def forward(
        self,
        embeddings: torch.Tensor,
        return_separate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            embeddings: ESM2嵌入 [batch_size, esm_dim]
            return_separate: 是否返回分离的logits（用于分别计算loss）

        返回:
            如果return_separate=True:
                {'C': logits_C, 'F': logits_F, 'P': logits_P}
            否则:
                {'logits': 拼接后的logits [batch_size, 1500]}
        """
        # 共享层
        if self.shared_layer is not None:
            shared_features = self.shared_layer(embeddings)
        else:
            shared_features = embeddings

        # 三个头的输出
        outputs = {
            'C': self.heads['C'](shared_features),
            'F': self.heads['F'](shared_features),
            'P': self.heads['P'](shared_features)
        }

        if return_separate:
            return outputs
        else:
            # 拼接成完整的输出 [batch_size, 1500]
            logits = torch.cat([outputs['C'], outputs['F'], outputs['P']], dim=1)
            return {'logits': logits}

    def get_num_params(self) -> Dict[str, int]:
        """获取参数数量统计"""
        stats = {}

        if self.shared_layer is not None:
            stats['shared'] = sum(p.numel() for p in self.shared_layer.parameters())

        for aspect in ['C', 'F', 'P']:
            stats[f'head_{aspect}'] = sum(p.numel() for p in self.heads[aspect].parameters())

        stats['total'] = sum(p.numel() for p in self.parameters())

        return stats


class ESM2WithThreeHeadMLP(nn.Module):
    """
    完整模型：ESM2 + 三头MLP
    """

    def __init__(
        self,
        esm_model_name: str = "esm2_t6_8M_UR50D",  # ESM2-8M
        num_labels_per_aspect: Dict[str, int] = None,
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        use_shared_layer: bool = True,
        freeze_esm: bool = True,  # 是否冻结ESM2
        pooling: str = 'mean'  # 'mean' or 'cls'
    ):
        super().__init__()

        if num_labels_per_aspect is None:
            num_labels_per_aspect = {'C': 500, 'F': 500, 'P': 500}

        # 加载ESM2模型
        try:
            import esm
            self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
            self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
            esm_dim = self.esm_model.embed_dim
        except Exception as e:
            raise RuntimeError(f"无法加载ESM2模型: {e}\n请确保已安装fair-esm: pip install fair-esm")

        # 冻结ESM2参数
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
            self.esm_model.eval()

        self.freeze_esm = freeze_esm
        self.pooling = pooling

        # 三头MLP
        self.mlp = ThreeHeadMLP(
            esm_dim=esm_dim,
            hidden_dims=hidden_dims,
            num_labels_per_aspect=num_labels_per_aspect,
            dropout=dropout,
            use_shared_layer=use_shared_layer
        )

    def get_esm_embeddings(
        self,
        sequences: list,
        device: torch.device
    ) -> torch.Tensor:
        """
        获取ESM2嵌入

        参数:
            sequences: 蛋白质序列列表
            device: 设备

        返回:
            embeddings: [batch_size, esm_dim]
        """
        # 准备批次数据
        batch_labels = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(batch_labels)
        batch_tokens = batch_tokens.to(device)

        # ESM2前向传播
        if self.freeze_esm:
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[self.esm_model.num_layers])
        else:
            results = self.esm_model(batch_tokens, repr_layers=[self.esm_model.num_layers])

        # 提取表示
        token_representations = results["representations"][self.esm_model.num_layers]

        # 池化
        if self.pooling == 'mean':
            # 平均池化（不包括特殊token）
            embeddings = []
            for i, seq_len in enumerate([len(seq) for seq in batch_strs]):
                # token_representations[i, 1:seq_len+1] 不包括<cls>和<eos>
                embeddings.append(token_representations[i, 1:seq_len+1].mean(0))
            embeddings = torch.stack(embeddings)
        elif self.pooling == 'cls':
            # 使用<cls> token
            embeddings = token_representations[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return embeddings

    def forward(
        self,
        sequences: list,
        return_separate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            sequences: 蛋白质序列列表
            return_separate: 是否返回分离的logits

        返回:
            模型输出字典
        """
        device = next(self.parameters()).device

        # 获取ESM2嵌入
        embeddings = self.get_esm_embeddings(sequences, device)

        # MLP前向传播
        outputs = self.mlp(embeddings, return_separate=return_separate)

        return outputs

    def train(self, mode: bool = True):
        """重写train方法，确保ESM2保持frozen"""
        super().train(mode)
        if self.freeze_esm:
            self.esm_model.eval()
        return self

    def get_num_params(self) -> Dict[str, int]:
        """获取参数数量统计"""
        stats = self.mlp.get_num_params()

        esm_params = sum(p.numel() for p in self.esm_model.parameters())
        esm_trainable = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)

        stats['esm_total'] = esm_params
        stats['esm_trainable'] = esm_trainable
        stats['mlp_total'] = stats['total']
        stats['total'] = esm_params + stats['mlp_total']
        stats['total_trainable'] = esm_trainable + stats['mlp_total']

        return stats


if __name__ == "__main__":
    # 测试模型
    print("="*70)
    print("测试三头MLP模型")
    print("="*70)

    # 仅测试MLP部分
    print("\n1. 测试MLP模块:")
    mlp = ThreeHeadMLP(
        esm_dim=320,
        hidden_dims=[512, 256],
        num_labels_per_aspect={'C': 500, 'F': 500, 'P': 500},
        dropout=0.3,
        use_shared_layer=True
    )

    # 模拟输入
    batch_size = 4
    fake_embeddings = torch.randn(batch_size, 320)

    # 测试前向传播
    outputs_combined = mlp(fake_embeddings, return_separate=False)
    print(f"  合并输出形状: {outputs_combined['logits'].shape}")  # 应该是 [4, 1500]

    outputs_separate = mlp(fake_embeddings, return_separate=True)
    print(f"  分离输出形状:")
    for aspect in ['C', 'F', 'P']:
        print(f"    {aspect}: {outputs_separate[aspect].shape}")  # 应该是 [4, 500]

    # 参数统计
    print(f"\n  参数统计:")
    params = mlp.get_num_params()
    for key, value in params.items():
        print(f"    {key}: {value:,}")

    print("\n模型测试完成！")
