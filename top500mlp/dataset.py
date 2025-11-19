"""
CAFA6数据集加载器
支持预计算ESM2嵌入和动态计算两种模式
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import pickle


class CAFA6Dataset(Dataset):
    """
    CAFA6数据集

    支持两种模式:
    1. 动态模式：返回序列，由模型计算ESM2嵌入
    2. 预计算模式：返回预先计算好的ESM2嵌入
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        protein_ids: np.ndarray,
        go_aspects: np.ndarray,
        use_precomputed_embeddings: bool = False,
        embeddings: Optional[np.ndarray] = None
    ):
        """
        参数:
            sequences: 蛋白质序列数组
            labels: 标签矩阵 [num_samples, 1500]
            protein_ids: 蛋白质ID
            go_aspects: GO术语的aspect信息 [1500]
            use_precomputed_embeddings: 是否使用预计算的嵌入
            embeddings: 预计算的嵌入 [num_samples, esm_dim]
        """
        self.sequences = sequences
        self.labels = labels
        self.protein_ids = protein_ids
        self.go_aspects = go_aspects
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.embeddings = embeddings

        if use_precomputed_embeddings and embeddings is None:
            raise ValueError("use_precomputed_embeddings=True 但未提供embeddings")

        # 创建aspect到索引的映射
        self.aspect_indices = {
            'C': np.where(go_aspects == 'C')[0],
            'F': np.where(go_aspects == 'F')[0],
            'P': np.where(go_aspects == 'P')[0]
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            'protein_id': self.protein_ids[idx],
            'labels': torch.FloatTensor(self.labels[idx]),
        }

        if self.use_precomputed_embeddings:
            item['embedding'] = torch.FloatTensor(self.embeddings[idx])
        else:
            item['sequence'] = self.sequences[idx]

        # 分离各aspect的标签
        item['labels_C'] = torch.FloatTensor(self.labels[idx, self.aspect_indices['C']])
        item['labels_F'] = torch.FloatTensor(self.labels[idx, self.aspect_indices['F']])
        item['labels_P'] = torch.FloatTensor(self.labels[idx, self.aspect_indices['P']])

        return item


def collate_fn_dynamic(batch):
    """
    动态模式的collate函数（用于动态计算ESM2嵌入）
    """
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    labels_C = torch.stack([item['labels_C'] for item in batch])
    labels_F = torch.stack([item['labels_F'] for item in batch])
    labels_P = torch.stack([item['labels_P'] for item in batch])
    protein_ids = [item['protein_id'] for item in batch]

    return {
        'sequences': sequences,
        'labels': labels,
        'labels_C': labels_C,
        'labels_F': labels_F,
        'labels_P': labels_P,
        'protein_ids': protein_ids
    }


def collate_fn_precomputed(batch):
    """
    预计算模式的collate函数（使用预先计算的嵌入）
    """
    embeddings = torch.stack([item['embedding'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    labels_C = torch.stack([item['labels_C'] for item in batch])
    labels_F = torch.stack([item['labels_F'] for item in batch])
    labels_P = torch.stack([item['labels_P'] for item in batch])
    protein_ids = [item['protein_id'] for item in batch]

    return {
        'embeddings': embeddings,
        'labels': labels,
        'labels_C': labels_C,
        'labels_F': labels_F,
        'labels_P': labels_P,
        'protein_ids': protein_ids
    }


def load_cafa6_data(data_dir: str = "processed_data_v3") -> Dict:
    """
    加载CAFA6数据

    返回:
        包含train, val, test数据和元数据的字典
    """
    import os

    npz_file = os.path.join(data_dir, "cafa6_data.npz")
    pkl_file = os.path.join(data_dir, "cafa6_metadata.pkl")

    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"数据文件不存在: {npz_file}")

    # 读取主数据
    print(f"读取数据: {npz_file}")
    npz_data = np.load(npz_file, allow_pickle=True)

    # 读取元数据
    metadata = {}
    if os.path.exists(pkl_file):
        print(f"读取元数据: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            metadata = pickle.load(f)

    data = {
        'train': {
            'sequences': npz_data['train_sequences'],
            'labels': npz_data['train_labels'],
            'ids': npz_data['train_ids']
        },
        'val': {
            'sequences': npz_data['val_sequences'],
            'labels': npz_data['val_labels'],
            'ids': npz_data['val_ids']
        },
        'test': {
            'sequences': npz_data['test_sequences'],
            'ids': npz_data['test_ids']
        },
        'go_terms': npz_data['go_terms'].tolist(),
        'ia_weights': npz_data['ia_weights'],
        'go_aspects': npz_data['go_aspects'],
        'metadata': metadata
    }

    return data


def create_dataloaders(
    data_dir: str = "processed_data_v3",
    batch_size: int = 8,
    num_workers: int = 4,
    use_precomputed_embeddings: bool = False,
    train_embeddings: Optional[np.ndarray] = None,
    val_embeddings: Optional[np.ndarray] = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证数据加载器

    参数:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        use_precomputed_embeddings: 是否使用预计算的嵌入
        train_embeddings: 训练集的预计算嵌入
        val_embeddings: 验证集的预计算嵌入

    返回:
        (train_loader, val_loader, metadata)
    """
    # 加载数据
    data = load_cafa6_data(data_dir)

    # 创建数据集
    train_dataset = CAFA6Dataset(
        sequences=data['train']['sequences'],
        labels=data['train']['labels'],
        protein_ids=data['train']['ids'],
        go_aspects=data['go_aspects'],
        use_precomputed_embeddings=use_precomputed_embeddings,
        embeddings=train_embeddings
    )

    val_dataset = CAFA6Dataset(
        sequences=data['val']['sequences'],
        labels=data['val']['labels'],
        protein_ids=data['val']['ids'],
        go_aspects=data['go_aspects'],
        use_precomputed_embeddings=use_precomputed_embeddings,
        embeddings=val_embeddings
    )

    # 选择collate函数
    collate_fn = collate_fn_precomputed if use_precomputed_embeddings else collate_fn_dynamic

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    metadata = {
        'go_terms': data['go_terms'],
        'ia_weights': data['ia_weights'],
        'go_aspects': data['go_aspects'],
        'num_labels': len(data['go_terms']),
        'num_labels_per_aspect': {
            'C': (data['go_aspects'] == 'C').sum(),
            'F': (data['go_aspects'] == 'F').sum(),
            'P': (data['go_aspects'] == 'P').sum()
        }
    }

    print(f"\n数据加载完成:")
    print(f"  训练集: {len(train_dataset):,} 样本")
    print(f"  验证集: {len(val_dataset):,} 样本")
    print(f"  GO术语: {metadata['num_labels']} (C: {metadata['num_labels_per_aspect']['C']}, "
          f"F: {metadata['num_labels_per_aspect']['F']}, P: {metadata['num_labels_per_aspect']['P']})")
    print(f"  批次大小: {batch_size}")
    print(f"  模式: {'预计算嵌入' if use_precomputed_embeddings else '动态计算'}")

    return train_loader, val_loader, metadata


if __name__ == "__main__":
    # 测试数据加载
    print("="*70)
    print("测试数据加载器")
    print("="*70)

    try:
        train_loader, val_loader, metadata = create_dataloaders(
            data_dir="processed_data_v3",
            batch_size=4,
            num_workers=0,  # 测试时使用0
            use_precomputed_embeddings=False
        )

        # 测试一个批次
        print("\n测试一个批次:")
        batch = next(iter(train_loader))
        print(f"  序列数: {len(batch['sequences'])}")
        print(f"  标签形状: {batch['labels'].shape}")
        print(f"  C标签形状: {batch['labels_C'].shape}")
        print(f"  F标签形状: {batch['labels_F'].shape}")
        print(f"  P标签形状: {batch['labels_P'].shape}")
        print(f"  第一个序列长度: {len(batch['sequences'][0])}")
        print(f"  第一个序列前50个字符: {batch['sequences'][0][:50]}")

        print("\n数据加载测试完成！")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 save_processed_data_v3.py 生成数据")
