"""
Pairwise数据集 - 每个样本是(sequence, go_term)对
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import pickle
from collections import defaultdict


class PairwiseDataset(Dataset):
    """
    Pairwise数据集
    每个样本: (protein_id, go_id) -> label (0/1)
    返回: (seq_embedding, go_embedding, label)
    """
    
    def __init__(
        self,
        protein_ids: np.ndarray,
        seq_embeddings: np.ndarray,
        go_embeddings: np.ndarray,
        go_metadata: dict,
        labels_dict: Dict[str, List[str]],  # {protein_id: [go_ids]}
        sampling_strategy: str = "all",
        negative_ratio: int = 5,
        is_training: bool = True
    ):
        """
        参数:
            protein_ids: 蛋白质ID数组
            seq_embeddings: 序列embeddings [num_proteins, esm_dim]
            go_embeddings: GO embeddings [num_gos, go_dim]
            go_metadata: GO元数据（包含go_id_to_idx映射）
            labels_dict: {protein_id: [positive_go_ids]}
            sampling_strategy: 采样策略 (all, negative_sampling, mixed)
            negative_ratio: 负样本比例
            is_training: 是否训练模式
        """
        self.protein_ids = protein_ids
        self.seq_embeddings = seq_embeddings
        self.go_embeddings = go_embeddings
        self.go_metadata = go_metadata
        self.labels_dict = labels_dict
        self.sampling_strategy = sampling_strategy
        self.negative_ratio = negative_ratio
        self.is_training = is_training
        
        # 构建GO ID列表和映射
        self.go_id_to_idx = go_metadata['go_id_to_idx']
        self.idx_to_go_id = go_metadata['idx_to_go_id']
        self.all_go_ids = list(self.go_id_to_idx.keys())
        
        # 检查GO数量
        num_gos = len(self.all_go_ids)
        num_proteins = len(protein_ids)
        
        print(f"\n  数据集参数:")
        print(f"    蛋白质数: {num_proteins:,}")
        print(f"    GO terms数: {num_gos:,}")
        print(f"    采样策略: {sampling_strategy}")
        
        # 警告：GO数量异常
        if num_gos > 30000:
            print(f"  ⚠️  警告: GO数量过多 ({num_gos:,})！")
            print(f"      这可能是使用了GO GNN的全部GO而非processed数据的GO")
            print(f"      建议检查go_embeddings是否与processed_data匹配")
            
            if sampling_strategy == "all" or sampling_strategy == "mixed":
                print(f"  ⚠️  强烈建议: 使用 'negative_sampling' 策略")
                print(f"      否则内存可能不足！")
        
        # 为训练集构建样本对
        if is_training:
            self.pairs = self._build_training_pairs()
        else:
            # 验证/测试时，遍历所有(protein, go)组合
            self.pairs = self._build_all_pairs()
        
        print(f"  ✓ 数据集构建完成: {len(self.pairs):,} 样本对")
    
    def _build_training_pairs(self) -> List[Tuple[str, str, int]]:
        """构建训练样本对 - 内存优化版本"""
        pairs = []
        
        print(f"  构建训练样本对...")
        print(f"  蛋白质数: {len(self.protein_ids)}")
        print(f"  GO terms数: {len(self.all_go_ids)}")
        
        # 预估样本对数量
        estimated_pairs = 0
        for protein_id in self.protein_ids[:100]:  # 采样估计
            positive_gos = set(self.labels_dict.get(protein_id, []))
            if self.sampling_strategy == "all":
                estimated_pairs += len(self.all_go_ids)
            elif self.sampling_strategy == "negative_sampling":
                estimated_pairs += len(positive_gos) * (1 + self.negative_ratio)
            elif self.sampling_strategy == "mixed":
                if len(positive_gos) < 20:
                    estimated_pairs += len(self.all_go_ids)
                else:
                    estimated_pairs += len(positive_gos) * (1 + min(self.negative_ratio, 1000 // len(positive_gos)))
        
        estimated_total = (estimated_pairs // 100) * len(self.protein_ids)
        print(f"  预估样本对数: {estimated_total:,}")
        
        if estimated_total > 100_000_000:  # 1亿
            print(f"  ⚠️  警告: 样本对数量过大，建议使用negative_sampling策略")
        
        # 分批构建，避免内存峰值
        batch_size = 1000
        for i in range(0, len(self.protein_ids), batch_size):
            batch_proteins = self.protein_ids[i:i+batch_size]
            
            for protein_id in batch_proteins:
                positive_gos = set(self.labels_dict.get(protein_id, []))
                
                if self.sampling_strategy == "all":
                    # 所有GO terms
                    for go_id in self.all_go_ids:
                        label = 1 if go_id in positive_gos else 0
                        pairs.append((protein_id, go_id, label))
                
                elif self.sampling_strategy == "negative_sampling":
                    # 所有正样本 + 采样负样本
                    for go_id in positive_gos:
                        if go_id in self.go_id_to_idx:
                            pairs.append((protein_id, go_id, 1))
                    
                    # 采样负样本
                    num_negatives = len(positive_gos) * self.negative_ratio
                    negative_gos = set(self.all_go_ids) - positive_gos
                    
                    if len(negative_gos) > 0:
                        sampled_negatives = np.random.choice(
                            list(negative_gos),
                            size=min(num_negatives, len(negative_gos)),
                            replace=False
                        )
                        for go_id in sampled_negatives:
                            pairs.append((protein_id, go_id, 0))
                
                elif self.sampling_strategy == "mixed":
                    # 混合策略：部分全样本，部分采样
                    if len(positive_gos) < 20:
                        # 正样本少，用全样本
                        for go_id in self.all_go_ids:
                            label = 1 if go_id in positive_gos else 0
                            pairs.append((protein_id, go_id, label))
                    else:
                        # 正样本多，用采样
                        for go_id in positive_gos:
                            if go_id in self.go_id_to_idx:
                                pairs.append((protein_id, go_id, 1))
                        
                        num_negatives = min(len(positive_gos) * self.negative_ratio, 1000)
                        negative_gos = set(self.all_go_ids) - positive_gos
                        
                        if len(negative_gos) > 0:
                            sampled_negatives = np.random.choice(
                                list(negative_gos),
                                size=min(num_negatives, len(negative_gos)),
                                replace=False
                            )
                            for go_id in sampled_negatives:
                                pairs.append((protein_id, go_id, 0))
            
            # 每处理1000个蛋白质打印进度
            if (i + batch_size) % 10000 == 0:
                print(f"    处理进度: {i + batch_size:,} / {len(self.protein_ids):,} 蛋白质, 当前样本对: {len(pairs):,}")
        
        return pairs
    
    def _build_all_pairs(self) -> List[Tuple[str, str, int]]:
        """构建所有样本对（用于验证/测试）"""
        pairs = []
        
        for protein_id in self.protein_ids:
            positive_gos = set(self.labels_dict.get(protein_id, []))
            
            for go_id in self.all_go_ids:
                label = 1 if go_id in positive_gos else 0
                pairs.append((protein_id, go_id, label))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        protein_id, go_id, label = self.pairs[idx]
        
        # 获取embeddings
        protein_idx = np.where(self.protein_ids == protein_id)[0][0]
        go_idx = self.go_id_to_idx[go_id]
        
        seq_emb = torch.FloatTensor(self.seq_embeddings[protein_idx])
        go_emb = torch.FloatTensor(self.go_embeddings[go_idx])
        label = torch.FloatTensor([label])
        
        return {
            'seq_emb': seq_emb,
            'go_emb': go_emb,
            'label': label,
            'protein_id': protein_id,
            'go_id': go_id
        }


def collate_fn(batch):
    """Batch整理函数"""
    seq_embs = torch.stack([item['seq_emb'] for item in batch])
    go_embs = torch.stack([item['go_emb'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    protein_ids = [item['protein_id'] for item in batch]
    go_ids = [item['go_id'] for item in batch]
    
    return {
        'seq_embs': seq_embs,
        'go_embs': go_embs,
        'labels': labels,
        'protein_ids': protein_ids,
        'go_ids': go_ids
    }


def load_go_embeddings(go_embeddings_path: str) -> Tuple[np.ndarray, dict]:
    """加载GO GNN embeddings"""
    embeddings = np.load(f'{go_embeddings_path}/go_term_embeddings.npy')
    with open(f'{go_embeddings_path}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✓ 加载GO embeddings: {embeddings.shape}")
    return embeddings, metadata


def load_processed_data(processed_dir: str) -> dict:
    """加载processed_data_v4（不含序列）"""
    npz_file = os.path.join(processed_dir, "cafa6_data.npz")
    
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"找不到数据文件: {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    
    print(f"✓ 加载processed数据:")
    print(f"  训练集: {len(data['train_ids']):,} 蛋白质")
    print(f"  验证集: {len(data['val_ids']):,} 蛋白质")
    if 'test_ids' in data:
        print(f"  测试集: {len(data['test_ids']):,} 蛋白质")
    print(f"  ⚠️  注意: 序列数据需要从ESM2 embeddings获取")
    
    return data


def create_labels_dict(protein_ids: np.ndarray, labels: np.ndarray, go_terms: list) -> dict:
    """
    从标签矩阵创建labels_dict
    
    参数:
        protein_ids: 蛋白质ID数组
        labels: 标签矩阵 [num_proteins, num_gos] (uint8或float32)
        go_terms: GO term列表
    
    返回:
        {protein_id: [positive_go_ids]}
    """
    labels_dict = {}
    
    for i, protein_id in enumerate(protein_ids):
        positive_indices = np.where(labels[i] > 0)[0]
        positive_gos = [go_terms[idx] for idx in positive_indices]
        labels_dict[protein_id] = positive_gos
    
    return labels_dict


def create_dataloaders(config: dict, base_dir: str = '.') -> Tuple:
    """
    创建数据加载器
    
    返回:
        train_loader, val_loader, metadata
    """
    # 加载processed数据
    processed_dir = os.path.join(base_dir, config['data']['processed_dir'])
    data = load_processed_data(processed_dir)
    
    # 获取processed数据中的GO列表
    processed_go_terms = data['go_terms'].tolist()
    processed_go_set = set(processed_go_terms)
    
    print(f"  Processed数据的GO数: {len(processed_go_terms):,}")
    
    # 加载GO embeddings
    go_embeddings_path = os.path.join(base_dir, config['data']['go_embeddings_dir'])
    full_go_embeddings, full_go_metadata = load_go_embeddings(go_embeddings_path)
    
    print(f"  GO GNN完整GO数: {len(full_go_metadata['go_id_to_idx']):,}")
    
    # 只保留processed数据中的GO
    print(f"\n  过滤GO embeddings以匹配processed数据...")
    filtered_indices = []
    filtered_go_ids = []
    
    for go_id in processed_go_terms:
        if go_id in full_go_metadata['go_id_to_idx']:
            idx = full_go_metadata['go_id_to_idx'][go_id]
            filtered_indices.append(idx)
            filtered_go_ids.append(go_id)
        else:
            print(f"    ⚠️  警告: GO {go_id} 不在GO GNN embeddings中")
    
    # 提取对应的embeddings
    go_embeddings = full_go_embeddings[filtered_indices]
    
    # 构建新的metadata
    go_metadata = {
        'go_id_to_idx': {go_id: i for i, go_id in enumerate(filtered_go_ids)},
        'idx_to_go_id': {i: go_id for i, go_id in enumerate(filtered_go_ids)}
    }
    
    print(f"  ✓ 过滤后GO数: {len(filtered_go_ids):,}")
    print(f"  ✓ GO embeddings shape: {go_embeddings.shape}")
    
    # 加载ESM2 embeddings
    emb_file = os.path.join(base_dir, config['esm2']['embeddings_file'])
    print(f"\n✓ 加载ESM2 embeddings: {emb_file}")
    emb_data = np.load(emb_file)
    train_seq_embeddings = emb_data['train_embeddings']
    val_seq_embeddings = emb_data['val_embeddings']
    print(f"  训练集: {train_seq_embeddings.shape}")
    print(f"  验证集: {val_seq_embeddings.shape}")
    
    # 创建labels_dict（只使用过滤后的GO）
    train_labels_dict = create_labels_dict(data['train_ids'], data['train_labels'], filtered_go_ids)
    val_labels_dict = create_labels_dict(data['val_ids'], data['val_labels'], filtered_go_ids)
    
    print(f"\n创建Pairwise数据集:")
    print(f"  采样策略: {config['training']['sampling']['strategy']}")
    print(f"  负样本比例: {config['training']['sampling']['negative_ratio']}")
    
    # 创建数据集
    train_dataset = PairwiseDataset(
        protein_ids=data['train_ids'],
        seq_embeddings=train_seq_embeddings,
        go_embeddings=go_embeddings,
        go_metadata=go_metadata,
        labels_dict=train_labels_dict,
        sampling_strategy=config['training']['sampling']['strategy'],
        negative_ratio=config['training']['sampling']['negative_ratio'],
        is_training=True
    )
    
    val_dataset = PairwiseDataset(
        protein_ids=data['val_ids'],
        seq_embeddings=val_seq_embeddings,
        go_embeddings=go_embeddings,
        go_metadata=go_metadata,
        labels_dict=val_labels_dict,
        sampling_strategy="all",  # 验证时用所有对
        negative_ratio=0,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['training']['pin_memory']
    )
    
    metadata = {
        'go_terms': filtered_go_ids,
        'go_metadata': go_metadata,
        'num_proteins': {
            'train': len(data['train_ids']),
            'val': len(data['val_ids'])
        },
        'num_gos': len(filtered_go_ids)
    }
    
    print(f"\n✓ 数据加载完成:")
    print(f"  训练集: {len(train_dataset):,} 样本对")
    print(f"  验证集: {len(val_dataset):,} 样本对")
    print(f"  GO terms: {metadata['num_gos']:,}")
    
    return train_loader, val_loader, metadata