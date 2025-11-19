"""
数据集模块
包含原始实现和企业级可扩展实现
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from .scalable_dataset import (
    ScalablePairwiseDataset,
    UniformNegativeSampler,
    StratifiedNegativeSampler,
    BatchEvaluator,
    StreamingMetricsCalculator,
    collate_fn as scalable_collate_fn,
    create_labels_dict as create_labels_dict_from_matrix,
    estimate_epoch_size
)


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


def load_go_embeddings(go_embeddings_dir: str) -> Tuple[np.ndarray, dict]:
    """加载GO embeddings"""
    emb_file = os.path.join(go_embeddings_dir, "go_term_embeddings.npy")
    metadata_file = os.path.join(go_embeddings_dir, "metadata.pkl")
    
    if not os.path.exists(emb_file):
        raise FileNotFoundError(f"找不到GO embeddings: {emb_file}")
    
    embeddings = np.load(emb_file)
    metadata = np.load(metadata_file, allow_pickle=True)
    
    print(f"✓ 加载GO embeddings: {embeddings.shape}")
    
    return embeddings, metadata


def create_scalable_dataloaders(config: dict, base_dir: str = '.') -> Tuple:
    """
    创建可扩展的数据加载器（企业级实现）⭐ 推荐
    
    特性:
    - IterableDataset - 动态生成，内存O(1)
    - 可插拔负采样策略
    - 支持任意规模的蛋白质和GO
    
    返回:
        train_loader, val_loader, metadata
    """
    print("\n" + "="*70)
    print("创建可扩展数据加载器 (Enterprise Edition)")
    print("="*70)
    
    # 加载processed数据
    processed_dir = os.path.join(base_dir, config['data']['processed_dir'])
    data = load_processed_data(processed_dir)
    
    # 获取processed数据中的GO列表
    processed_go_terms = data['go_terms'].tolist()
    processed_go_set = set(processed_go_terms)
    
    print(f"  Processed数据的GO数: {len(processed_go_terms):,}")
    
    # 加载GO embeddings并过滤
    go_embeddings_path = os.path.join(base_dir, config['data']['go_embeddings_dir'])
    full_go_embeddings, full_go_metadata = load_go_embeddings(go_embeddings_path)
    
    print(f"  GO GNN完整GO数: {len(full_go_metadata['go_id_to_idx']):,}")
    print(f"  过滤GO embeddings以匹配processed数据...")
    
    filtered_indices = []
    filtered_go_ids = []
    
    for go_id in processed_go_terms:
        if go_id in full_go_metadata['go_id_to_idx']:
            idx = full_go_metadata['go_id_to_idx'][go_id]
            filtered_indices.append(idx)
            filtered_go_ids.append(go_id)
    
    go_embeddings = full_go_embeddings[filtered_indices]
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
    
    # 创建labels_dict
    train_labels_dict = create_labels_dict_from_matrix(
        data['train_ids'], data['train_labels'], filtered_go_ids
    )
    val_labels_dict = create_labels_dict_from_matrix(
        data['val_ids'], data['val_labels'], filtered_go_ids
    )
    
    # 构建GO aspects字典（用于分层采样）
    go_aspects_dict = {}
    go_aspects_array = data['go_aspects']
    for i, go_id in enumerate(processed_go_terms):
        if go_id in go_metadata['go_id_to_idx']:
            go_aspects_dict[go_id] = go_aspects_array[i]
    
    # 创建负采样器
    sampler_type = config['training']['sampling'].get('sampler_type', 'uniform')
    
    if sampler_type == 'uniform':
        train_sampler = UniformNegativeSampler(filtered_go_ids)
        print(f"\n✓ 使用均匀负采样器")
    elif sampler_type == 'stratified':
        train_sampler = StratifiedNegativeSampler(filtered_go_ids, go_aspects_dict)
        print(f"\n✓ 使用分层负采样器（按aspect）")
    else:
        raise ValueError(f"不支持的采样器类型: {sampler_type}")
    
    # 验证集采样器（也用负采样，但比例小）
    val_sampler = UniformNegativeSampler(filtered_go_ids)
    
    # 估算epoch大小
    train_epoch_size = estimate_epoch_size(
        train_labels_dict,
        negative_ratio=config['training']['sampling']['negative_ratio']
    )
    
    # 验证集也用负采样（大幅减少验证时间）
    val_negative_ratio = config['training']['sampling'].get('val_negative_ratio', 1)
    val_epoch_size = estimate_epoch_size(val_labels_dict, negative_ratio=val_negative_ratio)
    
    print(f"\n创建可扩展Pairwise数据集:")
    print(f"  训练采样策略: {sampler_type}")
    print(f"  训练负样本比例: {config['training']['sampling']['negative_ratio']}")
    print(f"  验证负样本比例: {val_negative_ratio}")
    
    # 创建数据集
    train_dataset = ScalablePairwiseDataset(
        protein_ids=data['train_ids'],
        seq_embeddings=train_seq_embeddings,
        go_embeddings=go_embeddings,
        go_metadata=go_metadata,
        labels_dict=train_labels_dict,
        negative_sampler=train_sampler,
        mode='train',
        epoch_size=train_epoch_size,
        seed=config['training'].get('seed', 42)
    )
    
    val_dataset = ScalablePairwiseDataset(
        protein_ids=data['val_ids'],
        seq_embeddings=val_seq_embeddings,
        go_embeddings=go_embeddings,
        go_metadata=go_metadata,
        labels_dict=val_labels_dict,
        negative_sampler=val_sampler,
        mode='val',
        epoch_size=val_epoch_size,
        seed=config['training'].get('seed', 42) + 1
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        collate_fn=scalable_collate_fn,
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        num_workers=config['training']['num_workers'],
        collate_fn=scalable_collate_fn,
        pin_memory=config['training']['pin_memory']
    )
    
    metadata = {
        'go_terms': filtered_go_ids,
        'go_metadata': go_metadata,
        'go_embeddings': go_embeddings,
        'go_aspects': go_aspects_dict,
        'num_proteins': {
            'train': len(data['train_ids']),
            'val': len(data['val_ids'])
        },
        'num_gos': len(filtered_go_ids),
        'epoch_sizes': {
            'train': train_epoch_size,
            'val': val_epoch_size
        }
    }
    
    print(f"\n✓ 可扩展数据加载器创建完成:")
    print(f"  训练集epoch大小: {train_epoch_size:,} 样本对")
    print(f"  验证集epoch大小: {val_epoch_size:,} 样本对")
    print(f"  GO terms: {metadata['num_gos']:,}")
    print(f"  内存使用: O(batch_size) - 恒定！")
    
    return train_loader, val_loader, metadata
