"""
CAFA6数据预处理脚本（企业级兼容版本）

从cafa主目录运行: python GNN_esm2/preprocess_data.py

输出:
  processed_data_v4/
    ├── cafa6_data.npz         (不含序列，只有ID和标签)
    └── statistics.json
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from collections import defaultdict
import json
import gc


def print_memory_usage():
    """打印当前内存使用"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"  当前内存使用: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")


def load_ia_weights(ia_file: str) -> dict:
    """加载IA权重"""
    print("\n" + "="*70)
    print("加载IA权重")
    print("="*70)
    
    ia_weights = {}
    
    with open(ia_file, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                go_id = parts[0]
                ia = float(parts[1])
                if ia > 0:  # 只保留IA > 0的GO
                    ia_weights[go_id] = ia
    
    print(f"✓ 加载IA权重: {len(ia_weights)} 个GO terms (IA > 0)")
    return ia_weights


def load_sequences(fasta_file: str) -> dict:
    """加载蛋白质序列"""
    print("\n" + "="*70)
    print("加载蛋白质序列")
    print("="*70)
    
    sequences = {}
    
    for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="读取FASTA"):
        if '|' in record.id:
            uniprot_id = record.id.split('|')[1]
        else:
            uniprot_id = record.id
        
        sequences[uniprot_id] = str(record.seq)
    
    print(f"✓ 加载序列: {len(sequences)} 条")
    return sequences


def load_annotations(anno_file: str, ia_weights: dict) -> dict:
    """
    加载GO标注（只保留IA > 0的GO）
    
    返回:
        {protein_id: {go_id: aspect}}
    """
    print("\n" + "="*70)
    print("加载GO标注")
    print("="*70)
    
    df = pd.read_csv(anno_file, sep='\t', header=None, 
                     names=['protein_id', 'go_id', 'aspect'])
    
    print(f"✓ 读取标注文件: {len(df)} 条记录")
    
    # 过滤IA > 0的GO
    df_filtered = df[df['go_id'].isin(ia_weights.keys())]
    print(f"✓ 过滤后（IA > 0）: {len(df_filtered)} 条记录")
    
    # 构建字典
    annotations = defaultdict(dict)
    
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="处理标注"):
        protein_id = row['protein_id']
        go_id = row['go_id']
        aspect = row['aspect']
        annotations[protein_id][go_id] = aspect
    
    print(f"✓ 标注蛋白质数: {len(annotations)}")
    print(f"✓ 唯一GO terms: {len(set(df_filtered['go_id']))}")
    
    # 统计各aspect的GO数量
    go_aspects = df_filtered.groupby('aspect')['go_id'].nunique()
    print("各aspect的GO数量:")
    for aspect in ['C', 'F', 'P']:
        if aspect in go_aspects:
            print(f"  {aspect}: {go_aspects[aspect]}")
    
    return dict(annotations)


def build_go_vocabulary(annotations: dict, ia_weights: dict) -> tuple:
    """
    构建GO词汇表
    
    返回:
        go_terms: GO ID列表
        go_to_idx: {go_id: index}
        go_aspects: aspect数组
    """
    print("\n" + "="*70)
    print("构建GO词汇表")
    print("="*70)
    
    # 收集所有GO
    all_gos = set()
    go_aspect_map = {}
    
    for protein_id, protein_gos in annotations.items():
        for go_id, aspect in protein_gos.items():
            all_gos.add(go_id)
            go_aspect_map[go_id] = aspect
    
    # 只保留有IA权重的GO
    all_gos = all_gos & set(ia_weights.keys())
    
    # 排序
    go_terms = sorted(all_gos)
    go_to_idx = {go: idx for idx, go in enumerate(go_terms)}
    go_aspects = np.array([go_aspect_map.get(go, 'P') for go in go_terms])
    
    print(f"✓ 总GO数: {len(go_terms)}")
    
    # 统计
    aspect_counts = {'C': 0, 'F': 0, 'P': 0}
    for aspect in go_aspects:
        if aspect in aspect_counts:
            aspect_counts[aspect] += 1
    
    print("各aspect的GO数量:")
    for aspect in ['C', 'F', 'P']:
        print(f"  {aspect}: {aspect_counts[aspect]}")
    
    return go_terms, go_to_idx, go_aspects


def build_label_matrix(
    protein_ids: list,
    annotations: dict,
    go_to_idx: dict
) -> np.ndarray:
    """
    构建稀疏标签矩阵（uint8格式）
    
    返回:
        labels: [num_proteins, num_gos] uint8数组
    """
    print_memory_usage()
    print(f"构建标签矩阵: [{len(protein_ids)}, {len(go_to_idx)}]")
    
    # 使用uint8节省内存
    labels = np.zeros((len(protein_ids), len(go_to_idx)), dtype=np.uint8)
    
    for i, protein_id in enumerate(tqdm(protein_ids, desc="构建标签")):
        if protein_id in annotations:
            for go_id in annotations[protein_id]:
                if go_id in go_to_idx:
                    labels[i, go_to_idx[go_id]] = 1
    
    # 统计
    total_labels = labels.sum()
    avg_labels = total_labels / len(protein_ids)
    sparsity = 1 - (total_labels / labels.size)
    
    print(f"✓ 标注总数: {total_labels:,}")
    print(f"✓ 平均每蛋白标注数: {avg_labels:.1f}")
    print(f"✓ 稀疏度: {sparsity * 100:.2f}%")
    print_memory_usage()
    
    return labels


def split_train_val(
    protein_ids: list,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> tuple:
    """
    划分训练集和验证集（不保存序列）
    
    返回:
        train_ids, train_labels,
        val_ids, val_labels
    """
    print("\n" + "="*70)
    print("划分训练集和验证集")
    print("="*70)
    
    np.random.seed(random_seed)
    
    num_proteins = len(protein_ids)
    indices = np.random.permutation(num_proteins)
    
    split_idx = int(num_proteins * (1 - val_ratio))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"✓ 生成索引完成")
    print(f"  训练集索引: {len(train_indices):,}")
    print(f"  验证集索引: {len(val_indices):,}")
    
    # 只保存ID和标签，不保存序列
    train_ids = np.array([protein_ids[i] for i in train_indices])
    train_labels = labels[train_indices]
    
    val_ids = np.array([protein_ids[i] for i in val_indices])
    val_labels = labels[val_indices]
    
    print(f"\n✓ 训练集: {len(train_ids):,} 蛋白质")
    print(f"  标签矩阵: {train_labels.shape}, {train_labels.dtype}")
    print(f"✓ 验证集: {len(val_ids):,} 蛋白质")
    print(f"  标签矩阵: {val_labels.shape}, {val_labels.dtype}")
    
    return train_ids, train_labels, val_ids, val_labels


def process_test_set(test_fasta: str) -> np.ndarray:
    """处理测试集（只保存ID）"""
    print("\n" + "="*70)
    print("处理测试集")
    print("="*70)
    
    if not os.path.exists(test_fasta):
        print(f"⚠️  测试集文件不存在: {test_fasta}")
        return None
    
    test_ids_list = []
    
    for record in tqdm(SeqIO.parse(test_fasta, "fasta"), desc="读取测试集"):
        if '|' in record.id:
            uniprot_id = record.id.split('|')[1]
        else:
            uniprot_id = record.id
        
        test_ids_list.append(uniprot_id)
    
    test_ids = np.array(test_ids_list)
    print(f"✓ 测试集: {len(test_ids):,} 蛋白质")
    
    return test_ids


def save_processed_data(
    output_dir: str,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    val_ids: np.ndarray,
    val_labels: np.ndarray,
    test_ids: np.ndarray,
    go_terms: list,
    go_aspects: np.ndarray,
    ia_weights_dict: dict
):
    """保存处理后的数据（不保存序列）"""
    print("\n" + "="*70)
    print("保存处理后的数据")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建IA权重数组
    ia_weights = np.array([ia_weights_dict.get(go, 0.0) for go in go_terms], dtype=np.float32)
    
    # 保存数据
    output_file = os.path.join(output_dir, "cafa6_data.npz")
    
    save_dict = {
        'train_ids': train_ids,
        'train_labels': train_labels,
        'val_ids': val_ids,
        'val_labels': val_labels,
        'go_terms': np.array(go_terms),
        'go_aspects': go_aspects,
        'ia_weights': ia_weights
    }
    
    if test_ids is not None:
        save_dict['test_ids'] = test_ids
    
    np.savez_compressed(output_file, **save_dict)
    
    print(f"✓ 保存到: {output_file}")
    print(f"✓ 文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # 保存统计信息
    stats = {
        'num_train': int(len(train_ids)),
        'num_val': int(len(val_ids)),
        'num_test': int(len(test_ids)) if test_ids is not None else 0,
        'num_gos': int(len(go_terms)),
        'num_gos_per_aspect': {
            'C': int((go_aspects == 'C').sum()),
            'F': int((go_aspects == 'F').sum()),
            'P': int((go_aspects == 'P').sum())
        },
        'sparsity': float(1 - train_labels.sum() / train_labels.size),
        'avg_labels_per_protein': float(train_labels.sum() / len(train_labels)),
        'note': '序列数据需要从原始FASTA文件读取或使用ESM2 embeddings'
    }
    
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ 统计信息保存到: {stats_file}")
    
    # 打印摘要
    print("\n" + "="*70)
    print("数据摘要")
    print("="*70)
    print(f"训练集: {stats['num_train']:,} 蛋白质")
    print(f"验证集: {stats['num_val']:,} 蛋白质")
    if stats['num_test'] > 0:
        print(f"测试集: {stats['num_test']:,} 蛋白质")
    print(f"\nGO Terms总数: {stats['num_gos']:,}")
    print(f"  C: {stats['num_gos_per_aspect']['C']:,}")
    print(f"  F: {stats['num_gos_per_aspect']['F']:,}")
    print(f"  P: {stats['num_gos_per_aspect']['P']:,}")
    print(f"\n稀疏度: {stats['sparsity'] * 100:.2f}%")
    print(f"平均每蛋白标注数: {stats['avg_labels_per_protein']:.1f}")
    print(f"\n⚠️  注意: 序列数据未保存，训练时使用ESM2 embeddings")


def main():
    parser = argparse.ArgumentParser(description="CAFA6数据预处理（企业级兼容）")
    parser.add_argument('--data_dir', type=str, 
                       default='cafa-6-protein-function-prediction',
                       help='原始数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='processed_data_v4',
                       help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CAFA6数据预处理 - processed_data_v4（企业级兼容）")
    print("="*70)
    print(f"输入目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"验证集比例: {args.val_ratio}")
    print("="*70)
    
    # 文件路径
    ia_file = os.path.join(args.data_dir, 'IA.tsv')
    train_fasta = os.path.join(args.data_dir, 'Train', 'train_sequences.fasta')
    train_anno = os.path.join(args.data_dir, 'Train', 'train_terms.tsv')
    test_fasta = os.path.join(args.data_dir, 'Test', 'testsuperset.fasta')
    
    # 1. 加载IA权重
    ia_weights_dict = load_ia_weights(ia_file)
    
    # 2. 加载序列（只用于验证，不保存）
    sequences = load_sequences(train_fasta)
    
    # 3. 加载标注
    annotations = load_annotations(train_anno, ia_weights_dict)
    
    # 4. 过滤有效蛋白质
    print("\n" + "="*70)
    print("过滤有效蛋白质")
    print("="*70)
    
    valid_proteins = set(sequences.keys()) & set(annotations.keys())
    protein_ids = sorted(valid_proteins)
    
    print(f"✓ 有序列的蛋白质: {len(sequences)}")
    print(f"✓ 有标注的蛋白质: {len(annotations)}")
    print(f"✓ 有效蛋白质（交集）: {len(protein_ids)}")
    
    # 5. 构建GO词汇表
    go_terms, go_to_idx, go_aspects = build_go_vocabulary(annotations, ia_weights_dict)
    
    # 6. 构建标签矩阵
    labels = build_label_matrix(protein_ids, annotations, go_to_idx)
    
    # 释放内存
    print("\n释放annotations字典以节省内存...")
    del annotations
    del sequences
    gc.collect()
    print_memory_usage()
    
    # 7. 划分训练集和验证集
    train_ids, train_labels, val_ids, val_labels = split_train_val(
        protein_ids, labels, args.val_ratio, args.random_seed
    )
    
    # 释放原始数据
    del labels
    del protein_ids
    gc.collect()
    print_memory_usage()
    
    # 8. 处理测试集
    test_ids = process_test_set(test_fasta)
    
    # 9. 保存数据
    save_processed_data(
        args.output_dir,
        train_ids, train_labels,
        val_ids, val_labels,
        test_ids,
        go_terms, go_aspects, ia_weights_dict
    )
    
    print("\n" + "="*70)
    print("✅ 数据预处理完成！")
    print("="*70)
    print(f"\n下一步:")
    print(f"  1. 预计算ESM2 embeddings:")
    print(f"     python GNN_esm2/precompute_embeddings.py")
    print(f"  2. 开始训练:")
    print(f"     python GNN_esm2/train_scalable.py --exp_name my_experiment")


if __name__ == "__main__":
    main()
