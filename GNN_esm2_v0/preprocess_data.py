"""
CAFA6数据预处理脚本 - 生成processed_data_v4
从cafa主目录运行: python GNN_esm2/preprocess_data.py

关键特性:
1. 只保留IA > 0的GO terms（过滤无意义的GO）
2. 保留所有有效GO terms（不限制top500）
3. 与Pairwise模型训练接口完全一致
4. 内存优化：使用uint8，及时释放内存
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm
import gc


def print_memory_usage():
    """打印当前内存使用"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"  当前内存使用: {mem_info.rss / (1024**3):.2f} GB")
    except ImportError:
        pass


def load_ia_weights(ia_file: str) -> dict:
    """
    加载IA权重，过滤IA=0的GO
    
    返回:
        {go_id: ia_weight} 只包含IA>0的GO
    """
    print("="*70)
    print("加载IA权重")
    print("="*70)
    
    ia_weights = {}
    
    with open(ia_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                go_id = parts[0]
                ia = float(parts[1])
                
                # 只保留IA > 0的GO
                if ia > 0:
                    ia_weights[go_id] = ia
    
    print(f"✓ 加载IA权重: {len(ia_weights)} 个GO terms (IA > 0)")
    
    return ia_weights


def load_sequences(fasta_file: str) -> dict:
    """
    加载蛋白质序列
    
    返回:
        {uniprot_id: sequence}
    """
    print("\n" + "="*70)
    print("加载蛋白质序列")
    print("="*70)
    
    sequences = {}
    
    for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="读取FASTA"):
        # 从 'sp|A0A0C5B5G6|MOTSC_HUMAN' 提取 'A0A0C5B5G6'
        if '|' in record.id:
            uniprot_id = record.id.split('|')[1]
        else:
            uniprot_id = record.id
        
        sequences[uniprot_id] = str(record.seq)
    
    print(f"✓ 加载序列: {len(sequences)} 条")
    
    return sequences


def load_annotations(terms_file: str, valid_go_ids: set) -> tuple:
    """
    加载GO标注
    
    参数:
        terms_file: train_terms.tsv路径
        valid_go_ids: 有效的GO ID集合（IA > 0）
    
    返回:
        annotations: {protein_id: {aspect: [go_ids]}}
        go_aspect_map: {go_id: aspect}
    """
    print("\n" + "="*70)
    print("加载GO标注")
    print("="*70)
    
    df = pd.read_csv(terms_file, sep='\t')
    print(f"✓ 读取标注文件: {len(df)} 条记录")
    
    # 过滤：只保留IA > 0的GO
    df_filtered = df[df['term'].isin(valid_go_ids)]
    print(f"✓ 过滤后（IA > 0）: {len(df_filtered)} 条记录")
    
    # 构建标注字典
    annotations = defaultdict(lambda: {'C': [], 'F': [], 'P': []})
    go_aspect_map = {}
    
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="处理标注"):
        protein_id = row['EntryID']
        go_id = row['term']
        aspect = row['aspect']
        
        annotations[protein_id][aspect].append(go_id)
        go_aspect_map[go_id] = aspect
    
    print(f"✓ 标注蛋白质数: {len(annotations)}")
    print(f"✓ 唯一GO terms: {len(go_aspect_map)}")
    
    # 统计各aspect
    aspect_counts = defaultdict(int)
    for go_id, aspect in go_aspect_map.items():
        aspect_counts[aspect] += 1
    
    print(f"\n各aspect的GO数量:")
    for aspect in ['C', 'F', 'P']:
        print(f"  {aspect}: {aspect_counts[aspect]}")
    
    return dict(annotations), go_aspect_map


def filter_valid_proteins(sequences: dict, annotations: dict) -> tuple:
    """
    过滤有效蛋白质（既有序列又有标注）
    
    返回:
        valid_protein_ids: 有效蛋白质ID列表
        valid_sequences: {protein_id: sequence}
        valid_annotations: {protein_id: {aspect: [go_ids]}}
    """
    print("\n" + "="*70)
    print("过滤有效蛋白质")
    print("="*70)
    
    # 既有序列又有标注的蛋白质
    seq_proteins = set(sequences.keys())
    ann_proteins = set(annotations.keys())
    valid_proteins = seq_proteins & ann_proteins
    
    print(f"✓ 有序列的蛋白质: {len(seq_proteins)}")
    print(f"✓ 有标注的蛋白质: {len(ann_proteins)}")
    print(f"✓ 有效蛋白质（交集）: {len(valid_proteins)}")
    
    valid_protein_ids = sorted(list(valid_proteins))
    valid_sequences = {pid: sequences[pid] for pid in valid_protein_ids}
    valid_annotations = {pid: annotations[pid] for pid in valid_protein_ids}
    
    return valid_protein_ids, valid_sequences, valid_annotations


def build_go_vocabulary(annotations: dict, go_aspect_map: dict) -> tuple:
    """
    构建GO词汇表
    
    返回:
        go_terms: 所有GO ID的列表
        go_to_idx: {go_id: index}
        go_aspects: 每个GO的aspect数组
    """
    print("\n" + "="*70)
    print("构建GO词汇表")
    print("="*70)
    
    # 收集所有GO
    all_gos = set()
    for protein_anns in annotations.values():
        for go_list in protein_anns.values():
            all_gos.update(go_list)
    
    # 排序（保证可复现）
    go_terms = sorted(list(all_gos))
    go_to_idx = {go: idx for idx, go in enumerate(go_terms)}
    
    # 构建aspect数组
    go_aspects = np.array([go_aspect_map[go] for go in go_terms])
    
    print(f"✓ 总GO数: {len(go_terms)}")
    
    # 统计各aspect
    aspect_counts = {
        'C': (go_aspects == 'C').sum(),
        'F': (go_aspects == 'F').sum(),
        'P': (go_aspects == 'P').sum()
    }
    
    print(f"\n各aspect的GO数量:")
    for aspect in ['C', 'F', 'P']:
        print(f"  {aspect}: {aspect_counts[aspect]}")
    
    return go_terms, go_to_idx, go_aspects


def build_label_matrix(
    protein_ids: list,
    annotations: dict,
    go_to_idx: dict
) -> np.ndarray:
    """
    构建标签矩阵 - 使用uint8节省内存
    
    返回:
        labels: [num_proteins, num_gos] 稀疏0/1矩阵
    """
    num_proteins = len(protein_ids)
    num_gos = len(go_to_idx)
    
    print(f"\n构建标签矩阵: [{num_proteins}, {num_gos}]")
    
    # 使用uint8而不是float32，节省75%内存
    labels = np.zeros((num_proteins, num_gos), dtype=np.uint8)
    
    for i, protein_id in enumerate(tqdm(protein_ids, desc="构建标签")):
        protein_anns = annotations[protein_id]
        
        # 收集该蛋白的所有GO
        all_protein_gos = []
        for go_list in protein_anns.values():
            all_protein_gos.extend(go_list)
        
        # 标记为1
        for go_id in all_protein_gos:
            if go_id in go_to_idx:
                labels[i, go_to_idx[go_id]] = 1
    
    # 统计稀疏度
    num_positive = (labels > 0).sum()
    sparsity = 1 - (num_positive / labels.size)
    
    print(f"✓ 标注总数: {num_positive:,}")
    print(f"✓ 平均每蛋白标注数: {num_positive / num_proteins:.1f}")
    print(f"✓ 稀疏度: {sparsity * 100:.2f}%")
    
    return labels


def split_train_val(
    protein_ids: list,
    sequences: dict,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> tuple:
    """
    划分训练集和验证集 - 不保存序列字符串，只保存ID
    
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
    
    # 只保存ID和标签，不保存序列（序列可以从原始fasta读取）
    print(f"\n构建训练集...")
    train_ids = np.array([protein_ids[i] for i in train_indices])
    train_labels = labels[train_indices]
    
    print(f"✓ 训练集: {len(train_ids):,} 蛋白质")
    print(f"  标签矩阵: {train_labels.shape}, {train_labels.dtype}")
    
    # 验证集
    print(f"\n构建验证集...")
    val_ids = np.array([protein_ids[i] for i in val_indices])
    val_labels = labels[val_indices]
    
    print(f"✓ 验证集: {len(val_ids):,} 蛋白质")
    print(f"  标签矩阵: {val_labels.shape}, {val_labels.dtype}")
    
    return train_ids, train_labels, val_ids, val_labels


def process_test_set(
    test_fasta: str,
    go_to_idx: dict
) -> np.ndarray:
    """
    处理测试集（无标签）- 只保存ID
    
    返回:
        test_ids
    """
    print("\n" + "="*70)
    print("处理测试集")
    print("="*70)
    
    if not os.path.exists(test_fasta):
        print(f"⚠️  测试集文件不存在: {test_fasta}")
        print("   跳过测试集处理")
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
    """保存处理后的数据 - 不保存序列字符串"""
    print("\n" + "="*70)
    print("保存处理后的数据")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建IA权重数组（按go_terms顺序）
    print("构建IA权重数组...")
    ia_weights = np.array([ia_weights_dict.get(go, 0.0) for go in go_terms], dtype=np.float32)
    
    # 保存数据（不包含序列）
    output_file = os.path.join(output_dir, "cafa6_data.npz")
    
    print("准备保存字典...")
    save_dict = {
        'train_ids': train_ids,
        'train_labels': train_labels,  # uint8
        'val_ids': val_ids,
        'val_labels': val_labels,  # uint8
        'go_terms': np.array(go_terms),
        'go_aspects': go_aspects,
        'ia_weights': ia_weights
    }
    
    # 添加测试集（如果存在）
    if test_ids is not None:
        save_dict['test_ids'] = test_ids
    
    print("保存到磁盘（压缩格式）...")
    np.savez_compressed(output_file, **save_dict)
    
    print(f"✓ 保存到: {output_file}")
    
    # 文件大小
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ 文件大小: {file_size_mb:.2f} MB")
    
    # 保存统计信息
    print("\n保存统计信息...")
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
        'note': '序列数据需要从原始FASTA文件读取'
    }
    
    import json
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
    print(f"\n⚠️  注意: 序列数据未保存，需要从原始FASTA读取")


def main():
    parser = argparse.ArgumentParser(description="CAFA6数据预处理 - 生成processed_data_v4")
    parser.add_argument('--data_dir', type=str,
                        default='cafa-6-protein-function-prediction',
                        help='CAFA6原始数据目录')
    parser.add_argument('--output_dir', type=str,
                        default='processed_data_v4',
                        help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 确保在cafa主目录下运行
    if not os.path.exists(args.data_dir):
        print(f"错误: 找不到数据目录 {args.data_dir}")
        print("请在cafa主目录下运行此脚本:")
        print("  python GNN_esm2/preprocess_data.py")
        sys.exit(1)
    
    print("="*70)
    print("CAFA6数据预处理 - processed_data_v4")
    print("="*70)
    print(f"\n输入目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"验证集比例: {args.val_ratio}")
    
    # 文件路径
    train_dir = os.path.join(args.data_dir, 'Train')
    ia_file = os.path.join(args.data_dir, 'IA.tsv')
    train_fasta = os.path.join(train_dir, 'train_sequences.fasta')
    train_terms = os.path.join(train_dir, 'train_terms.tsv')
    test_fasta = os.path.join(args.data_dir, 'Test/testsuperset.fasta')
    
    # 1. 加载IA权重（过滤IA=0的GO）
    ia_weights_dict = load_ia_weights(ia_file)
    valid_go_ids = set(ia_weights_dict.keys())
    
    # 2. 加载序列
    sequences = load_sequences(train_fasta)
    
    # 3. 加载标注（只保留IA>0的GO）
    annotations, go_aspect_map = load_annotations(train_terms, valid_go_ids)
    
    # 4. 过滤有效蛋白质
    protein_ids, sequences, annotations = filter_valid_proteins(
        sequences, annotations
    )
    
    # 5. 构建GO词汇表
    go_terms, go_to_idx, go_aspects = build_go_vocabulary(
        annotations, go_aspect_map
    )
    print_memory_usage()
    
    # 6. 构建标签矩阵
    labels = build_label_matrix(protein_ids, annotations, go_to_idx)
    print_memory_usage()
    
    # 释放annotations以节省内存
    print("\n释放annotations字典以节省内存...")
    del annotations
    del go_aspect_map
    gc.collect()
    print_memory_usage()
    
    # 7. 划分训练集和验证集
    train_ids, train_labels, val_ids, val_labels = split_train_val(
        protein_ids, sequences, labels, args.val_ratio, args.random_seed
    )
    print_memory_usage()
    
    # 释放原始labels矩阵和sequences字典
    print("\n释放原始数据...")
    del labels
    del sequences
    del protein_ids
    gc.collect()
    print_memory_usage()
    
    # 8. 处理测试集
    test_ids = process_test_set(test_fasta, go_to_idx)
    
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
    print(f"  1. 预计算ESM2 embeddings (如果还没有)")
    print(f"  2. 运行测试: python GNN_esm2/test_all.py")
    print(f"  3. 开始训练: python GNN_esm2/train.py")


if __name__ == "__main__":
    main()