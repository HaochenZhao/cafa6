"""
改进版数据处理器 v3 - 分aspect处理
移动到 top500mlp 目录版本
运行命令示例: python top500mlp/save_processed_data_v3.py --data_dir cafa-6-protein-function-prediction
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse  # [新增] 用于参数解析
from collections import defaultdict, deque
from Bio import SeqIO
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# [新增] 路径修复，确保脚本能找到模块（如果将来需要）
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
project_root = os.path.dirname(current_dir) # 获取 cafa/ 根目录
# ==========================================


# ============================================
# 1. 读取IA值并过滤
# ============================================
def load_ia_values(ia_file: str, min_ia: float = 0.0) -> Dict[str, float]:
    """
    读取IA值，过滤掉IA=0的GO术语
    """
    print(f"\n{'='*70}")
    print(f"读取IA值: {ia_file}")
    print('='*70)

    if not os.path.exists(ia_file):
        raise FileNotFoundError(f"找不到IA文件: {ia_file}")

    ia_df = pd.read_csv(ia_file, sep='\t', header=None, names=['GO', 'IA'])

    print(f"✓ 总GO术语数: {len(ia_df):,}")
    print(f"✓ IA=0的GO: {(ia_df['IA'] == 0).sum():,} ({(ia_df['IA'] == 0).sum()/len(ia_df)*100:.1f}%)")
    
    # 过滤
    valid_ia = ia_df[ia_df['IA'] > min_ia]
    go_to_ia = dict(zip(valid_ia['GO'], valid_ia['IA']))

    print(f"\n✓ 保留的GO术语: {len(go_to_ia):,}")
    return go_to_ia


# ============================================
# 2. 解析GO本体
# ============================================
def parse_go_ontology(obo_file: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str]]:
    """解析GO本体"""
    print(f"\n{'='*70}")
    print(f"解析GO本体: {obo_file}")
    print('='*70)

    if not os.path.exists(obo_file):
        raise FileNotFoundError(f"找不到OBO文件: {obo_file}")

    go_parents = defaultdict(set)
    go_children = defaultdict(set)
    go_aspect = {}

    current_go = None
    current_namespace = None
    is_obsolete = False

    namespace_map = {
        'cellular_component': 'C',
        'molecular_function': 'F',
        'biological_process': 'P'
    }

    with open(obo_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                current_go = None
                current_namespace = None
                is_obsolete = False
            elif line.startswith("id: GO:"):
                current_go = line.split("id: ")[1]
            elif line.startswith("namespace:"):
                namespace = line.split("namespace: ")[1]
                current_namespace = namespace_map.get(namespace)
            elif line.startswith("is_obsolete:"):
                is_obsolete = True
            elif line.startswith("is_a: GO:") and current_go and not is_obsolete:
                parent_go = line.split("is_a: ")[1].split(" !")[0]
                go_parents[current_go].add(parent_go)
                go_children[parent_go].add(current_go)

            if current_go and current_namespace and not is_obsolete:
                go_aspect[current_go] = current_namespace

    print(f"✓ 解析了 {len(go_parents):,} 个GO术语的层次关系")
    
    return go_parents, go_children, go_aspect


# ============================================
# 3. 找到最近的Top-K祖先
# ============================================
def find_nearest_topk_ancestors(
    go_term: str,
    go_parents: Dict[str, Set[str]],
    topk_gos: Set[str],
    go_aspect: Dict[str, str]
) -> Set[str]:
    if go_term in topk_gos:
        return {go_term}

    current_aspect = go_aspect.get(go_term)
    if current_aspect is None:
        return set()

    ancestors = set()
    visited = set()
    queue = deque([go_term])

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        if current in topk_gos:
            if go_aspect.get(current) == current_aspect:
                ancestors.add(current)
                continue 

        if current in go_parents:
            for parent in go_parents[current]:
                if parent not in visited:
                    queue.append(parent)

    return ancestors


# ============================================
# 4. 读取序列
# ============================================
def load_sequences(fasta_file: str, name: str = "数据") -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"读取{name}: {fasta_file}")
    print('='*70)

    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"找不到FASTA文件: {fasta_file}")

    sequences = []
    ids = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        if '|' in record.id:
            protein_id = record.id.split('|')[1]
        else:
            protein_id = record.id.split()[0]

        sequence = str(record.seq)
        ids.append(protein_id)
        sequences.append(sequence)

    df = pd.DataFrame({
        'EntryID': ids,
        'sequence': sequences,
        'seq_length': [len(s) for s in sequences]
    })

    print(f"✓ 蛋白质数量: {len(df):,}")
    return df


# ============================================
# 5. 构建智能标签矩阵
# ============================================
def build_smart_label_matrix_v3(
    go_annotations: pd.DataFrame,
    protein_ids: List[str],
    go_to_ia: Dict[str, float],
    go_parents: Dict[str, Set[str]],
    go_aspect: Dict[str, str],
    top_k_per_aspect: int = 500,
    use_hierarchy: bool = True
) -> Tuple[np.ndarray, List[str], Dict]:
    
    print(f"\n{'='*70}")
    print("构建智能标签矩阵 v3")
    print('='*70)

    # 1. 过滤
    go_annotations = go_annotations[go_annotations['term'].isin(go_to_ia.keys())]
    
    # 2. 选择Top-K
    go_annotations['aspect'] = go_annotations['term'].map(go_aspect)
    go_annotations = go_annotations[go_annotations['aspect'].notna()]

    top_go_terms = []
    aspect_stats = {}

    for aspect in ['C', 'F', 'P']:
        aspect_annotations = go_annotations[go_annotations['aspect'] == aspect]
        if len(aspect_annotations) == 0:
            continue

        go_counts = aspect_annotations['term'].value_counts()
        k = min(top_k_per_aspect, len(go_counts))
        aspect_top_go = go_counts.head(k).index.tolist()
        top_go_terms.extend(aspect_top_go)
        
        aspect_stats[aspect] = {'count': k}

    topk_gos = set(top_go_terms)
    print(f"  总共选择了 {len(top_go_terms)} 个GO术语")

    # 3. 映射
    protein_go_map = defaultdict(set)
    mapping_stats = {'direct': defaultdict(int), 'mapped': defaultdict(int), 'lost': defaultdict(int)}

    for _, row in tqdm(go_annotations.iterrows(), total=len(go_annotations), desc="处理标注"):
        protein_id = row['EntryID']
        go_term = row['term']
        aspect = row['aspect']

        if go_term in topk_gos:
            protein_go_map[protein_id].add(go_term)
            mapping_stats['direct'][aspect] += 1
        elif use_hierarchy:
            nearest_ancestors = find_nearest_topk_ancestors(
                go_term, go_parents, topk_gos, go_aspect
            )
            if nearest_ancestors:
                protein_go_map[protein_id].update(nearest_ancestors)
                mapping_stats['mapped'][aspect] += len(nearest_ancestors)
            else:
                mapping_stats['lost'][aspect] += 1
        else:
            mapping_stats['lost'][aspect] += 1

    # 4. 矩阵
    go_term_to_idx = {term: idx for idx, term in enumerate(top_go_terms)}
    labels = np.zeros((len(protein_ids), len(top_go_terms)), dtype=np.float32)

    for i, protein_id in enumerate(protein_ids):
        if protein_id in protein_go_map:
            for go_term in protein_go_map[protein_id]:
                j = go_term_to_idx[go_term]
                labels[i, j] = 1.0

    stats = {
        'mapping_stats': mapping_stats,
        'aspect_stats': aspect_stats
    }

    return labels, top_go_terms, stats


# ============================================
# 6. Train/Val划分
# ============================================
def split_train_val(
    sequences_df: pd.DataFrame,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    remove_empty_labels: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    
    np.random.seed(random_seed)
    n_samples = len(sequences_df)
    indices = np.random.permutation(n_samples)

    n_val = int(n_samples * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_df = sequences_df.iloc[train_indices].reset_index(drop=True)
    val_df = sequences_df.iloc[val_indices].reset_index(drop=True)
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    filter_stats = {}

    if remove_empty_labels:
        # 训练集过滤
        train_has_labels = train_labels.sum(axis=1) > 0
        train_df = train_df[train_has_labels].reset_index(drop=True)
        train_labels = train_labels[train_has_labels]
        
        # 验证集过滤
        val_has_labels = val_labels.sum(axis=1) > 0
        val_df = val_df[val_has_labels].reset_index(drop=True)
        val_labels = val_labels[val_has_labels]
        
        filter_stats['train_after'] = len(train_df)
        filter_stats['val_after'] = len(val_df)

    return train_df, val_df, train_labels, val_labels, filter_stats


# ============================================
# 7. 主函数
# ============================================
def save_processed_data(
    data_dir: str,
    output_dir: str,
    top_k_per_aspect: int = 500,
    val_ratio: float = 0.1,
    use_hierarchy: bool = True,
    remove_empty_labels: bool = True
):
    """处理并保存数据"""
    print("\n" + "="*70)
    print("CAFA6 智能数据处理器 v3")
    print("="*70)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 构建文件路径
    ia_file = os.path.join(data_dir, "IA.tsv")
    train_sequences_file = os.path.join(data_dir, "Train", "train_sequences.fasta")
    train_terms_file = os.path.join(data_dir, "Train", "train_terms.tsv")
    go_obo_file = os.path.join(data_dir, "Train", "go-basic.obo")
    # 注意: 测试集路径可能因解压方式不同而异，这里假设在 Test/ 下
    test_sequences_file = os.path.join(data_dir, "Test", "testsuperset.fasta")
    
    # 检查必要文件是否存在
    if not os.path.exists(test_sequences_file):
        # 尝试另一种常见的测试集路径
        test_sequences_file = os.path.join(data_dir, "testsuperset.fasta")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取IA值
    go_to_ia = load_ia_values(ia_file)

    # 2. 解析GO本体
    go_parents, go_children, go_aspect = parse_go_ontology(go_obo_file)

    # 3. 读取序列
    train_seq = load_sequences(train_sequences_file, "训练序列")
    
    if os.path.exists(test_sequences_file):
        test_seq = load_sequences(test_sequences_file, "测试序列")
    else:
        print(f"⚠️  警告: 未找到测试集文件 ({test_sequences_file})，将跳过测试集处理")
        test_seq = pd.DataFrame({'EntryID': [], 'sequence': []})

    # 4. 读取GO标注
    print(f"\n读取GO标注: {train_terms_file}")
    go_annotations = pd.read_csv(train_terms_file, sep='\t')

    # 5. 构建标签矩阵
    labels, go_terms, stats = build_smart_label_matrix_v3(
        go_annotations,
        train_seq['EntryID'].tolist(),
        go_to_ia,
        go_parents,
        go_aspect,
        top_k_per_aspect=top_k_per_aspect,
        use_hierarchy=use_hierarchy
    )

    # 6. Train/Val划分
    train_df, val_df, train_labels, val_labels, filter_stats = split_train_val(
        train_seq, labels, val_ratio=val_ratio, remove_empty_labels=remove_empty_labels
    )

    # 7. 获取辅助信息
    ia_weights = np.array([go_to_ia[go] for go in go_terms])
    go_aspects = np.array([go_aspect.get(go, 'Unknown') for go in go_terms])

    # 8. 保存
    print(f"\n{'='*70}")
    print("保存数据")
    print('='*70)

    npz_file = os.path.join(output_dir, "cafa6_data.npz")
    np.savez_compressed(
        npz_file,
        train_sequences=train_df['sequence'].values,
        train_labels=train_labels,
        train_ids=train_df['EntryID'].values,
        val_sequences=val_df['sequence'].values,
        val_labels=val_labels,
        val_ids=val_df['EntryID'].values,
        test_sequences=test_seq['sequence'].values,
        test_ids=test_seq['EntryID'].values,
        go_terms=np.array(go_terms, dtype=object),
        ia_weights=ia_weights,
        go_aspects=go_aspects
    )

    print(f"✓ 数据已保存到: {npz_file}")
    
    # 保存元数据
    pkl_file = os.path.join(output_dir, "cafa6_metadata.pkl")
    metadata = {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_seq,
        'go_terms': go_terms,
        'ia_weights': ia_weights,
        'go_aspects': go_aspects,
        'stats': stats
    }
    with open(pkl_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ 元数据已保存到: {pkl_file}")
    print(f"\n完成! 训练集: {len(train_df)}, 验证集: {len(val_df)}")


# ============================================
# 8. 快速读取函数 (供其他脚本使用)
# ============================================
def load_processed_data(data_dir: str):
    """快速读取处理好的数据"""
    npz_file = os.path.join(data_dir, "cafa6_data.npz")
    pkl_file = os.path.join(data_dir, "cafa6_metadata.pkl")

    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"数据文件不存在: {npz_file}")

    print(f"读取: {npz_file}")
    npz_data = np.load(npz_file, allow_pickle=True)
    
    metadata = {}
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            metadata = pickle.load(f)

    return {
        'train': {'sequences': npz_data['train_sequences'], 'labels': npz_data['train_labels'], 'ids': npz_data['train_ids']},
        'val': {'sequences': npz_data['val_sequences'], 'labels': npz_data['val_labels'], 'ids': npz_data['val_ids']},
        'test': {'sequences': npz_data['test_sequences'], 'ids': npz_data['test_ids']},
        'go_terms': npz_data['go_terms'],
        'ia_weights': npz_data['ia_weights'],
        'go_aspects': npz_data['go_aspects'],
        'metadata': metadata
    }


if __name__ == "__main__":
    # 命令行入口
    parser = argparse.ArgumentParser(description="CAFA6 数据预处理")
    parser.add_argument('--data_dir', type=str, default='cafa-6-protein-function-prediction', 
                        help='原始数据目录 (Train/Test所在目录)')
    parser.add_argument('--output_dir', type=str, default='processed_data_v3',
                        help='输出目录')
    parser.add_argument('--top_k', type=int, default=500,
                        help='每个Aspect保留的Top-K GO术语')
    
    args = parser.parse_args()
    
    save_processed_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        top_k_per_aspect=args.top_k
    )