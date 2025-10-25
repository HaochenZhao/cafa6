"""
改进版数据处理器 v3 - 分aspect处理
1. 过滤IA=0的GO术语
2. 按照C、F、P三个aspect分别选择Top-K（默认各500）
3. 智能层次化：在同一aspect内映射到最近的Top-K祖先
"""

import numpy as np
import pandas as pd
import pickle
import os
from collections import defaultdict, deque
from Bio import SeqIO
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 配置路径
# ============================================
BASE_DIR = r"c:\Users\86150\Desktop\MachineLearningPractice\CAFA6\cafa-6-protein-function-prediction"

TRAIN_SEQUENCES = f"{BASE_DIR}/Train/train_sequences.fasta"
TRAIN_TERMS = f"{BASE_DIR}/Train/train_terms.tsv"
GO_OBO = f"{BASE_DIR}/Train/go-basic.obo"
TEST_SEQUENCES = f"{BASE_DIR}/Test/testsuperset.fasta"
IA_FILE = f"{BASE_DIR}/IA.tsv"


# ============================================
# 1. 读取IA值并过滤
# ============================================
def load_ia_values(ia_file: str, min_ia: float = 0.0) -> Dict[str, float]:
    """
    读取IA值，过滤掉IA=0的GO术语

    返回:
        go_to_ia: {GO_term: IA_value} 只包含IA > min_ia的
    """
    print(f"\n{'='*70}")
    print("读取IA值")
    print('='*70)

    ia_df = pd.read_csv(ia_file, sep='\t', header=None, names=['GO', 'IA'])

    print(f"✓ 总GO术语数: {len(ia_df):,}")
    print(f"✓ IA=0的GO: {(ia_df['IA'] == 0).sum():,} ({(ia_df['IA'] == 0).sum()/len(ia_df)*100:.1f}%)")
    print(f"✓ IA>0的GO: {(ia_df['IA'] > 0).sum():,} ({(ia_df['IA'] > 0).sum()/len(ia_df)*100:.1f}%)")

    # 过滤
    valid_ia = ia_df[ia_df['IA'] > min_ia]
    go_to_ia = dict(zip(valid_ia['GO'], valid_ia['IA']))

    print(f"\n✓ 保留的GO术语: {len(go_to_ia):,}")
    print(f"  - IA最大: {valid_ia['IA'].max():.2f}")
    print(f"  - IA最小: {valid_ia['IA'].min():.4f}")
    print(f"  - IA平均: {valid_ia['IA'].mean():.2f}")

    return go_to_ia


# ============================================
# 2. 解析GO本体（增强版：提取aspect信息）
# ============================================
def parse_go_ontology(obo_file: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str]]:
    """
    解析GO本体，提取父子关系和aspect信息

    返回:
        go_parents: {child: set(parents)}
        go_children: {parent: set(children)}
        go_aspect: {go_term: aspect} aspect为'C', 'F', 'P'之一
    """
    print(f"\n{'='*70}")
    print("解析GO本体")
    print('='*70)

    go_parents = defaultdict(set)
    go_children = defaultdict(set)
    go_aspect = {}

    current_go = None
    current_namespace = None
    is_obsolete = False

    # namespace到aspect的映射
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

            # 保存aspect信息
            if current_go and current_namespace and not is_obsolete:
                go_aspect[current_go] = current_namespace

    print(f"✓ 解析了 {len(go_parents):,} 个GO术语的层次关系")
    print(f"✓ 解析了 {len(go_aspect):,} 个GO术语的aspect信息")

    # 统计各aspect的数量
    aspect_counts = defaultdict(int)
    for aspect in go_aspect.values():
        aspect_counts[aspect] += 1

    print(f"\nAspect分布:")
    for aspect in ['C', 'F', 'P']:
        count = aspect_counts.get(aspect, 0)
        print(f"  - {aspect}: {count:,}")

    return go_parents, go_children, go_aspect


# ============================================
# 3. 找到最近的Top-K祖先（同一aspect内）
# ============================================
def find_nearest_topk_ancestors(
    go_term: str,
    go_parents: Dict[str, Set[str]],
    topk_gos: Set[str],
    go_aspect: Dict[str, str]
) -> Set[str]:
    """
    对于不在Top-K中的GO术语，找到最近的Top-K祖先

    重要改进：只在同一aspect内查找祖先

    策略：广度优先搜索，找到第一批在Top-K中且同一aspect的祖先
    """
    if go_term in topk_gos:
        return {go_term}

    # 获取当前GO的aspect
    current_aspect = go_aspect.get(go_term)
    if current_aspect is None:
        return set()  # 如果找不到aspect，返回空集

    ancestors = set()
    visited = set()
    queue = deque([go_term])

    # BFS查找
    while queue:
        current = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        # 如果当前节点在Top-K中，且aspect相同，记录它
        if current in topk_gos:
            # 验证aspect是否相同
            if go_aspect.get(current) == current_aspect:
                ancestors.add(current)
                continue  # 不再向上搜索

        # 继续向上搜索父节点
        if current in go_parents:
            for parent in go_parents[current]:
                if parent not in visited:
                    queue.append(parent)

    return ancestors


# ============================================
# 4. 读取序列
# ============================================
def load_sequences(fasta_file: str, name: str = "数据") -> pd.DataFrame:
    """读取FASTA文件"""
    print(f"\n{'='*70}")
    print(f"读取{name}")
    print('='*70)

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
    print(f"✓ 序列长度: {df['seq_length'].min()} - {df['seq_length'].max()} (平均: {df['seq_length'].mean():.1f})")

    return df


# ============================================
# 5. 构建智能标签矩阵（v3：分aspect处理）
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
    """
    构建智能标签矩阵 v3

    策略：
    1. 只考虑IA>0的GO术语
    2. 按C、F、P三个aspect分别选择Top-K个最常见的GO术语
    3. 对于不在Top-K的GO：在同一aspect内映射到最近的Top-K祖先
    """
    print(f"\n{'='*70}")
    print("构建智能标签矩阵 v3 (分aspect处理)")
    print('='*70)

    # 1. 过滤：只保留IA>0的标注
    print(f"\n步骤1: 过滤IA=0的GO术语")
    original_count = len(go_annotations)
    go_annotations = go_annotations[go_annotations['term'].isin(go_to_ia.keys())]
    print(f"  原始标注: {original_count:,}")
    print(f"  过滤后: {len(go_annotations):,}")
    print(f"  过滤掉: {original_count - len(go_annotations):,} ({(original_count - len(go_annotations))/original_count*100:.1f}%)")

    # 2. 按aspect分组并选择Top-K
    print(f"\n步骤2: 按aspect分别选择Top-{top_k_per_aspect} GO术语")

    # 为每个GO添加aspect信息
    go_annotations['aspect'] = go_annotations['term'].map(go_aspect)
    # 过滤掉没有aspect信息的
    go_annotations = go_annotations[go_annotations['aspect'].notna()]

    top_go_terms = []
    aspect_stats = {}

    for aspect in ['C', 'F', 'P']:
        # 筛选当前aspect的标注
        aspect_annotations = go_annotations[go_annotations['aspect'] == aspect]

        if len(aspect_annotations) == 0:
            print(f"\n  [{aspect}] 无标注数据，跳过")
            aspect_stats[aspect] = {'count': 0, 'top_go': []}
            continue

        # 统计频率
        go_counts = aspect_annotations['term'].value_counts()

        # 选择Top-K
        k = min(top_k_per_aspect, len(go_counts))
        aspect_top_go = go_counts.head(k).index.tolist()
        top_go_terms.extend(aspect_top_go)

        # 统计信息
        aspect_stats[aspect] = {
            'count': k,
            'top_go': aspect_top_go,
            'total_annotations': len(aspect_annotations),
            'unique_gos': len(go_counts)
        }

        print(f"\n  [{aspect}] Aspect统计:")
        print(f"    - 标注数: {len(aspect_annotations):,}")
        print(f"    - 唯一GO数: {len(go_counts):,}")
        print(f"    - 选择Top-K: {k}")
        print(f"    - 最常见: {aspect_top_go[0]} (出现 {go_counts.iloc[0]:,} 次, IA={go_to_ia[aspect_top_go[0]]:.2f})")
        if k > 0:
            print(f"    - 第{k}名: {aspect_top_go[-1]} (出现 {go_counts.iloc[k-1]:,} 次, IA={go_to_ia[aspect_top_go[-1]]:.2f})")

    topk_gos = set(top_go_terms)
    print(f"\n  总共选择了 {len(top_go_terms)} 个GO术语")

    # 3. 构建蛋白质->GO映射（使用层次化）
    print(f"\n步骤3: 构建蛋白质-GO映射 (层次化={use_hierarchy})")
    protein_go_map = defaultdict(set)
    mapping_stats = {
        'direct': {'C': 0, 'F': 0, 'P': 0},
        'mapped': {'C': 0, 'F': 0, 'P': 0},
        'lost': {'C': 0, 'F': 0, 'P': 0}
    }

    for _, row in go_annotations.iterrows():
        protein_id = row['EntryID']
        go_term = row['term']
        aspect = row['aspect']

        if go_term in topk_gos:
            # 直接在Top-K中
            protein_go_map[protein_id].add(go_term)
            mapping_stats['direct'][aspect] += 1
        elif use_hierarchy:
            # 使用层次化：找到最近的Top-K祖先（同一aspect内）
            nearest_ancestors = find_nearest_topk_ancestors(
                go_term, go_parents, topk_gos, go_aspect
            )
            if nearest_ancestors:
                protein_go_map[protein_id].update(nearest_ancestors)
                mapping_stats['mapped'][aspect] += len(nearest_ancestors)
            else:
                mapping_stats['lost'][aspect] += 1
        else:
            # 不使用层次化：丢弃
            mapping_stats['lost'][aspect] += 1

    print(f"\n  映射统计 (按aspect):")
    for aspect in ['C', 'F', 'P']:
        print(f"  [{aspect}]")
        print(f"    - 直接匹配: {mapping_stats['direct'][aspect]:,}")
        if use_hierarchy:
            print(f"    - 层次化映射: {mapping_stats['mapped'][aspect]:,}")
        print(f"    - 丢失: {mapping_stats['lost'][aspect]:,}")

    # 4. 创建标签矩阵
    print(f"\n步骤4: 创建标签矩阵")
    go_term_to_idx = {term: idx for idx, term in enumerate(top_go_terms)}
    labels = np.zeros((len(protein_ids), len(top_go_terms)), dtype=np.float32)

    for i, protein_id in enumerate(protein_ids):
        if protein_id in protein_go_map:
            for go_term in protein_go_map[protein_id]:
                j = go_term_to_idx[go_term]
                labels[i, j] = 1.0

    # 5. 统计
    labels_per_protein = labels.sum(axis=1)
    print(f"  标签矩阵形状: {labels.shape}")
    print(f"  平均每个蛋白质: {labels_per_protein.mean():.2f} 个标签")
    print(f"  有标签的蛋白质: {(labels_per_protein > 0).sum():,} / {len(protein_ids):,} ({(labels_per_protein > 0).sum()/len(protein_ids)*100:.2f}%)")
    print(f"  无标签的蛋白质: {(labels_per_protein == 0).sum():,} ({(labels_per_protein == 0).sum()/len(protein_ids)*100:.2f}%)")

    # 按aspect统计标签数
    print(f"\n  按aspect统计标签数:")
    for aspect in ['C', 'F', 'P']:
        aspect_go_indices = [i for i, go in enumerate(top_go_terms) if go_aspect.get(go) == aspect]
        if aspect_go_indices:
            aspect_labels = labels[:, aspect_go_indices].sum(axis=1)
            avg_labels = aspect_labels.mean()
            print(f"  [{aspect}] 平均每个蛋白质: {avg_labels:.2f} 个标签")

    # 返回额外的统计信息
    stats = {
        'mapping_stats': mapping_stats,
        'labels_per_protein': labels_per_protein,
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
    """
    划分训练集和验证集

    参数:
        remove_empty_labels: 是否移除无标签的样本（默认True）
                           注意：只对训练集和验证集过滤，测试集保持完整
    """
    print(f"\n{'='*70}")
    print(f"划分训练集/验证集 (验证集比例: {val_ratio:.1%})")
    print('='*70)

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

    print(f"\n划分前:")
    print(f"  - 训练集: {len(train_df):,}")
    print(f"  - 验证集: {len(val_df):,}")

    filter_stats = {
        'train_before': len(train_df),
        'val_before': len(val_df),
        'train_after': len(train_df),
        'val_after': len(val_df),
        'train_removed': 0,
        'val_removed': 0
    }

    # 过滤无标签样本
    if remove_empty_labels:
        print(f"\n过滤无标签样本...")

        # 训练集过滤
        train_has_labels = train_labels.sum(axis=1) > 0
        train_empty_count = (~train_has_labels).sum()

        if train_empty_count > 0:
            train_df = train_df[train_has_labels].reset_index(drop=True)
            train_labels = train_labels[train_has_labels]
            filter_stats['train_removed'] = train_empty_count
            filter_stats['train_after'] = len(train_df)
            print(f"  - 训练集移除无标签样本: {train_empty_count:,}")
        else:
            print(f"  - 训练集无需过滤（所有样本都有标签）")

        # 验证集过滤
        val_has_labels = val_labels.sum(axis=1) > 0
        val_empty_count = (~val_has_labels).sum()

        if val_empty_count > 0:
            val_df = val_df[val_has_labels].reset_index(drop=True)
            val_labels = val_labels[val_has_labels]
            filter_stats['val_removed'] = val_empty_count
            filter_stats['val_after'] = len(val_df)
            print(f"  - 验证集移除无标签样本: {val_empty_count:,}")
        else:
            print(f"  - 验证集无需过滤（所有样本都有标签）")

    print(f"\n过滤后:")
    print(f"  - 训练集: {len(train_df):,}")
    print(f"  - 验证集: {len(val_df):,}")

    return train_df, val_df, train_labels, val_labels, filter_stats


# ============================================
# 7. 主函数：处理并保存数据
# ============================================
def save_processed_data(
    output_dir: str = "processed_data_v3",
    top_k_per_aspect: int = 500,
    val_ratio: float = 0.1,
    use_hierarchy: bool = True,
    remove_empty_labels: bool = True
):
    """
    处理并保存数据 v3

    参数:
        remove_empty_labels: 是否移除训练集和验证集中的无标签样本（默认True）
    """
    print("\n" + "="*70)
    print("CAFA6 智能数据处理器 v3 (分aspect处理)")
    print("="*70)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取IA值
    go_to_ia = load_ia_values(IA_FILE)

    # 2. 解析GO本体（包含aspect信息）
    go_parents, go_children, go_aspect = parse_go_ontology(GO_OBO)

    # 3. 读取序列
    train_seq = load_sequences(TRAIN_SEQUENCES, "训练序列")
    test_seq = load_sequences(TEST_SEQUENCES, "测试序列")

    # 4. 读取GO标注
    print(f"\n{'='*70}")
    print("读取GO标注")
    print('='*70)
    go_annotations = pd.read_csv(TRAIN_TERMS, sep='\t')
    print(f"✓ GO标注: {len(go_annotations):,} 条")

    # 5. 构建标签矩阵 v3
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

    # 7. 获取GO的IA权重和aspect信息
    ia_weights = np.array([go_to_ia[go] for go in go_terms])
    go_aspects = np.array([go_aspect.get(go, 'Unknown') for go in go_terms])

    # 8. 保存为 .npz
    print(f"\n{'='*70}")
    print("保存数据")
    print('='*70)

    npz_file = os.path.join(output_dir, "cafa6_data.npz")
    np.savez_compressed(
        npz_file,
        # 训练集
        train_sequences=train_df['sequence'].values,
        train_labels=train_labels,
        train_ids=train_df['EntryID'].values,
        # 验证集
        val_sequences=val_df['sequence'].values,
        val_labels=val_labels,
        val_ids=val_df['EntryID'].values,
        # 测试集
        test_sequences=test_seq['sequence'].values,
        test_ids=test_seq['EntryID'].values,
        # GO术语、IA权重和aspect信息
        go_terms=np.array(go_terms, dtype=object),
        ia_weights=ia_weights,
        go_aspects=go_aspects
    )

    file_size_mb = os.path.getsize(npz_file) / (1024 * 1024)
    print(f"✓ 保存到: {npz_file}")
    print(f"✓ 文件大小: {file_size_mb:.2f} MB")

    # 9. 保存元数据
    pkl_file = os.path.join(output_dir, "cafa6_metadata.pkl")
    metadata = {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_seq,
        'go_terms': go_terms,
        'ia_weights': ia_weights,
        'go_aspects': go_aspects,
        'go_to_ia': go_to_ia,
        'go_aspect': go_aspect,
        'use_hierarchy': use_hierarchy,
        'top_k_per_aspect': top_k_per_aspect,
        'val_ratio': val_ratio,
        'remove_empty_labels': remove_empty_labels,
        'stats': stats,
        'filter_stats': filter_stats
    }

    with open(pkl_file, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✓ 保存元数据: {pkl_file}")

    # 10. 保存统计信息
    stats_file = os.path.join(output_dir, "data_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CAFA6 数据统计 (v3 - 分aspect处理)\n")
        f.write("="*70 + "\n\n")

        f.write(f"配置:\n")
        f.write(f"  - 每个aspect的Top-K: {top_k_per_aspect}\n")
        f.write(f"  - 总GO术语数: {len(go_terms)}\n")
        f.write(f"  - 验证集比例: {val_ratio:.1%}\n")
        f.write(f"  - 使用层次化: {use_hierarchy}\n")
        f.write(f"  - 过滤IA=0: 是\n")
        f.write(f"  - 移除无标签样本: {'是' if remove_empty_labels else '否'}\n\n")

        f.write(f"数据量:\n")
        if remove_empty_labels and filter_stats['train_removed'] > 0:
            f.write(f"  - 训练集 (过滤前): {filter_stats['train_before']:,}\n")
            f.write(f"  - 训练集 (过滤后): {len(train_df):,} (移除 {filter_stats['train_removed']:,} 个无标签样本)\n")
            f.write(f"  - 验证集 (过滤前): {filter_stats['val_before']:,}\n")
            f.write(f"  - 验证集 (过滤后): {len(val_df):,} (移除 {filter_stats['val_removed']:,} 个无标签样本)\n")
        else:
            f.write(f"  - 训练集: {len(train_df):,}\n")
            f.write(f"  - 验证集: {len(val_df):,}\n")
        f.write(f"  - 测试集: {len(test_seq):,}\n\n")

        f.write(f"GO术语分布 (按aspect):\n")
        for aspect in ['C', 'F', 'P']:
            count = (go_aspects == aspect).sum()
            f.write(f"  - {aspect}: {count}\n")
        f.write("\n")

        f.write(f"GO映射统计 (按aspect):\n")
        for aspect in ['C', 'F', 'P']:
            f.write(f"  [{aspect}]\n")
            f.write(f"    - 直接匹配: {stats['mapping_stats']['direct'][aspect]:,}\n")
            if use_hierarchy:
                f.write(f"    - 层次化映射: {stats['mapping_stats']['mapped'][aspect]:,}\n")
            f.write(f"    - 丢失: {stats['mapping_stats']['lost'][aspect]:,}\n")
        f.write("\n")

        f.write(f"标签统计:\n")
        train_lc = train_labels.sum(axis=1)
        val_lc = val_labels.sum(axis=1)
        f.write(f"  - 训练集平均标签数: {train_lc.mean():.2f}\n")
        f.write(f"  - 训练集有标签样本: {(train_lc > 0).sum():,} ({(train_lc > 0).sum()/len(train_lc)*100:.2f}%)\n")
        f.write(f"  - 验证集平均标签数: {val_lc.mean():.2f}\n")
        f.write(f"  - 验证集有标签样本: {(val_lc > 0).sum():,} ({(val_lc > 0).sum()/len(val_lc)*100:.2f}%)\n")

    print(f"✓ 保存统计信息: {stats_file}")

    print("\n" + "="*70)
    print("保存完成！")
    print("="*70)
    print(f"\nv3 关键改进:")
    print(f"  ✓ 过滤了IA=0的GO术语")
    print(f"  ✓ 按C、F、P三个aspect分别选择Top-{top_k_per_aspect}")
    print(f"  ✓ 同一aspect内的智能层次化映射")
    print(f"  ✓ 包含aspect信息用于后续分析")
    print(f"  ✓ 总GO术语数: {len(go_terms)}")
    if remove_empty_labels and filter_stats['train_removed'] > 0:
        print(f"  ✓ 移除了无标签样本 (训练集: {filter_stats['train_removed']}, 验证集: {filter_stats['val_removed']})")

    return metadata


# ============================================
# 8. 快速读取
# ============================================
def load_processed_data(data_dir: str = "processed_data_v3"):
    """快速读取处理好的数据"""
    print("="*70)
    print("读取处理后的数据 v3")
    print("="*70)

    npz_file = os.path.join(data_dir, "cafa6_data.npz")
    pkl_file = os.path.join(data_dir, "cafa6_metadata.pkl")

    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"数据文件不存在: {npz_file}")

    # 读取主数据
    print(f"\n读取: {npz_file}")
    npz_data = np.load(npz_file, allow_pickle=True)

    # 读取元数据
    metadata = {}
    if os.path.exists(pkl_file):
        print(f"读取: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            metadata = pickle.load(f)

    # 组装数据
    data = {
        'train': {
            'sequences': npz_data['train_sequences'],
            'labels': npz_data['train_labels'],
            'ids': npz_data['train_ids'],
            'df': metadata.get('train_df', None)
        },
        'val': {
            'sequences': npz_data['val_sequences'],
            'labels': npz_data['val_labels'],
            'ids': npz_data['val_ids'],
            'df': metadata.get('val_df', None)
        },
        'test': {
            'sequences': npz_data['test_sequences'],
            'ids': npz_data['test_ids'],
            'df': metadata.get('test_df', None)
        },
        'go_terms': npz_data['go_terms'].tolist(),
        'ia_weights': npz_data['ia_weights'],
        'go_aspects': npz_data['go_aspects'],
        'metadata': metadata
    }

    print("\n✓ 数据读取完成！")
    print(f"\n数据规模:")
    print(f"  - 训练集: {len(data['train']['ids']):,}")
    print(f"  - 验证集: {len(data['val']['ids']):,}")
    print(f"  - 测试集: {len(data['test']['ids']):,}")
    print(f"  - GO术语: {len(data['go_terms'])}")
    print(f"  - IA权重范围: {data['ia_weights'].min():.4f} - {data['ia_weights'].max():.2f}")

    # 显示aspect分布
    print(f"\nGO术语aspect分布:")
    for aspect in ['C', 'F', 'P']:
        count = (data['go_aspects'] == aspect).sum()
        print(f"  - {aspect}: {count}")

    return data


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 处理并保存
    if not os.path.exists("processed_data_v3/cafa6_data.npz"):
        print("首次运行，正在处理数据...\n")
        save_processed_data(
            output_dir="processed_data_v3",
            top_k_per_aspect=500,
            val_ratio=0.1,
            use_hierarchy=True
        )
    else:
        print("数据已存在，跳过处理。\n")

    # 读取数据
    data = load_processed_data("processed_data_v3")

    # 示例使用
    print("\n" + "="*70)
    print("使用示例")
    print("="*70)

    idx = 0
    print(f"\n第1个训练样本:")
    print(f"  ID: {data['train']['ids'][idx]}")
    print(f"  序列: {data['train']['sequences'][idx][:50]}...")
    print(f"  标签数: {int(data['train']['labels'][idx].sum())}")

    label_indices = np.where(data['train']['labels'][idx] == 1)[0]
    if len(label_indices) > 0:
        print(f"  GO术语及其IA值和aspect:")
        for i in label_indices[:5]:
            print(f"    - {data['go_terms'][i]}: IA={data['ia_weights'][i]:.2f}, Aspect={data['go_aspects'][i]}")
