"""
CAFA6 简单数据加载器
只负责：读取原始数据 + train/val划分 + 基本统计
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from Bio import SeqIO
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 配置路径
# ============================================
BASE_DIR = r"c:\Users\86150\Desktop\MachineLearningPractice\CAFA6\cafa-6-protein-function-prediction"

# 训练数据
TRAIN_SEQUENCES = f"{BASE_DIR}/Train/train_sequences.fasta"
TRAIN_TERMS = f"{BASE_DIR}/Train/train_terms.tsv"
TRAIN_TAXONOMY = f"{BASE_DIR}/Train/train_taxonomy.tsv"

# 测试数据
TEST_SEQUENCES = f"{BASE_DIR}/Test/testsuperset.fasta"

# ============================================
# 1. 读取FASTA格式的蛋白质序列
# ============================================
def load_sequences(fasta_file: str, name: str = "数据") -> pd.DataFrame:
    """
    读取FASTA文件，返回DataFrame

    返回:
        DataFrame with columns: [EntryID, sequence, seq_length]
    """
    print(f"\n{'='*60}")
    print(f"读取{name}: {fasta_file}")
    print('='*60)

    sequences = []
    ids = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        # 解析蛋白质ID
        if '|' in record.id:
            # 格式: sp|A0A0C5B5G6|MOTSC_HUMAN
            protein_id = record.id.split('|')[1]
        else:
            # 格式: A0A0C5B5G6 9606
            protein_id = record.id.split()[0]

        sequence = str(record.seq)
        ids.append(protein_id)
        sequences.append(sequence)

    df = pd.DataFrame({
        'EntryID': ids,
        'sequence': sequences,
        'seq_length': [len(s) for s in sequences]
    })

    print(f"✓ 蛋白质数量: {len(df)}")
    print(f"✓ 序列长度统计:")
    print(f"  - 最短: {df['seq_length'].min()}")
    print(f"  - 最长: {df['seq_length'].max()}")
    print(f"  - 平均: {df['seq_length'].mean():.1f}")
    print(f"  - 中位数: {df['seq_length'].median():.1f}")

    return df


# ============================================
# 2. 读取GO术语标注
# ============================================
def load_go_annotations(terms_file: str) -> pd.DataFrame:
    """
    读取GO术语标注

    返回:
        DataFrame with columns: [EntryID, term, aspect]
    """
    print(f"\n{'='*60}")
    print(f"读取GO标注: {terms_file}")
    print('='*60)

    df = pd.read_csv(terms_file, sep='\t')

    print(f"✓ GO标注总数: {len(df)}")
    print(f"✓ 涉及蛋白质数: {df['EntryID'].nunique()}")
    print(f"✓ 不同GO术语数: {df['term'].nunique()}")
    print(f"✓ GO类别分布:")
    for aspect, count in df['aspect'].value_counts().items():
        aspect_name = {'P': '生物过程', 'F': '分子功能', 'C': '细胞组分'}.get(aspect, aspect)
        print(f"  - {aspect} ({aspect_name}): {count}")

    return df


# ============================================
# 3. 构建多标签矩阵
# ============================================
def build_label_matrix(
    go_annotations: pd.DataFrame,
    protein_ids: List[str],
    top_k: int = 500
) -> Tuple[np.ndarray, List[str]]:
    """
    构建多标签矩阵（Top-K GO术语）

    返回:
        labels: (N, K) 二进制矩阵
        go_terms: K个GO术语列表
    """
    print(f"\n{'='*60}")
    print(f"构建Top-{top_k}多标签矩阵")
    print('='*60)

    # 1. 统计GO术语频率
    go_counts = go_annotations['term'].value_counts()
    top_go_terms = go_counts.head(top_k).index.tolist()

    print(f"✓ 选择Top-{top_k}个最常见GO术语")
    print(f"  - 最常见: {top_go_terms[0]} (出现 {go_counts.iloc[0]} 次)")
    print(f"  - 第{top_k}名: {top_go_terms[-1]} (出现 {go_counts.iloc[top_k-1]} 次)")

    # 2. 构建蛋白质->GO映射
    protein_go_map = defaultdict(set)
    for _, row in go_annotations.iterrows():
        if row['term'] in top_go_terms:
            protein_go_map[row['EntryID']].add(row['term'])

    # 3. 创建标签矩阵
    go_term_to_idx = {term: idx for idx, term in enumerate(top_go_terms)}
    labels = np.zeros((len(protein_ids), top_k), dtype=np.float32)

    for i, protein_id in enumerate(protein_ids):
        if protein_id in protein_go_map:
            for go_term in protein_go_map[protein_id]:
                j = go_term_to_idx[go_term]
                labels[i, j] = 1.0

    # 4. 统计
    labels_per_protein = labels.sum(axis=1)
    print(f"✓ 标签统计:")
    print(f"  - 每个蛋白质平均标签数: {labels_per_protein.mean():.2f}")
    print(f"  - 标签数范围: {int(labels_per_protein.min())} - {int(labels_per_protein.max())}")
    print(f"  - 有标签的蛋白质: {(labels_per_protein > 0).sum()} / {len(protein_ids)}")
    print(f"  - 无标签的蛋白质: {(labels_per_protein == 0).sum()}")

    return labels, top_go_terms


# ============================================
# 4. Train/Val划分
# ============================================
def split_train_val(
    sequences_df: pd.DataFrame,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    划分训练集和验证集

    返回:
        train_df, val_df, train_labels, val_labels
    """
    print(f"\n{'='*60}")
    print(f"划分训练集/验证集 (验证集比例: {val_ratio:.1%})")
    print('='*60)

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

    print(f"✓ 训练集: {len(train_df)} 个样本")
    print(f"✓ 验证集: {len(val_df)} 个样本")

    # 训练集标签统计
    train_label_counts = train_labels.sum(axis=1)
    print(f"\n训练集标签统计:")
    print(f"  - 平均每个蛋白质: {train_label_counts.mean():.2f} 个标签")
    print(f"  - 有标签的样本: {(train_label_counts > 0).sum()}")

    # 验证集标签统计
    val_label_counts = val_labels.sum(axis=1)
    print(f"\n验证集标签统计:")
    print(f"  - 平均每个蛋白质: {val_label_counts.mean():.2f} 个标签")
    print(f"  - 有标签的样本: {(val_label_counts > 0).sum()}")

    return train_df, val_df, train_labels, val_labels


# ============================================
# 5. 数据统计
# ============================================
def print_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    go_terms: List[str]
):
    """
    打印详细的数据统计信息
    """
    print(f"\n{'='*60}")
    print("数据统计总览")
    print('='*60)

    print(f"\n【样本数量】")
    print(f"  训练集: {len(train_df):>8}")
    print(f"  验证集: {len(val_df):>8}")
    print(f"  测试集: {len(test_df):>8}")
    print(f"  总计:   {len(train_df) + len(val_df) + len(test_df):>8}")

    print(f"\n【序列长度】")
    all_sequences = pd.concat([train_df, val_df, test_df])
    print(f"  最短序列: {all_sequences['seq_length'].min()}")
    print(f"  最长序列: {all_sequences['seq_length'].max()}")
    print(f"  平均长度: {all_sequences['seq_length'].mean():.1f}")
    print(f"  长度中位数: {all_sequences['seq_length'].median():.1f}")
    print(f"  长度标准差: {all_sequences['seq_length'].std():.1f}")

    print(f"\n【标签分布】")
    print(f"  GO术语总数: {len(go_terms)}")

    # 训练集标签分析
    train_label_sum = train_labels.sum(axis=1)
    print(f"\n  训练集:")
    print(f"    - 有标签样本: {(train_label_sum > 0).sum()} ({(train_label_sum > 0).sum()/len(train_df)*100:.1f}%)")
    print(f"    - 无标签样本: {(train_label_sum == 0).sum()} ({(train_label_sum == 0).sum()/len(train_df)*100:.1f}%)")
    print(f"    - 平均标签数: {train_label_sum.mean():.2f}")
    print(f"    - 标签数中位数: {np.median(train_label_sum):.0f}")

    # 验证集标签分析
    val_label_sum = val_labels.sum(axis=1)
    print(f"\n  验证集:")
    print(f"    - 有标签样本: {(val_label_sum > 0).sum()} ({(val_label_sum > 0).sum()/len(val_df)*100:.1f}%)")
    print(f"    - 无标签样本: {(val_label_sum == 0).sum()} ({(val_label_sum == 0).sum()/len(val_df)*100:.1f}%)")
    print(f"    - 平均标签数: {val_label_sum.mean():.2f}")
    print(f"    - 标签数中位数: {np.median(val_label_sum):.0f}")

    # 每个GO术语的频率
    train_go_freq = train_labels.sum(axis=0)
    print(f"\n  GO术语频率:")
    print(f"    - 最常见GO: {train_go_freq.max():.0f} 个蛋白质")
    print(f"    - 最少见GO: {train_go_freq.min():.0f} 个蛋白质")
    print(f"    - 平均每个GO: {train_go_freq.mean():.0f} 个蛋白质")

    print(f"\n{'='*60}")


# ============================================
# 主函数
# ============================================
def load_cafa6_data(top_k_go: int = 500, val_ratio: float = 0.1):
    """
    加载CAFA6数据集

    参数:
        top_k_go: 选择最常见的K个GO术语
        val_ratio: 验证集比例

    返回:
        data: dict containing all datasets and labels
    """
    print("\n" + "="*60)
    print("CAFA6 数据加载器")
    print("="*60)

    # 1. 读取序列
    train_seq = load_sequences(TRAIN_SEQUENCES, "训练序列")
    test_seq = load_sequences(TEST_SEQUENCES, "测试序列")

    # 2. 读取GO标注
    go_annotations = load_go_annotations(TRAIN_TERMS)

    # 3. 构建标签矩阵
    labels, go_terms = build_label_matrix(
        go_annotations,
        train_seq['EntryID'].tolist(),
        top_k=top_k_go
    )

    # 4. Train/Val划分
    train_df, val_df, train_labels, val_labels = split_train_val(
        train_seq, labels, val_ratio=val_ratio
    )

    # 5. 统计信息
    print_statistics(train_df, val_df, test_seq, train_labels, val_labels, go_terms)

    # 6. 返回数据
    data = {
        'train': {
            'df': train_df,
            'labels': train_labels,
            'ids': train_df['EntryID'].values,
            'sequences': train_df['sequence'].values
        },
        'val': {
            'df': val_df,
            'labels': val_labels,
            'ids': val_df['EntryID'].values,
            'sequences': val_df['sequence'].values
        },
        'test': {
            'df': test_seq,
            'ids': test_seq['EntryID'].values,
            'sequences': test_seq['sequence'].values
        },
        'go_terms': go_terms
    }

    print(f"\n✓ 数据加载完成！")
    print(f"✓ 使用方式:")
    print(f"    data['train']['sequences']  - 训练序列")
    print(f"    data['train']['labels']     - 训练标签")
    print(f"    data['val']['sequences']    - 验证序列")
    print(f"    data['test']['sequences']   - 测试序列")
    print(f"    data['go_terms']            - GO术语列表")

    return data


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 加载数据
    data = load_cafa6_data(top_k_go=500, val_ratio=0.1)

    # 访问数据
    print("\n" + "="*60)
    print("数据访问示例")
    print("="*60)

    print(f"\n第1个训练样本:")
    print(f"  - ID: {data['train']['ids'][0]}")
    print(f"  - 序列: {data['train']['sequences'][0][:50]}...")
    print(f"  - 序列长度: {len(data['train']['sequences'][0])}")
    print(f"  - 标签数: {data['train']['labels'][0].sum():.0f}")

    # 找到有标签的GO术语
    label_indices = np.where(data['train']['labels'][0] == 1)[0]
    if len(label_indices) > 0:
        print(f"  - GO术语示例: {[data['go_terms'][i] for i in label_indices]}")

    print(f"\n第1个测试样本:")
    print(f"  - ID: {data['test']['ids'][0]}")
    print(f"  - 序列: {data['test']['sequences'][0][:50]}...")
    print(f"  - 序列长度: {len(data['test']['sequences'][0])}")

    print("\n" + "="*60)
    print("完成！")
    print("="*60)
