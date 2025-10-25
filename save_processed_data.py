"""
保存处理后的数据为快速读取格式
支持：.npz (numpy压缩格式) 和 .pkl (pickle格式)
"""

import numpy as np
import pickle
from hierarchical_data_loader import load_cafa6_hierarchical
import os

# ============================================
# 保存数据
# ============================================
def save_processed_data(
    output_dir: str = "processed_data",
    top_k_go: int = 500,
    val_ratio: float = 0.1,
    use_hierarchy: bool = True
):
    """
    处理并保存数据

    参数:
        output_dir: 输出目录
        top_k_go: Top-K GO术语数量
        val_ratio: 验证集比例
        use_hierarchy: 是否使用层次化
    """
    print("="*70)
    print("保存处理后的数据")
    print("="*70)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ 创建目录: {output_dir}")

    # 1. 加载数据
    print(f"\n正在加载数据...")
    data = load_cafa6_hierarchical(
        top_k_go=top_k_go,
        val_ratio=val_ratio,
        use_hierarchy=use_hierarchy
    )

    # 2. 保存为 .npz 格式（推荐，压缩且快速）
    print(f"\n保存为 .npz 格式...")
    npz_file = os.path.join(output_dir, "cafa6_data.npz")

    np.savez_compressed(
        npz_file,
        # 训练集
        train_sequences=data['train']['sequences'],
        train_labels=data['train']['labels'],
        train_ids=data['train']['ids'],
        # 验证集
        val_sequences=data['val']['sequences'],
        val_labels=data['val']['labels'],
        val_ids=data['val']['ids'],
        # 测试集
        test_sequences=data['test']['sequences'],
        test_ids=data['test']['ids'],
        # GO术语
        go_terms=np.array(data['go_terms'], dtype=object)
    )

    file_size_mb = os.path.getsize(npz_file) / (1024 * 1024)
    print(f"✓ 保存到: {npz_file}")
    print(f"✓ 文件大小: {file_size_mb:.2f} MB")

    # 3. 保存元数据为 pickle（包含DataFrame等）
    print(f"\n保存元数据为 .pkl 格式...")
    pkl_file = os.path.join(output_dir, "cafa6_metadata.pkl")

    metadata = {
        'train_df': data['train']['df'],
        'val_df': data['val']['df'],
        'test_df': data['test']['df'],
        'go_terms': data['go_terms'],
        'use_hierarchy': use_hierarchy,
        'top_k_go': top_k_go,
        'val_ratio': val_ratio,
        'train_size': len(data['train']['ids']),
        'val_size': len(data['val']['ids']),
        'test_size': len(data['test']['ids'])
    }

    with open(pkl_file, 'wb') as f:
        pickle.dump(metadata, f)

    file_size_mb = os.path.getsize(pkl_file) / (1024 * 1024)
    print(f"✓ 保存到: {pkl_file}")
    print(f"✓ 文件大小: {file_size_mb:.2f} MB")

    # 4. 保存统计信息为文本文件
    print(f"\n保存统计信息...")
    stats_file = os.path.join(output_dir, "data_statistics.txt")

    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CAFA6 数据统计\n")
        f.write("="*70 + "\n\n")

        f.write(f"配置:\n")
        f.write(f"  - Top-K GO术语: {top_k_go}\n")
        f.write(f"  - 验证集比例: {val_ratio:.1%}\n")
        f.write(f"  - 使用层次化: {use_hierarchy}\n\n")

        f.write(f"数据量:\n")
        f.write(f"  - 训练集: {len(data['train']['ids']):,}\n")
        f.write(f"  - 验证集: {len(data['val']['ids']):,}\n")
        f.write(f"  - 测试集: {len(data['test']['ids']):,}\n")
        f.write(f"  - 总计: {len(data['train']['ids']) + len(data['val']['ids']) + len(data['test']['ids']):,}\n\n")

        f.write(f"标签统计:\n")
        train_label_counts = data['train']['labels'].sum(axis=1)
        val_label_counts = data['val']['labels'].sum(axis=1)
        f.write(f"  - 训练集平均标签数: {train_label_counts.mean():.2f}\n")
        f.write(f"  - 训练集有标签样本: {(train_label_counts > 0).sum():,} ({(train_label_counts > 0).sum()/len(train_label_counts)*100:.2f}%)\n")
        f.write(f"  - 验证集平均标签数: {val_label_counts.mean():.2f}\n")
        f.write(f"  - 验证集有标签样本: {(val_label_counts > 0).sum():,} ({(val_label_counts > 0).sum()/len(val_label_counts)*100:.2f}%)\n\n")

        f.write(f"序列长度:\n")
        all_seq_lens = [len(s) for s in data['train']['sequences']] + \
                       [len(s) for s in data['val']['sequences']] + \
                       [len(s) for s in data['test']['sequences']]
        f.write(f"  - 最短: {min(all_seq_lens)}\n")
        f.write(f"  - 最长: {max(all_seq_lens)}\n")
        f.write(f"  - 平均: {np.mean(all_seq_lens):.1f}\n")
        f.write(f"  - 中位数: {np.median(all_seq_lens):.1f}\n")

    print(f"✓ 保存到: {stats_file}")

    print("\n" + "="*70)
    print("保存完成！")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  1. {npz_file} - 主数据文件（.npz）")
    print(f"  2. {pkl_file} - 元数据文件（.pkl）")
    print(f"  3. {stats_file} - 统计信息（.txt）")
    print("\n使用 load_processed_data() 函数快速读取！")


# ============================================
# 快速读取数据
# ============================================
def load_processed_data(data_dir: str = "processed_data"):
    """
    快速读取处理好的数据

    参数:
        data_dir: 数据目录

    返回:
        data: dict 包含所有数据
    """
    print("="*70)
    print("快速读取处理后的数据")
    print("="*70)

    npz_file = os.path.join(data_dir, "cafa6_data.npz")
    pkl_file = os.path.join(data_dir, "cafa6_metadata.pkl")

    # 检查文件是否存在
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"数据文件不存在: {npz_file}\n请先运行 save_processed_data() 生成数据！")

    # 1. 读取主数据
    print(f"\n读取主数据: {npz_file}")
    npz_data = np.load(npz_file, allow_pickle=True)

    # 2. 读取元数据（如果存在）
    metadata = {}
    if os.path.exists(pkl_file):
        print(f"读取元数据: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            metadata = pickle.load(f)

    # 3. 组装数据
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
        'metadata': metadata
    }

    # 4. 打印信息
    print("\n✓ 数据读取完成！")
    print(f"\n数据规模:")
    print(f"  - 训练集: {len(data['train']['ids']):,} 样本")
    print(f"  - 验证集: {len(data['val']['ids']):,} 样本")
    print(f"  - 测试集: {len(data['test']['ids']):,} 样本")
    print(f"  - GO术语: {len(data['go_terms'])} 个")

    print(f"\n标签信息:")
    print(f"  - 训练标签形状: {data['train']['labels'].shape}")
    print(f"  - 验证标签形状: {data['val']['labels'].shape}")

    if metadata:
        print(f"\n配置信息:")
        print(f"  - Top-K GO: {metadata.get('top_k_go', 'N/A')}")
        print(f"  - 验证集比例: {metadata.get('val_ratio', 'N/A')}")
        print(f"  - 使用层次化: {metadata.get('use_hierarchy', 'N/A')}")

    print("\n" + "="*70)

    return data


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    import sys

    # 检查是否需要重新处理数据
    if not os.path.exists("processed_data/cafa6_data.npz"):
        print("首次运行，正在处理并保存数据...\n")
        save_processed_data(
            output_dir="processed_data",
            top_k_go=500,
            val_ratio=0.1,
            use_hierarchy=True  # 使用层次化方法
        )
    else:
        print("数据文件已存在，跳过处理步骤。")
        print("如需重新处理，请删除 processed_data 目录后重新运行。\n")

    # 快速读取数据
    print("\n" + "="*70)
    print("测试快速读取...")
    print("="*70)

    data = load_processed_data("processed_data")

    # 使用示例
    print("\n" + "="*70)
    print("使用示例")
    print("="*70)

    print(f"\n# 访问训练数据")
    print(f"train_sequences = data['train']['sequences']")
    print(f"train_labels = data['train']['labels']")
    print(f"train_ids = data['train']['ids']")

    print(f"\n# 示例：第1个训练样本")
    print(f"ID: {data['train']['ids'][0]}")
    print(f"序列: {data['train']['sequences'][0][:50]}...")
    print(f"序列长度: {len(data['train']['sequences'][0])}")
    print(f"标签数: {data['train']['labels'][0].sum():.0f}")

    # 找到有标签的GO术语
    label_indices = np.where(data['train']['labels'][0] == 1)[0]
    if len(label_indices) > 0:
        print(f"GO术语示例: {[data['go_terms'][i] for i in label_indices[:3]]}")

    print("\n" + "="*70)
    print("完成！数据已准备好供训练使用。")
    print("="*70)
