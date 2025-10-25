"""
V2版本快速开始
包含所有改进：过滤IA=0 + 智能层次化映射
"""

from save_processed_data_v2 import save_processed_data, load_processed_data
import numpy as np
import os

def main():
    print("="*70)
    print("CAFA6 数据处理器 V2 - 快速开始")
    print("="*70)
    print("\n改进点:")
    print("  ✓ 过滤IA=0的GO术语（45% → 0%）")
    print("  ✓ 智能层次化映射（只映射到最近祖先）")
    print("  ✓ 包含IA权重用于评估")
    print("  ✓ 详细的映射统计")

    # 处理数据
    if not os.path.exists("processed_data_v2/cafa6_data.npz"):
        print("\n" + "="*70)
        print("首次运行 - 处理数据")
        print("="*70)

        save_processed_data(
            output_dir="processed_data_v2",
            top_k_go=500,
            val_ratio=0.1,
            use_hierarchy=True
        )
    else:
        print("\n数据已存在，跳过处理。")

    # 读取数据
    print("\n" + "="*70)
    print("读取数据")
    print("="*70)

    data = load_processed_data("processed_data_v2")

    # 展示数据
    print("\n" + "="*70)
    print("数据概览")
    print("="*70)

    print(f"\n【样本数量】")
    print(f"  训练集: {len(data['train']['ids']):,}")
    print(f"  验证集: {len(data['val']['ids']):,}")
    print(f"  测试集: {len(data['test']['ids']):,}")

    print(f"\n【GO术语】")
    print(f"  数量: {len(data['go_terms'])}")
    print(f"  前5个: {data['go_terms'][:5]}")

    print(f"\n【IA权重】")
    print(f"  最小IA: {data['ia_weights'].min():.4f}")
    print(f"  最大IA: {data['ia_weights'].max():.2f}")
    print(f"  平均IA: {data['ia_weights'].mean():.2f}")
    print(f"  中位IA: {np.median(data['ia_weights']):.2f}")

    # Top-5 最高IA的GO
    top_ia_indices = np.argsort(data['ia_weights'])[-5:][::-1]
    print(f"\n  Top-5 最高IA的GO术语:")
    for i, idx in enumerate(top_ia_indices, 1):
        print(f"    {i}. {data['go_terms'][idx]}: IA={data['ia_weights'][idx]:.2f}")

    # 标签统计
    print(f"\n【标签统计】")
    train_label_counts = data['train']['labels'].sum(axis=1)
    val_label_counts = data['val']['labels'].sum(axis=1)

    print(f"  训练集:")
    print(f"    - 平均标签数: {train_label_counts.mean():.2f}")
    print(f"    - 最多标签数: {int(train_label_counts.max())}")
    print(f"    - 有标签样本: {(train_label_counts > 0).sum():,} ({(train_label_counts > 0).sum()/len(train_label_counts)*100:.2f}%)")
    print(f"    - 无标签样本: {(train_label_counts == 0).sum():,} ({(train_label_counts == 0).sum()/len(train_label_counts)*100:.2f}%)")

    print(f"\n  验证集:")
    print(f"    - 平均标签数: {val_label_counts.mean():.2f}")
    print(f"    - 有标签样本: {(val_label_counts > 0).sum():,} ({(val_label_counts > 0).sum()/len(val_label_counts)*100:.2f}%)")

    # 映射统计（如果有）
    if 'metadata' in data and 'stats' in data['metadata']:
        stats = data['metadata']['stats']['mapping_stats']
        total = stats['direct'] + stats['mapped'] + stats['lost']

        print(f"\n【GO映射统计】")
        print(f"  总标注数: {total:,}")
        print(f"  直接匹配Top-K: {stats['direct']:,} ({stats['direct']/total*100:.1f}%)")
        print(f"  层次化映射: {stats['mapped']:,} ({stats['mapped']/total*100:.1f}%)")
        print(f"  丢失: {stats['lost']:,} ({stats['lost']/total*100:.1f}%)")

    # 示例数据
    print(f"\n" + "="*70)
    print("示例数据")
    print("="*70)

    idx = 0
    print(f"\n第1个训练样本:")
    print(f"  ID: {data['train']['ids'][idx]}")
    print(f"  序列: {data['train']['sequences'][idx][:60]}...")
    print(f"  序列长度: {len(data['train']['sequences'][idx])}")
    print(f"  标签数: {int(data['train']['labels'][idx].sum())}")

    label_indices = np.where(data['train']['labels'][idx] == 1)[0]
    if len(label_indices) > 0:
        print(f"\n  具有的GO术语及其IA权重:")
        for i in label_indices[:5]:
            print(f"    - {data['go_terms'][i]}: IA={data['ia_weights'][i]:.2f}")
        if len(label_indices) > 5:
            print(f"    ... 还有 {len(label_indices)-5} 个")

    # 使用建议
    print(f"\n" + "="*70)
    print("使用建议")
    print("="*70)

    print(f"""
1. 访问数据:
   train_seqs = data['train']['sequences']
   train_labels = data['train']['labels']
   ia_weights = data['ia_weights']

2. 训练时:
   - 可以使用IA权重作为样本权重
   - 或者在loss中使用IA权重

3. 评估时:
   - 必须使用IA加权的评估指标
   - 参考 v2改进说明.md 中的评估代码

4. 提交时:
   - 使用IA>0的GO术语更符合CAFA标准
   - 预测结果会更准确

查看详细说明: v2改进说明.md
    """)

    print("="*70)
    print("准备完成！可以开始训练模型了。")
    print("="*70)


if __name__ == "__main__":
    main()
