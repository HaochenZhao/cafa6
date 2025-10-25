"""
最简单的使用示例
只需要两步：
  1. 首次运行：处理并保存数据
  2. 后续运行：快速读取数据
"""

from save_processed_data import save_processed_data, load_processed_data
import os
import numpy as np

def main():
    # ========================================
    # 步骤1：处理并保存数据（首次运行）
    # ========================================
    if not os.path.exists("processed_data/cafa6_data.npz"):
        print("检测到首次运行，正在处理数据...\n")

        save_processed_data(
            output_dir="processed_data",
            top_k_go=500,           # Top-500 GO术语
            val_ratio=0.1,          # 10%验证集
            use_hierarchy=True      # 使用层次化（推荐）
        )

        print("\n" + "="*70)
        print("数据处理完成！已保存到 processed_data/ 目录")
        print("下次运行将直接读取，无需重新处理。")
        print("="*70 + "\n")

    else:
        print("数据文件已存在，跳过处理步骤。\n")


    # ========================================
    # 步骤2：快速读取数据
    # ========================================
    print("\n" + "="*70)
    print("快速读取数据")
    print("="*70 + "\n")

    data = load_processed_data("processed_data")


    # ========================================
    # 步骤3：使用数据（示例）
    # ========================================
    print("\n" + "="*70)
    print("数据使用示例")
    print("="*70)

    # 训练数据
    print(f"\n【训练集】")
    print(f"样本数: {len(data['train']['ids'])}")
    print(f"特征: 序列（字符串数组）")
    print(f"标签: {data['train']['labels'].shape} (样本数 × GO术语数)")

    # 验证数据
    print(f"\n【验证集】")
    print(f"样本数: {len(data['val']['ids'])}")
    print(f"标签: {data['val']['labels'].shape}")

    # 测试数据
    print(f"\n【测试集】")
    print(f"样本数: {len(data['test']['ids'])}")

    # GO术语
    print(f"\n【GO术语】")
    print(f"总数: {len(data['go_terms'])}")
    print(f"前5个: {data['go_terms'][:5]}")

    # 查看一个样本
    print(f"\n【示例：第1个训练样本】")
    idx = 0
    print(f"ID: {data['train']['ids'][idx]}")
    print(f"序列: {data['train']['sequences'][idx][:60]}...")
    print(f"序列长度: {len(data['train']['sequences'][idx])}")
    print(f"标签向量: {data['train']['labels'][idx][:10]}... (前10维)")
    print(f"标签总数: {int(data['train']['labels'][idx].sum())}")

    # 找出该蛋白质的GO术语
    label_indices = np.where(data['train']['labels'][idx] == 1)[0]
    if len(label_indices) > 0:
        print(f"具有的GO术语:")
        for i in label_indices[:5]:
            print(f"  - {data['go_terms'][i]}")
        if len(label_indices) > 5:
            print(f"  ... 还有 {len(label_indices)-5} 个")


    # ========================================
    # 步骤4：基本统计
    # ========================================
    print(f"\n" + "="*70)
    print("数据统计")
    print("="*70)

    # 标签分布
    train_label_counts = data['train']['labels'].sum(axis=1)
    val_label_counts = data['val']['labels'].sum(axis=1)

    print(f"\n【标签分布】")
    print(f"训练集:")
    print(f"  - 平均标签数: {train_label_counts.mean():.2f}")
    print(f"  - 最多标签数: {int(train_label_counts.max())}")
    print(f"  - 有标签样本: {(train_label_counts > 0).sum()} ({(train_label_counts > 0).sum()/len(train_label_counts)*100:.2f}%)")
    print(f"  - 无标签样本: {(train_label_counts == 0).sum()} ({(train_label_counts == 0).sum()/len(train_label_counts)*100:.2f}%)")

    print(f"\n验证集:")
    print(f"  - 平均标签数: {val_label_counts.mean():.2f}")
    print(f"  - 有标签样本: {(val_label_counts > 0).sum()} ({(val_label_counts > 0).sum()/len(val_label_counts)*100:.2f}%)")

    # 序列长度分布
    print(f"\n【序列长度分布】")
    all_seq_lens = [len(s) for s in data['train']['sequences']]
    print(f"  - 最短: {min(all_seq_lens)}")
    print(f"  - 最长: {max(all_seq_lens)}")
    print(f"  - 平均: {np.mean(all_seq_lens):.1f}")
    print(f"  - 中位数: {np.median(all_seq_lens):.1f}")

    # GO术语频率
    print(f"\n【GO术语频率（训练集）】")
    go_freq = data['train']['labels'].sum(axis=0)
    print(f"  - 最常见GO: {go_freq.max():.0f} 个蛋白质")
    print(f"  - 最少见GO: {go_freq.min():.0f} 个蛋白质")
    print(f"  - 平均每个GO: {go_freq.mean():.0f} 个蛋白质")

    top_5_indices = np.argsort(go_freq)[-5:][::-1]
    print(f"\n  Top-5 最常见GO术语:")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"    {i}. {data['go_terms'][idx]}: {int(go_freq[idx])} 个蛋白质")


    # ========================================
    # 完成
    # ========================================
    print(f"\n" + "="*70)
    print("数据已准备就绪！")
    print("="*70)
    print(f"\n下一步：")
    print(f"1. 提取序列特征（如氨基酸组成、嵌入向量等）")
    print(f"2. 构建模型（如MLP、CNN、Transformer等）")
    print(f"3. 训练和评估")
    print(f"\n提示：查看 快速使用指南.md 获取更多示例代码")


if __name__ == "__main__":
    main()
