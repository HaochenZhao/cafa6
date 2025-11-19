"""
测试v3代码的无标签样本过滤功能
"""
import os

# 删除旧的数据文件以重新生成
if os.path.exists("processed_data_v3/cafa6_data.npz"):
    print("删除旧数据文件...")
    os.remove("processed_data_v3/cafa6_data.npz")
    os.remove("processed_data_v3/cafa6_metadata.pkl")
    os.remove("processed_data_v3/data_statistics.txt")

# 导入并运行数据处理
from save_processed_data_v3 import save_processed_data, load_processed_data

print("="*70)
print("测试 v3 数据处理 (移除无标签样本)")
print("="*70)

# 运行数据处理
metadata = save_processed_data(
    output_dir="processed_data_v3",
    top_k_per_aspect=500,
    val_ratio=0.1,
    use_hierarchy=True,
    remove_empty_labels=True  # 开启无标签样本过滤
)

print("\n" + "="*70)
print("过滤统计:")
print("="*70)
filter_stats = metadata['filter_stats']
print(f"训练集:")
print(f"  - 过滤前: {filter_stats['train_before']:,}")
print(f"  - 过滤后: {filter_stats['train_after']:,}")
print(f"  - 移除: {filter_stats['train_removed']:,} ({filter_stats['train_removed']/filter_stats['train_before']*100:.2f}%)")

print(f"\n验证集:")
print(f"  - 过滤前: {filter_stats['val_before']:,}")
print(f"  - 过滤后: {filter_stats['val_after']:,}")
print(f"  - 移除: {filter_stats['val_removed']:,} ({filter_stats['val_removed']/filter_stats['val_before']*100:.2f}%)")

print("\n测试完成！")
