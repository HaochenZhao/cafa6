"""
预计算ESM2 embeddings
将所有蛋白质序列的ESM2嵌入提前计算好并保存，加速训练
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from typing import List, Tuple
import h5py


def load_esm2_model(model_name: str = "esm2_t6_8M_UR50D", device: torch.device = None):
    """
    加载ESM2模型

    参数:
        model_name: ESM2模型名称
            - esm2_t6_8M_UR50D (8M参数，推荐用于快速实验)
            - esm2_t12_35M_UR50D (35M参数)
            - esm2_t30_150M_UR50D (150M参数)
            - esm2_t33_650M_UR50D (650M参数，需要大GPU)
        device: 计算设备

    返回:
        model, alphabet, batch_converter
    """
    try:
        import esm
    except ImportError:
        raise ImportError("请安装 fair-esm: pip install fair-esm")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n加载ESM2模型: {model_name}")
    print(f"使用设备: {device}")

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()

    model = model.to(device)
    model.eval()

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    print(f"✓ 模型加载完成")
    print(f"  嵌入维度: {model.embed_dim}")
    print(f"  层数: {model.num_layers}")

    return model, alphabet, batch_converter


def compute_embeddings_batch(
    sequences: List[str],
    model,
    batch_converter,
    device: torch.device,
    pooling: str = 'mean'
) -> np.ndarray:
    """
    批量计算ESM2嵌入

    参数:
        sequences: 蛋白质序列列表
        model: ESM2模型
        batch_converter: ESM2的batch converter
        device: 计算设备
        pooling: 池化方式 ('mean' 或 'cls')

    返回:
        embeddings: [batch_size, embed_dim]
    """
    # 准备批次数据
    batch_labels = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_labels)
    batch_tokens = batch_tokens.to(device)

    # 前向传播
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers])

    # 提取表示
    token_representations = results["representations"][model.num_layers]

    # 池化
    embeddings = []
    if pooling == 'mean':
        # 平均池化（不包括特殊token）
        for i, seq_len in enumerate([len(seq) for seq in batch_strs]):
            # 不包括<cls>和<eos>
            seq_embedding = token_representations[i, 1:seq_len+1].mean(0)
            embeddings.append(seq_embedding.cpu().numpy())
    elif pooling == 'cls':
        # 使用<cls> token
        for i in range(len(batch_strs)):
            embeddings.append(token_representations[i, 0].cpu().numpy())
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    return np.array(embeddings)


def compute_embeddings_for_dataset(
    sequences: np.ndarray,
    protein_ids: np.ndarray,
    model,
    batch_converter,
    device: torch.device,
    batch_size: int = 16,
    pooling: str = 'mean',
    desc: str = "Computing embeddings",
    max_seq_len: int = 1024  # 最大序列长度
) -> np.ndarray:
    """
    为整个数据集计算嵌入（智能批次处理）

    参数:
        sequences: 蛋白质序列数组
        protein_ids: 蛋白质ID
        model: ESM2模型
        batch_converter: batch converter
        device: 计算设备
        batch_size: 批次大小
        pooling: 池化方式
        desc: 进度条描述
        max_seq_len: 最大序列长度，超过此长度的序列单独处理

    返回:
        embeddings: [num_samples, embed_dim]
    """
    num_samples = len(sequences)
    all_embeddings = [None] * num_samples  # 预分配列表保持顺序

    # 按序列长度分组
    seq_lengths = np.array([len(seq) for seq in sequences])
    long_seq_mask = seq_lengths > max_seq_len

    # 分别处理短序列和长序列
    with tqdm(total=num_samples, desc=desc) as pbar:
        # 1. 批量处理短序列
        short_indices = np.where(~long_seq_mask)[0]
        if len(short_indices) > 0:
            for i in range(0, len(short_indices), batch_size):
                batch_indices = short_indices[i:i+batch_size]
                batch_seqs = [sequences[idx] for idx in batch_indices]

                try:
                    # 计算嵌入
                    batch_embeddings = compute_embeddings_batch(
                        batch_seqs,
                        model,
                        batch_converter,
                        device,
                        pooling
                    )

                    # 保存到对应位置
                    for j, idx in enumerate(batch_indices):
                        all_embeddings[idx] = batch_embeddings[j]

                    pbar.update(len(batch_indices))

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # OOM错误，逐个处理这个batch
                        print(f"\n警告: Batch处理失败，切换到逐个处理模式")
                        torch.cuda.empty_cache()

                        for idx in batch_indices:
                            seq = [sequences[idx]]
                            emb = compute_embeddings_batch(
                                seq, model, batch_converter, device, pooling
                            )
                            all_embeddings[idx] = emb[0]
                            pbar.update(1)
                    else:
                        raise e

        # 2. 逐个处理长序列（避免OOM）
        long_indices = np.where(long_seq_mask)[0]
        if len(long_indices) > 0:
            print(f"\n处理 {len(long_indices)} 个长序列 (长度>{max_seq_len})，使用单样本模式...")
            for idx in long_indices:
                seq = [sequences[idx]]

                try:
                    emb = compute_embeddings_batch(
                        seq, model, batch_converter, device, pooling
                    )
                    all_embeddings[idx] = emb[0]
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # 序列太长，截断处理
                        print(f"\n警告: 序列 {protein_ids[idx]} 太长 ({len(sequences[idx])}), 截断到 {max_seq_len}")
                        torch.cuda.empty_cache()

                        truncated_seq = [sequences[idx][:max_seq_len]]
                        emb = compute_embeddings_batch(
                            truncated_seq, model, batch_converter, device, pooling
                        )
                        all_embeddings[idx] = emb[0]
                    else:
                        raise e

                pbar.update(1)

    # 合并所有嵌入
    embeddings = np.array(all_embeddings)

    return embeddings


def save_embeddings_npz(
    embeddings_dict: dict,
    output_file: str
):
    """
    保存嵌入到npz文件

    参数:
        embeddings_dict: {'train': embeddings, 'val': embeddings, 'test': embeddings}
        output_file: 输出文件路径
    """
    print(f"\n保存嵌入到: {output_file}")

    np.savez_compressed(
        output_file,
        **embeddings_dict
    )

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ 保存完成")
    print(f"  文件大小: {file_size_mb:.2f} MB")


def save_embeddings_h5(
    embeddings_dict: dict,
    output_file: str
):
    """
    保存嵌入到HDF5文件（更高效的存储格式）

    参数:
        embeddings_dict: {'train': embeddings, 'val': embeddings, 'test': embeddings}
        output_file: 输出文件路径
    """
    print(f"\n保存嵌入到: {output_file}")

    with h5py.File(output_file, 'w') as f:
        for key, embeddings in embeddings_dict.items():
            f.create_dataset(
                key,
                data=embeddings,
                compression='gzip',
                compression_opts=9
            )

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ 保存完成")
    print(f"  文件大小: {file_size_mb:.2f} MB")


def load_embeddings_npz(embeddings_file: str) -> dict:
    """加载npz格式的嵌入"""
    print(f"\n加载嵌入: {embeddings_file}")
    data = np.load(embeddings_file)
    embeddings = {key: data[key] for key in data.files}
    print(f"✓ 加载完成")
    for key, emb in embeddings.items():
        print(f"  {key}: {emb.shape}")
    return embeddings


def load_embeddings_h5(embeddings_file: str) -> dict:
    """加载HDF5格式的嵌入"""
    print(f"\n加载嵌入: {embeddings_file}")
    embeddings = {}
    with h5py.File(embeddings_file, 'r') as f:
        for key in f.keys():
            embeddings[key] = f[key][:]
    print(f"✓ 加载完成")
    for key, emb in embeddings.items():
        print(f"  {key}: {emb.shape}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="预计算ESM2 embeddings")
    parser.add_argument('--data_dir', type=str, default='processed_data_v3',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default='embeddings',
                        help='输出目录')
    parser.add_argument('--model_name', type=str, default='esm2_t6_8M_UR50D',
                        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D',
                                'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D', 
                                'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D'],
                        help='ESM2模型名称')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'cls'],
                        help='池化方式')
    parser.add_argument('--format', type=str, default='npz',
                        choices=['npz', 'h5'],
                        help='保存格式 (npz或h5)')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (cuda或cpu，默认自动选择)')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='最大序列长度，超过此长度的序列单独处理')

    args = parser.parse_args()

    print("="*70)
    print("ESM2 Embeddings 预计算")
    print("="*70)
    print(f"\n配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  模型: {args.model_name}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  池化方式: {args.pooling}")
    print(f"  保存格式: {args.format}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # 加载数据
    print(f"\n{'='*70}")
    print("加载数据")
    print('='*70)

    npz_file = os.path.join(args.data_dir, "cafa6_data.npz")
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"数据文件不存在: {npz_file}")

    data = np.load(npz_file, allow_pickle=True)

    train_sequences = data['train_sequences']
    val_sequences = data['val_sequences']
    test_sequences = data['test_sequences']
    train_ids = data['train_ids']
    val_ids = data['val_ids']
    test_ids = data['test_ids']

    print(f"✓ 数据加载完成")
    print(f"  训练集: {len(train_sequences):,} 样本")
    print(f"  验证集: {len(val_sequences):,} 样本")
    print(f"  测试集: {len(test_sequences):,} 样本")

    # 加载ESM2模型
    print(f"\n{'='*70}")
    print("加载ESM2模型")
    print('='*70)

    model, alphabet, batch_converter = load_esm2_model(args.model_name, device)

    # 计算嵌入
    print(f"\n{'='*70}")
    print("计算嵌入")
    print('='*70)

    embeddings = {}

    # 训练集
    print("\n1. 训练集")
    embeddings['train_embeddings'] = compute_embeddings_for_dataset(
        train_sequences,
        train_ids,
        model,
        batch_converter,
        device,
        args.batch_size,
        args.pooling,
        desc="训练集",
        max_seq_len=args.max_seq_len
    )
    print(f"  形状: {embeddings['train_embeddings'].shape}")

    # 验证集
    print("\n2. 验证集")
    embeddings['val_embeddings'] = compute_embeddings_for_dataset(
        val_sequences,
        val_ids,
        model,
        batch_converter,
        device,
        args.batch_size,
        args.pooling,
        desc="验证集",
        max_seq_len=args.max_seq_len
    )
    print(f"  形状: {embeddings['val_embeddings'].shape}")

    # 测试集
    print("\n3. 测试集")
    embeddings['test_embeddings'] = compute_embeddings_for_dataset(
        test_sequences,
        test_ids,
        model,
        batch_converter,
        device,
        args.batch_size,
        args.pooling,
        desc="测试集",
        max_seq_len=args.max_seq_len
    )
    print(f"  形状: {embeddings['test_embeddings'].shape}")

    # 保存嵌入
    print(f"\n{'='*70}")
    print("保存嵌入")
    print('='*70)

    # 生成文件名
    model_short_name = args.model_name.replace('esm2_', '').replace('_UR50D', '')
    output_filename = f"embeddings_{model_short_name}_{args.pooling}.{args.format}"
    output_path = os.path.join(args.output_dir, output_filename)

    if args.format == 'npz':
        save_embeddings_npz(embeddings, output_path)
    else:  # h5
        save_embeddings_h5(embeddings, output_path)

    # 保存元数据
    import json
    metadata = {
        'model_name': args.model_name,
        'pooling': args.pooling,
        'batch_size': args.batch_size,
        'embed_dim': embeddings['train_embeddings'].shape[1],
        'num_train': len(train_sequences),
        'num_val': len(val_sequences),
        'num_test': len(test_sequences)
    }

    metadata_path = os.path.join(args.output_dir, f"metadata_{model_short_name}_{args.pooling}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ 保存元数据: {metadata_path}")

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n嵌入文件: {output_path}")
    print(f"元数据文件: {metadata_path}")
    print(f"\n使用方法:")
    print(f"  python train.py --use_precomputed_embeddings --embeddings_file {output_path}")


if __name__ == "__main__":
    main()
