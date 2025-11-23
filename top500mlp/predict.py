"""
预测脚本：使用训练好的模型生成CAFA6提交文件
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model_three_head_mlp import ThreeHeadMLP, ESM2WithThreeHeadMLP
from dataset import CAFA6Dataset, collate_fn_precomputed, collate_fn_dynamic
from torch.utils.data import DataLoader


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """加载模型权重"""
    print(f"\n加载模型: {checkpoint_path}")
    # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含numpy对象的checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型加载完成")

    # 打印训练信息
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if 'overall' in metrics:
            print(f"  验证集Fmax: {metrics['overall']['fmax']:.4f}")
            print(f"  验证集AUPR: {metrics['overall']['aupr']:.4f}")

    return model


@torch.no_grad()
def predict_on_dataset(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_precomputed: bool = False
) -> tuple:
    """
    在数据集上进行预测

    返回:
        predictions: [num_samples, num_labels]
        protein_ids: list of protein IDs
    """
    model.eval()
    all_predictions = []
    all_protein_ids = []

    print("\n开始预测...")
    for batch in tqdm(data_loader, desc="预测进度"):
        protein_ids = batch['protein_ids']

        # 前向传播
        if use_precomputed:
            embeddings = batch['embeddings'].to(device)
            outputs = model(embeddings, return_separate=True)
        else:
            sequences = batch['sequences']
            outputs = model(sequences, return_separate=True)

        # 拼接三个头的输出
        logits = torch.cat([outputs['C'], outputs['F'], outputs['P']], dim=1)
        predictions = torch.sigmoid(logits).cpu().numpy()

        all_predictions.append(predictions)
        all_protein_ids.extend(protein_ids)

    # 合并所有预测
    predictions = np.concatenate(all_predictions, axis=0)

    return predictions, all_protein_ids


def generate_submission(
    predictions: np.ndarray,
    protein_ids: List[str],
    go_terms: List[str],
    output_file: str,
    threshold: float = 0.01,
    top_k: int = None
):
    """
    生成CAFA6提交文件

    参数:
        predictions: [num_samples, num_labels] 预测概率
        protein_ids: 蛋白质ID列表
        go_terms: GO术语列表
        output_file: 输出文件路径
        threshold: 最小置信度阈值（低于此值的不输出）
        top_k: 每个蛋白质最多输出多少个GO术语
    """
    print(f"\n生成提交文件: {output_file}")
    print(f"  阈值: {threshold}")
    if top_k:
        print(f"  每个蛋白质最多Top-{top_k}个预测")

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建目录: {output_dir}")

    num_predictions = 0

    with open(output_file, 'w') as f:
        for i, protein_id in enumerate(tqdm(protein_ids, desc="写入文件")):
            protein_preds = predictions[i]

            # 筛选高于阈值的预测
            valid_indices = np.where(protein_preds >= threshold)[0]

            if len(valid_indices) > 0:
                # 按置信度排序
                sorted_indices = valid_indices[np.argsort(-protein_preds[valid_indices])]

                # 如果指定了top_k，只取前k个
                if top_k is not None:
                    sorted_indices = sorted_indices[:top_k]

                # 写入预测
                for idx in sorted_indices:
                    go_term = go_terms[idx]
                    confidence = protein_preds[idx]
                    f.write(f"{protein_id}\t{go_term}\t{confidence:.6f}\n")
                    num_predictions += 1

    print(f"\n✓ 提交文件生成完成")
    print(f"  总预测数: {num_predictions:,}")
    print(f"  平均每个蛋白质: {num_predictions/len(protein_ids):.1f} 个GO术语")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="生成CAFA6提交文件")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径 (best.pth)')
    parser.add_argument('--data_dir', type=str, default='processed_data_v3',
                        help='数据目录')
    parser.add_argument('--output', type=str, default='submission.tsv',
                        help='输出文件路径')
    parser.add_argument('--use_precomputed_embeddings', action='store_true',
                        help='使用预计算的嵌入')
    parser.add_argument('--embeddings_file', type=str, default=None,
                        help='预计算嵌入文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='最小置信度阈值')
    parser.add_argument('--top_k', type=int, default=None,
                        help='每个蛋白质最多输出的GO术语数（不限制则设为None）')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='预测哪个数据集')

    args = parser.parse_args()

    print("="*70)
    print("CAFA6 提交文件生成")
    print("="*70)
    print(f"\n配置:")
    print(f"  模型: {args.checkpoint}")
    print(f"  数据集: {args.dataset}")
    print(f"  输出: {args.output}")
    print(f"  阈值: {args.threshold}")
    print(f"  Top-K: {args.top_k if args.top_k else '不限制'}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n" + "="*70)
    print("加载数据")
    print("="*70)

    npz_file = os.path.join(args.data_dir, "cafa6_data.npz")
    data = np.load(npz_file, allow_pickle=True)

    # 根据数据集选择
    if args.dataset == 'train':
        sequences = data['train_sequences']
        protein_ids = data['train_ids']
        labels = data['train_labels']
    elif args.dataset == 'val':
        sequences = data['val_sequences']
        protein_ids = data['val_ids']
        labels = data['val_labels']
    else:  # test
        sequences = data['test_sequences']
        protein_ids = data['test_ids']
        labels = None  # 测试集没有标签

    go_terms = data['go_terms'].tolist()
    go_aspects = data['go_aspects']

    print(f"✓ 数据加载完成")
    print(f"  样本数: {len(sequences):,}")
    print(f"  GO术语: {len(go_terms)}")

    # 加载预计算嵌入（如果使用）
    embeddings = None
    if args.use_precomputed_embeddings:
        if args.embeddings_file is None:
            raise ValueError("使用预计算嵌入时必须指定 --embeddings_file")

        print(f"\n加载预计算嵌入: {args.embeddings_file}")

        if args.embeddings_file.endswith('.npz'):
            emb_data = np.load(args.embeddings_file)
            if args.dataset == 'train':
                embeddings = emb_data['train_embeddings']
            elif args.dataset == 'val':
                embeddings = emb_data['val_embeddings']
            else:
                embeddings = emb_data['test_embeddings']
        elif args.embeddings_file.endswith('.h5'):
            import h5py
            with h5py.File(args.embeddings_file, 'r') as f:
                key = f'{args.dataset}_embeddings'
                embeddings = f[key][:]

        print(f"✓ 嵌入加载完成: {embeddings.shape}")

    # 创建数据集
    dataset = CAFA6Dataset(
        sequences=sequences,
        labels=labels if labels is not None else np.zeros((len(sequences), len(go_terms))),
        protein_ids=protein_ids,
        go_aspects=go_aspects,
        use_precomputed_embeddings=args.use_precomputed_embeddings,
        embeddings=embeddings
    )

    # 创建数据加载器
    collate_fn = collate_fn_precomputed if args.use_precomputed_embeddings else collate_fn_dynamic

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 创建模型
    print("\n" + "="*70)
    print("创建模型")
    print("="*70)

    num_labels_per_aspect = {
        'C': (go_aspects == 'C').sum(),
        'F': (go_aspects == 'F').sum(),
        'P': (go_aspects == 'P').sum()
    }

    if args.use_precomputed_embeddings:
        esm_dim = embeddings.shape[1]
        model = ThreeHeadMLP(
            esm_dim=esm_dim,
            hidden_dims=[512, 256],
            num_labels_per_aspect=num_labels_per_aspect,
            dropout=0.3,
            use_shared_layer=True
        )
    else:
        model = ESM2WithThreeHeadMLP(
            esm_model_name="esm2_t6_8M_UR50D",
            num_labels_per_aspect=num_labels_per_aspect,
            hidden_dims=[512, 256],
            dropout=0.3,
            use_shared_layer=True,
            freeze_esm=True,
            pooling='mean'
        )

    # 加载权重
    model = load_checkpoint(args.checkpoint, model, device)
    model = model.to(device)

    # 预测
    print("\n" + "="*70)
    print("预测")
    print("="*70)

    predictions, pred_protein_ids = predict_on_dataset(
        model,
        data_loader,
        device,
        use_precomputed=args.use_precomputed_embeddings
    )

    print(f"\n✓ 预测完成")
    print(f"  预测样本数: {len(pred_protein_ids):,}")
    print(f"  预测形状: {predictions.shape}")
    print(f"  预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  平均置信度: {predictions.mean():.4f}")

    # 生成提交文件
    print("\n" + "="*70)
    print("生成提交文件")
    print("="*70)

    generate_submission(
        predictions,
        pred_protein_ids,
        go_terms,
        args.output,
        threshold=args.threshold,
        top_k=args.top_k
    )

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n提交文件: {args.output}")
    print(f"\n可以直接提交到CAFA6评估系统")


if __name__ == "__main__":
    main()
