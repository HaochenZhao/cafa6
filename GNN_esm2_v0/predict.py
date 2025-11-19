"""
CAFA6推理脚本 - Pairwise模型
从cafa主目录运行: python GNN_esm2/predict.py
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

import numpy as np
import torch
from tqdm import tqdm

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import load_go_embeddings, load_processed_data
from model import PairwiseScoringModel


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """加载模型"""
    print(f"\n加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✓ 模型加载完成")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        m = checkpoint['metrics']
        print(f"  验证Fmax: {m['fmax']:.4f}")
        print(f"  验证AUPR: {m['aupr']:.4f}")
    
    return model


@torch.no_grad()
def predict_for_proteins(
    model: torch.nn.Module,
    protein_ids: list,
    seq_embeddings: np.ndarray,
    go_embeddings: np.ndarray,
    go_id_to_idx: dict,
    all_go_ids: list,
    device: torch.device,
    batch_size: int = 2048
) -> dict:
    """
    为蛋白质列表预测所有GO terms的分数
    
    返回:
        {protein_id: {go_id: score}}
    """
    model.eval()
    
    # 准备GO embeddings (所有GO terms)
    go_indices = [go_id_to_idx[go_id] for go_id in all_go_ids]
    go_embs_all = torch.FloatTensor(go_embeddings[go_indices]).to(device)
    num_gos = len(all_go_ids)
    
    predictions = {}
    
    print(f"\n预测 {len(protein_ids)} 个蛋白质 × {num_gos} 个GO terms...")
    
    for i, protein_id in enumerate(tqdm(protein_ids, desc="预测进度")):
        seq_emb = torch.FloatTensor(seq_embeddings[i]).to(device)
        
        protein_predictions = {}
        
        # 分批预测所有GO terms
        for go_start in range(0, num_gos, batch_size):
            go_end = min(go_start + batch_size, num_gos)
            batch_go_embs = go_embs_all[go_start:go_end]
            
            # 复制seq_emb到batch大小
            batch_seq_embs = seq_emb.unsqueeze(0).expand(len(batch_go_embs), -1)
            
            # 预测
            scores = model.predict(batch_seq_embs, batch_go_embs)
            scores = scores.squeeze().cpu().numpy()
            
            # 记录
            for j, score in enumerate(scores):
                go_id = all_go_ids[go_start + j]
                protein_predictions[go_id] = float(score)
        
        predictions[protein_id] = protein_predictions
    
    return predictions


def generate_submission(
    predictions: dict,
    output_file: str,
    threshold: float = 0.5,
    top_k: int = None
):
    """生成CAFA提交文件"""
    print(f"\n生成提交文件: {output_file}")
    print(f"  阈值: {threshold}")
    if top_k:
        print(f"  每个蛋白Top-{top_k}")
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    num_predictions = 0
    
    with open(output_file, 'w') as f:
        for protein_id, go_scores in tqdm(predictions.items(), desc="写入文件"):
            # 筛选高于阈值的
            filtered = [(go_id, score) for go_id, score in go_scores.items() 
                       if score >= threshold]
            
            if len(filtered) > 0:
                # 排序
                filtered.sort(key=lambda x: x[1], reverse=True)
                
                # Top-K
                if top_k:
                    filtered = filtered[:top_k]
                
                # 写入
                for go_id, score in filtered:
                    f.write(f"{protein_id}\t{go_id}\t{score:.6f}\n")
                    num_predictions += 1
    
    print(f"\n✓ 提交文件生成完成")
    print(f"  总预测数: {num_predictions:,}")
    print(f"  平均每蛋白: {num_predictions/len(predictions):.1f}")
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  文件大小: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--output', type=str, 
                        default='GNN_esm2/outputs/submission.tsv',
                        help='输出文件路径')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=2048)
    
    args = parser.parse_args()
    
    # 确保在cafa主目录下运行
    if not os.path.exists('cafa-6-protein-function-prediction'):
        print("错误: 请在cafa主目录下运行此脚本")
        print("用法: python GNN_esm2/predict.py [options]")
        sys.exit(1)
    
    print("="*70)
    print("CAFA6 推理 (Pairwise模型)")
    print("="*70)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载GO embeddings
    print("\n加载GO embeddings...")
    go_embeddings_path = config['data']['go_embeddings_dir']
    go_embeddings, go_metadata = load_go_embeddings(go_embeddings_path)
    
    go_id_to_idx = go_metadata['go_id_to_idx']
    all_go_ids = list(go_id_to_idx.keys())
    
    # 加载processed数据
    print("\n加载processed数据...")
    processed_dir = config['data']['processed_dir']
    data = load_processed_data(processed_dir)
    
    # 选择数据集
    if args.dataset == 'train':
        protein_ids = data['train_ids']
        seq_data_key = 'train_sequences'
    elif args.dataset == 'val':
        protein_ids = data['val_ids']
        seq_data_key = 'val_sequences'
    else:
        protein_ids = data['test_ids']
        seq_data_key = 'test_sequences'
    
    print(f"✓ 数据集: {args.dataset}, {len(protein_ids):,} 蛋白质")
    
    # 加载ESM2 embeddings
    emb_file = config['esm2']['embeddings_file']
    print(f"\n加载ESM2 embeddings: {emb_file}")
    emb_data = np.load(emb_file)
    
    if args.dataset == 'train':
        seq_embeddings = emb_data['train_embeddings']
    elif args.dataset == 'val':
        seq_embeddings = emb_data['val_embeddings']
    else:
        seq_embeddings = emb_data['test_embeddings']
    
    print(f"✓ ESM2 embeddings: {seq_embeddings.shape}")
    
    # 创建模型
    print("\n创建模型...")
    model = PairwiseScoringModel(config).to(device)
    
    # 加载权重
    model = load_checkpoint(args.checkpoint, model, device)
    
    # 预测
    predictions = predict_for_proteins(
        model=model,
        protein_ids=protein_ids,
        seq_embeddings=seq_embeddings,
        go_embeddings=go_embeddings,
        go_id_to_idx=go_id_to_idx,
        all_go_ids=all_go_ids,
        device=device,
        batch_size=args.batch_size
    )
    
    print(f"\n✓ 预测完成")
    print(f"  蛋白质数: {len(predictions)}")
    print(f"  GO terms数: {len(all_go_ids)}")
    
    # 统计预测分数分布
    all_scores = []
    for protein_predictions in predictions.values():
        all_scores.extend(protein_predictions.values())
    all_scores = np.array(all_scores)
    
    print(f"\n分数统计:")
    print(f"  范围: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"  平均: {all_scores.mean():.4f}")
    print(f"  中位数: {np.median(all_scores):.4f}")
    print(f"  >{args.threshold}: {(all_scores > args.threshold).sum() / len(all_scores) * 100:.2f}%")
    
    # 生成提交文件
    generate_submission(
        predictions,
        args.output,
        threshold=args.threshold,
        top_k=args.top_k
    )
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70)


if __name__ == "__main__":
    main()
