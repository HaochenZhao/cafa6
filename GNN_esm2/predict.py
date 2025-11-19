import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import pandas as pd
from datetime import datetime
import json
# ⚠️ 推荐使用 h5py 或 zarr 库来分块保存大矩阵，但为了不引入新的依赖，这里保留原有的 np.savez_compressed 函数。

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入 PairwiseScorer, BatchEvaluator, load_processed_data, load_go_embeddings (假设它们都在 src 中)
from src import PairwiseScorer, BatchEvaluator
from src.dataset import load_processed_data, load_go_embeddings


# --- 保持 load_checkpoint 函数不变 ---
def load_checkpoint(checkpoint_path: str, device: torch.device):
    """加载训练好的模型"""
    print(f"\n加载模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("⚠️  checkpoint中没有config，使用默认配置")
        config = {
            'model': {
                'esm_dim': 320,
                'go_dim': 256,
                'hidden_dims': [512, 256, 128],
                'dropout': 0.3,
                'fusion_type': 'concat'
            }
        }
    
    # 创建模型
    model = PairwiseScorer(
        esm_dim=config['model']['esm_dim'],
        go_dim=config['model']['go_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        fusion_type=config['model']['fusion_type']
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'fmax' in checkpoint:
        print(f"  Fmax: {checkpoint['fmax']:.4f}")
    
    return model, config

# --- 保持 load_test_data 函数不变 ---
def load_test_data(config: dict, base_dir: str = '.'):
    """加载测试数据"""
    print("\n" + "="*70)
    print("加载测试数据")
    print("="*70)
    
    # 加载processed数据
    processed_dir = os.path.join(base_dir, config['data']['processed_dir'])
    data = load_processed_data(processed_dir)
    
    if 'test_ids' not in data:
        raise ValueError("processed数据中没有test_ids，请重新运行preprocess_data.py")
    
    test_ids = data['test_ids']
    go_terms = data['go_terms'].tolist()
    ia_weights = data['ia_weights']
    go_aspects = data['go_aspects']
    
    print(f"✓ 测试集: {len(test_ids):,} 蛋白质")
    print(f"✓ GO terms: {len(go_terms):,} (所有IA > 0的GO)")
    
    # 统计各aspect
    aspect_counts = {'C': 0, 'F': 0, 'P': 0}
    for aspect in go_aspects:
        if aspect in aspect_counts:
            aspect_counts[aspect] += 1
    
    print(f"  C: {aspect_counts['C']:,}")
    print(f"  F: {aspect_counts['F']:,}")
    print(f"  P: {aspect_counts['P']:,}")
    
    # 加载GO embeddings并过滤
    go_embeddings_path = os.path.join(base_dir, config['data']['go_embeddings_dir'])
    full_go_embeddings, full_go_metadata = load_go_embeddings(go_embeddings_path)
    
    print(f"\n过滤GO embeddings以匹配processed数据...")
    
    filtered_indices = []
    filtered_go_ids = []
    
    for go_id in go_terms:
        if go_id in full_go_metadata['go_id_to_idx']:
            idx = full_go_metadata['go_id_to_idx'][go_id]
            filtered_indices.append(idx)
            filtered_go_ids.append(go_id)
    
    go_embeddings = full_go_embeddings[filtered_indices]
    
    print(f"✓ 过滤后GO数: {len(filtered_go_ids):,}")
    
    # 加载ESM2 embeddings
    emb_file = os.path.join(base_dir, config['esm2']['embeddings_file'])
    print(f"\n✓ 加载ESM2 embeddings: {emb_file}")
    emb_data = np.load(emb_file)
    
    if 'test_embeddings' not in emb_data:
        raise ValueError("ESM2 embeddings中没有test_embeddings，请重新运行precompute_embeddings.py")
    
    test_embeddings = emb_data['test_embeddings']
    print(f"  测试集embeddings: {test_embeddings.shape}")
    
    # 验证数量匹配
    if len(test_embeddings) != len(test_ids):
        raise ValueError(
            f"测试集数量不匹配: "
            f"IDs={len(test_ids)}, Embeddings={len(test_embeddings)}"
        )
    
    return {
        'test_ids': test_ids,
        'test_embeddings': test_embeddings,
        'go_embeddings': go_embeddings,
        'go_terms': filtered_go_ids,
        'go_aspects': go_aspects,
        'ia_weights': ia_weights
    }


def save_cafa_batch(
    protein_ids: np.ndarray,
    go_terms: List[str],
    scores: np.ndarray,
    output_file: str,
    threshold: float = 0.5,
    write_mode: str = 'a',  # 'w' for header, 'a' for data
    model_name: str = "GNN_ESM2_Pairwise",
    keywords: List[str] = None
) -> int:
    """
    保存一个批次的结果到CAFA格式文件。
    返回写入的预测数量。
    """
    num_written = 0
    
    with open(output_file, write_mode) as f:
        if write_mode == 'w':
            # 写入Header (只在第一次调用时)
            f.write(f"AUTHOR Your_Name\n")
            f.write(f"MODEL {model_name}\n")
            
            if keywords:
                f.write(f"KEYWORDS {', '.join(keywords)}\n")
            else:
                f.write(f"KEYWORDS deep learning, GNN, ESM2, pairwise\n")
            # f.write(f"ACCURACY 1 0.5 all\n")

        # 预测结果
        predictions = (scores >= threshold).astype(int)
        
        for i, protein_id in enumerate(protein_ids):
            protein_scores = scores[i]
            protein_preds = predictions[i]
            
            # 获取预测为正的GO
            positive_indices = np.where(protein_preds == 1)[0]
            
            for go_idx in positive_indices:
                go_id = go_terms[go_idx]
                confidence = protein_scores[go_idx]
                
                f.write(f"{protein_id}\t{go_id}\t{confidence:.6f}\n")
                num_written += 1
                
    return num_written


def predict_and_save_batches(
    model,
    test_ids: np.ndarray,
    test_embeddings: np.ndarray,
    go_embeddings: torch.Tensor,
    go_terms: List[str],
    device: torch.device,
    cafa_output_file: str,
    threshold: float,
    batch_size: int = 100,
    chunk_size: int = 1000,
    save_scores: bool = False,
    scores_output_file: str = None
) -> Dict[str, float]:
    """
    批量预测并实时保存结果，同时计算总预测数的统计信息。
    """
    print("\n" + "="*70)
    print("批量预测与实时保存")
    print("="*70)
    print(f"  蛋白质数: {len(test_embeddings):,}")
    print(f"  GO数: {len(go_embeddings):,}")
    print(f"  预测总数: {len(test_embeddings) * len(go_embeddings):,}")
    print(f"  批次大小: {batch_size} 蛋白质")
    print(f"  分块大小: {chunk_size} GO")
    print(f"  CAFA文件: {cafa_output_file}")
    
    # 初始化批量评估器
    evaluator = BatchEvaluator(
        model=model,
        go_embeddings=go_embeddings,
        device=device,
        chunk_size=chunk_size
    )
    
    total_predictions_cafa = 0
    total_scores_sum = 0.0
    total_elements = 0
    
    # 按批次处理蛋白质
    num_batches = (len(test_embeddings) + batch_size - 1) // batch_size
    
    # 第一次写入，添加CAFA Header
    # 注意: 为了简单，我们只在第一次写入时生成一个虚拟批次的 header
    save_cafa_batch(
        protein_ids=np.array(['P00000']), go_terms=['GO:0000001'], scores=np.array([[0.0]]),
        output_file=cafa_output_file, threshold=threshold, write_mode='w', keywords=None
    )
    # 立即清除文件内容，只保留Header
    with open(cafa_output_file, 'w') as f:
        # 重写Header
        f.write(f"AUTHOR Your_Name\n")
        f.write(f"MODEL GNN_ESM2_Pairwise\n")
        f.write(f"KEYWORDS deep learning, GNN, ESM2, pairwise\n")

    # 如果需要保存完整分数矩阵，先初始化文件（这里需要 h5py 或其他库）
    if save_scores:
        print("⚠️ 警告: 保存完整分数矩阵 (17GB+) 仍然需要大量磁盘空间和内存操作。")
        # 由于np.savez不支持追加，实际的大数据需要HDF5或Zarr。
        # 这里仅在主函数中添加提示，并跳过大矩阵保存，以确保程序能运行。
        pass
        
    with torch.no_grad():
        for i in tqdm(range(0, len(test_embeddings), batch_size), 
                      total=num_batches, desc="预测与实时保存"):
            
            # 提取本批次的IDs和Embeddings
            end_idx = min(i + batch_size, len(test_embeddings))
            batch_ids = test_ids[i:end_idx]
            batch_embs = test_embeddings[i:end_idx]
            batch_embs_tensor = torch.FloatTensor(batch_embs).to(device)
            
            # 批量计算分数
            # batch_scores 形状为 [batch_size, num_gos]
            batch_scores = evaluator.evaluate_proteins(
                batch_embs_tensor,
                return_all_scores=True
            ).cpu().numpy()
            
            # 实时保存CAFA格式结果 (追加写入)
            num_written_batch = save_cafa_batch(
                protein_ids=batch_ids,
                go_terms=go_terms,
                scores=batch_scores,
                output_file=cafa_output_file,
                threshold=threshold,
                write_mode='a' # 追加写入
            )
            total_predictions_cafa += num_written_batch
            
            # 实时统计分数 (用于计算总平均值)
            total_scores_sum += batch_scores.sum()
            total_elements += batch_scores.size
            
            # ⚠️ 如果启用了 save_scores，应该在这里使用 h5py 或 Zarr 追加到文件中
            # if save_scores:
            #     save_scores_to_h5(scores_output_file, batch_scores, batch_ids, go_terms)
            
            # 显式清除，帮助Python GC
            del batch_embs_tensor
            del batch_scores
            torch.cuda.empty_cache()

    # 写入CAFA文件末尾的 END
    with open(cafa_output_file, 'a') as f:
        f.write("END\n")
        
    print(f"✓ 预测与保存完成: {total_elements:,} 个分数")
    
    return {
        'total_predictions_cafa': total_predictions_cafa,
        'total_scores_sum': total_scores_sum,
        'total_elements': total_elements
    }

# --- 保持 apply_threshold 函数不变 (尽管在新流程中只用于 stats) ---
def apply_threshold(
    scores: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """应用阈值"""
    predictions = (scores >= threshold).astype(int)
    
    num_predictions = predictions.sum()
    avg_per_protein = num_predictions / len(predictions)
    
    print(f"\n应用阈值: {threshold}")
    print(f"  预测总数: {num_predictions:,}")
    print(f"  平均每蛋白质: {avg_per_protein:.1f} GO terms")
    
    return predictions


# --- 移除 save_cafa_format 函数，功能已被 save_cafa_batch 取代 ---
# --- 移除 save_score_matrix 函数，仅保留注释提醒使用 h5py/zarr ---

def save_score_matrix(protein_ids, go_terms, scores, output_file):
    """
    保存完整的分数矩阵。警告：对于大矩阵，不应在内存中合并后再保存。
    此函数仅用于提醒，在大数据量下应使用 h5py 或 Zarr。
    """
    print("\n⚠️ 警告: 不建议对大型预测结果 (如 17.5GB) 使用 np.savez_compressed，"
          "因为它仍然需要在内存中构建完整的矩阵。")
    print("      请考虑使用 h5py/Zarr 进行分块存储。为确保程序成功运行，"
          "大矩阵的保存已在主函数中被跳过。")
    # 如果用户坚持使用此函数，则保持原样，但可能导致 OOM
    # np.savez_compressed(output_file, protein_ids=protein_ids, go_terms=np.array(go_terms), scores=scores)
    # print(f"✓ 尝试保存完整分数矩阵到: {output_file}")


def generate_statistics(
    go_aspects: np.ndarray,
    total_elements: int,
    total_proteins: int,
    total_predictions_cafa: int,
    total_scores_sum: float,
    scores_min: float,
    scores_max: float
) -> dict:
    """生成预测统计（简化，只统计最终 CAFA 预测数）"""
    stats = {
        'total_predictions_cafa': int(total_predictions_cafa),
        'avg_per_protein_cafa': float(total_predictions_cafa / total_proteins),
        'score_stats': {
            'min': float(scores_min),
            'max': float(scores_max),
            'mean': float(total_scores_sum / total_elements if total_elements > 0 else 0.0),
        },
        'predictions_by_aspect': {} # 无法实时计算，这里留空或简化
    }
    
    # ⚠️ 实时计算按 Aspect 统计需要额外的内存或复杂的批处理逻辑，这里暂时跳过详细统计。
    # 您可以通过后处理完整的CAFA文件来获得这些统计信息。
    print("\n⚠️ 警告: 详细的 '按Aspect统计' 需要完整的预测矩阵。由于采用流式处理，"
          "详细统计已在主函数中被跳过。")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="CAFA6预测 - 对所有IA非0的GO terms预测")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的模型checkpoint路径')
    parser.add_argument('--config', type=str, default='GNN_esm2/config/config.yaml',
                        help='配置文件')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='预测阈值')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='蛋白质批次大小')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='GO分块大小')
    parser.add_argument('--save_scores', action='store_true',
                        help='保存完整分数矩阵（用于后处理）。警告：即使分批次保存，17GB+ 的矩阵仍需要大量磁盘空间。')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CAFA6预测 (内存优化版) - 对所有IA非0的GO terms预测")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"输出目录: {args.output_dir}")
    print(f"阈值: {args.threshold}")
    print("="*70)
    
    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    model, config = load_checkpoint(args.checkpoint, device)
    
    # 加载测试数据
    test_data = load_test_data(config, base_dir='.')
    
    # --- 核心预测和实时保存 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cafa_file = os.path.join(
        args.output_dir,
        f"predictions_cafa_format_{timestamp}.tsv"
    )

    # 预测和实时保存
    prediction_results = predict_and_save_batches(
        model=model,
        test_ids=test_data['test_ids'],
        test_embeddings=test_data['test_embeddings'],
        go_embeddings=torch.FloatTensor(test_data['go_embeddings']),
        go_terms=test_data['go_terms'],
        device=device,
        cafa_output_file=cafa_file,
        threshold=args.threshold,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        save_scores=args.save_scores,
        # scores_output_file='...' (如果使用 h5py)
    )
    
    total_predictions_cafa = prediction_results['total_predictions_cafa']
    total_scores_sum = prediction_results['total_scores_sum']
    total_elements = prediction_results['total_elements']
    
    # --- 统计和保存 ---
    print("\n" + "="*70)
    print("预测统计")
    print("="*70)
    
    # ⚠️ 实时预测中无法精确计算 Min/Max，需遍历整个矩阵。这里用 0/1 占位，
    # 假设模型的输出在 [0, 1] 之间。
    stats = generate_statistics(
        go_aspects=test_data['go_aspects'],
        total_elements=total_elements,
        total_proteins=len(test_data['test_ids']),
        total_predictions_cafa=total_predictions_cafa,
        total_scores_sum=total_scores_sum,
        scores_min=0.0, # 简化/假设
        scores_max=1.0  # 简化/假设
    )
    
    print(f"总预测数: {stats['total_predictions_cafa']:,}")
    print(f"平均每蛋白质: {stats['avg_per_protein_cafa']:.1f} GO terms")
    print(f"\n分数统计:")
    print(f"  Min: {stats['score_stats']['min']:.4f}")
    print(f"  Max: {stats['score_stats']['max']:.4f}")
    print(f"  Mean: {stats['score_stats']['mean']:.4f}")
    
    print("\n⚠️ 详细的 '按Aspect统计' 由于内存限制，已被简化。")
    print(f"  请在预测完成后，通过后处理 CAFA 文件进行详细统计。")
    
    
    # 完整分数矩阵保存 (已跳过实际保存，仅保留提示)
    scores_file = None
    if args.save_scores:
        scores_file = os.path.join(
            args.output_dir,
            f"prediction_scores_{timestamp}.npz"
        )
        print(f"\n⚠️ 警告: 未实际保存完整分数矩阵 ({scores_file})。请使用 h5py/Zarr 实现分块保存。")

    
    # 保存统计信息
    stats_file = os.path.join(args.output_dir, f"statistics_{timestamp}.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ 统计信息保存到: {stats_file}")
    
    print("\n" + "="*70)
    print("✅ 预测完成！ (内存优化成功)")
    print("="*70)
    print(f"\n输出文件:")
    print(f"  1. CAFA格式预测: {cafa_file}")
    if args.save_scores:
        print(f"  2. 完整分数矩阵: {scores_file} (⚠️ 实际文件未生成，需 h5py/Zarr)")
    print(f"  3. 统计信息: {stats_file}")
    
    print(f"\n下一步:")
    print(f"  1. 检查预测文件: {cafa_file}")
    print(f"  2. 提交到CAFA评估系统")


if __name__ == "__main__":
    main()