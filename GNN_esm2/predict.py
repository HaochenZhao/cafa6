"""
CAFA6推理脚本 - Pairwise模型 (支持 Top-K + Low Threshold)
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src import PairwiseScorer, BatchEvaluator
from src.dataset import load_processed_data, load_go_embeddings

# ... (load_checkpoint, load_test_data 保持不变) ...
def load_checkpoint(checkpoint_path: str, device: torch.device):
    print(f"\n加载模型: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {'model': {'esm_dim': 320, 'go_dim': 256, 'hidden_dims': [512, 256, 128], 'dropout': 0.3, 'fusion_type': 'concat'}}
    
    model = PairwiseScorer(
        esm_dim=config['model']['esm_dim'],
        go_dim=config['model']['go_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        fusion_type=config['model']['fusion_type']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

def load_test_data(config: dict, base_dir: str = '.'):
    # ... (同原版 predict.py) ...
    # 为节省篇幅，此处省略，逻辑与原版一致
    # 需确保正确加载 test_ids, test_embeddings, go_embeddings
    processed_dir = os.path.join(base_dir, config['data']['processed_dir'])
    data = load_processed_data(processed_dir)
    test_ids = data['test_ids']
    go_terms = data['go_terms'].tolist()
    go_aspects = data['go_aspects']
    
    go_embeddings_path = os.path.join(base_dir, config['data']['go_embeddings_dir'])
    full_go_embeddings, full_go_metadata = load_go_embeddings(go_embeddings_path)
    
    filtered_indices = []
    filtered_go_ids = []
    for go_id in go_terms:
        if go_id in full_go_metadata['go_id_to_idx']:
            idx = full_go_metadata['go_id_to_idx'][go_id]
            filtered_indices.append(idx)
            filtered_go_ids.append(go_id)
    go_embeddings = full_go_embeddings[filtered_indices]
    
    emb_file = os.path.join(base_dir, config['esm2']['embeddings_file'])
    emb_data = np.load(emb_file)
    test_embeddings = emb_data['test_embeddings']
    
    return {'test_ids': test_ids, 'test_embeddings': test_embeddings, 'go_embeddings': go_embeddings, 'go_terms': filtered_go_ids, 'go_aspects': go_aspects}

def save_cafa_batch(
    protein_ids: np.ndarray,
    go_terms: List[str],
    scores: np.ndarray,
    output_file: str,
    threshold: float = 0.0,
    write_mode: str = 'a',
    top_k: int = None  # [新增]
) -> int:
    """
    保存批次结果，支持 Top-K 筛选
    """
    num_written = 0
    
    with open(output_file, write_mode) as f:
        if write_mode == 'w':
            f.write("AUTHOR Your_Name\nMODEL GNN_ESM2_Pairwise\nKEYWORDS deep learning\n")

        for i, protein_id in enumerate(protein_ids):
            protein_scores = scores[i]
            
            # 1. 预测 (如果是Logits，这里先Sigmoid；如果模型已有Sigmoid，则直接用)
            # 注意：新版模型输出了 Logits，所以这里最好再加一个 sigmoid 保护
            # 但 BatchEvaluator 通常直接返回模型输出。
            # 如果模型修改为输出 Logits，这里必须做 Sigmoid。
            # 假设我们在 BatchEvaluator 或模型外层已经处理了，或者在这里处理
            # 为了安全，这里做个判断，如果范围在 [0,1] 外，就 sigmoid
            # 但简单起见，假设输入已经是概率（BatchEvaluator应处理，或者模型保留了Sigmoid）
            # *修正*: 由于我们修改了模型输出Logits，BatchEvaluator直接返回Logits。
            # 我们需要在这里转概率。
            probs = 1 / (1 + np.exp(-protein_scores)) # Sigmoid
            
            # 2. 阈值筛选
            valid_indices = np.where(probs >= threshold)[0]
            if len(valid_indices) == 0:
                continue

            # 3. Top-K 筛选
            if top_k is not None and len(valid_indices) > top_k:
                valid_probs = probs[valid_indices]
                # 找到前 K 大的索引
                top_k_local = np.argsort(valid_probs)[-top_k:][::-1]
                final_indices = valid_indices[top_k_local]
            else:
                # 排序
                final_indices = valid_indices[np.argsort(probs[valid_indices])[::-1]]

            # 4. 写入
            for go_idx in final_indices:
                go_id = go_terms[go_idx]
                conf = probs[go_idx]
                f.write(f"{protein_id}\t{go_id}\t{conf:.6f}\n")
                num_written += 1
                
    return num_written

def predict_and_save_batches(model, test_ids, test_embeddings, go_embeddings, go_terms, device, cafa_output_file, threshold, batch_size, chunk_size, top_k):
    evaluator = BatchEvaluator(model, torch.FloatTensor(go_embeddings).to(device), device, chunk_size)
    num_batches = (len(test_embeddings) + batch_size - 1) // batch_size
    
    # Init file
    save_cafa_batch(np.array([]), [], np.array([]), cafa_output_file, threshold, 'w', top_k)
    
    total_pred = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_embeddings), batch_size), total=num_batches, desc="Predicting"):
            end_idx = min(i + batch_size, len(test_embeddings))
            batch_ids = test_ids[i:end_idx]
            batch_embs = torch.FloatTensor(test_embeddings[i:end_idx]).to(device)
            
            # 输出 Logits
            batch_logits = evaluator.evaluate_proteins(batch_embs, return_all_scores=True).cpu().numpy()
            
            # 保存 (内部会做 Sigmoid)
            n = save_cafa_batch(batch_ids, go_terms, batch_logits, cafa_output_file, threshold, 'a', top_k)
            total_pred += n
            
    return total_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='GNN_esm2/config/config.yaml')
    parser.add_argument('--output_dir', type=str, default='GNN_esm2/outputs')
    parser.add_argument('--threshold', type=float, default=0.001, help="建议设低 (0.001)")
    parser.add_argument('--top_k', type=int, default=1500, help="Top-K 筛选")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载
    model, config = load_checkpoint(args.checkpoint, device)
    test_data = load_test_data(config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"submission_{timestamp}.tsv")
    
    # 预测
    total = predict_and_save_batches(
        model, test_data['test_ids'], test_data['test_embeddings'],
        test_data['go_embeddings'], test_data['go_terms'],
        device, output_file, args.threshold, args.batch_size, 1000, args.top_k
    )
    
    print(f"完成! 总预测: {total:,}, 文件: {output_file}")

if __name__ == "__main__":
    main()