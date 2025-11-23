"""
CAFA6 企业级训练脚本 (HNM 版)
功能: 
1. Hard Negative Mining (HNM): 动态挖掘困难样本
2. Focal Loss: 专注难分样本
3. Full Matrix Validation: 真实评估
4. Monitoring: 实时绘图
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Set

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    create_scalable_dataloaders,
    PairwiseScorer,
    BatchEvaluator,
    StreamingMetricsCalculator,
    load_processed_data
)
# 尝试导入基类，如果失败则使用 object (鸭子类型)
try:
    from src.scalable_dataset import NegativeSampler
except ImportError:
    NegativeSampler = object

# ============================================================================
# 1. 困难负样本采样器 (Hard Aware Negative Sampler)
# ============================================================================
class HardAwareNegativeSampler(NegativeSampler):
    """
    支持困难样本挖掘的采样器
    逻辑: 以一定概率(hard_sample_rate)优先从挖掘到的困难池中采样
    """
    
    def __init__(self, all_go_ids: List[str], hard_sample_rate: float = 0.5):
        self.all_go_ids = all_go_ids
        self.all_go_set = set(all_go_ids)
        # 困难样本池: {protein_id: [go_id1, go_id2, ...]}
        self.hard_negatives = {} 
        self.hard_sample_rate = hard_sample_rate 

    def update_hard_negatives(self, new_hard_negatives: Dict[str, List[str]]):
        """更新困难样本池 (增量更新或全量替换，这里采用替换策略以保持新鲜度)"""
        # 也可以选择合并: self.hard_negatives.update(new_hard_negatives)
        self.hard_negatives = new_hard_negatives
        print(f"  [Sampler] 困难样本池已更新: 覆盖 {len(self.hard_negatives)} 个蛋白质")

    def sample(self, protein_id: str, positive_gos: set, k: int) -> List[str]:
        samples = []
        
        # 1. 尝试从困难样本池采样
        hard_candidates = self.hard_negatives.get(protein_id, [])
        # 过滤掉可能已经变成正样本的（极少见但为了安全）
        hard_candidates = [go for go in hard_candidates if go not in positive_gos]
        
        num_hard = 0
        if hard_candidates and random.random() < self.hard_sample_rate:
            # 决定采样多少个困难样本 (例如一半)
            num_hard = min(len(hard_candidates), k // 2)
            if num_hard > 0:
                samples.extend(random.sample(hard_candidates, num_hard))
            
        # 2. 剩余的用随机采样补足
        num_random = k - len(samples)
        if num_random > 0:
            # 简单的随机采样策略：随机选一个，如果不在正样本和已选样本中则加入
            # 对于巨大的空间，这种 rejection sampling 效率很高
            added = 0
            max_attempts = num_random * 5
            attempts = 0
            while added < num_random and attempts < max_attempts:
                neg = random.choice(self.all_go_ids)
                if neg not in positive_gos and neg not in samples:
                    samples.append(neg)
                    added += 1
                attempts += 1
            
            # 如果运气不好没采够，就用集合差集（慢但保证数量）
            if added < num_random:
                remaining_pool = list(self.all_go_set - positive_gos - set(samples))
                if remaining_pool:
                    needed = num_random - added
                    samples.extend(random.sample(remaining_pool, min(len(remaining_pool), needed)))
                
        return samples

# ============================================================================
# 2. Focal Loss
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# 3. 困难样本挖掘函数
# ============================================================================
@torch.no_grad()
def mine_hard_negatives(
    model, 
    train_ids, 
    seq_embeddings, 
    go_embeddings, 
    idx_to_go_id, 
    labels_dict, 
    device, 
    num_proteins=2000, 
    threshold=0.2
):
    """
    挖掘困难负样本: 随机抽样部分训练集 -> 全量预测 -> 找出 FP
    """
    model.eval()
    print(f"\n[Mining] 正在挖掘困难负样本 (采样 {num_proteins} 个蛋白质)...")
    
    # 1. 随机采样一部分训练集蛋白质
    # 确保不超过总数
    num_samples = min(len(train_ids), num_proteins)
    sampled_indices = np.random.choice(len(train_ids), num_samples, replace=False)
    
    batch_pids = train_ids[sampled_indices]
    batch_embs = torch.FloatTensor(seq_embeddings[sampled_indices]).to(device)
    go_embs_tensor = torch.FloatTensor(go_embeddings).to(device)
    
    # 2. 全量预测
    evaluator = BatchEvaluator(model, go_embs_tensor, device, chunk_size=2000)
    logits = evaluator.evaluate_proteins(batch_embs, return_all_scores=True)
    probs = torch.sigmoid(logits)
    
    hard_negatives_dict = {}
    count = 0
    
    # 3. 筛选 (CPU处理)
    probs_np = probs.cpu().numpy()
    
    for i, pid in enumerate(batch_pids):
        true_gos = set(labels_dict.get(pid, []))
        
        # 找出分数 > 阈值的索引 (潜在 FP)
        high_score_indices = np.where(probs_np[i] > threshold)[0]
        
        hard_gos = []
        for idx in high_score_indices:
            go_id = idx_to_go_id[idx]
            # 关键：必须是负样本 (不在真实标签中)
            if go_id not in true_gos:
                hard_gos.append(go_id)
        
        if hard_gos:
            # 限制数量，防止单个蛋白的负样本过于集中
            if len(hard_gos) > 50:
                hard_gos = np.random.choice(hard_gos, 50, replace=False).tolist()
            hard_negatives_dict[pid] = hard_gos
            count += len(hard_gos)
            
    print(f"[Mining] 挖掘完成! 找到 {count} 个困难负样本，涉及 {len(hard_negatives_dict)} 个蛋白质。")
    return hard_negatives_dict

# ============================================================================
# 4. 全量验证 & 绘图工具
# ============================================================================
def plot_training_curves(history, save_dir):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150); plt.close()

        if len(history['val_fmax']) > 0:
            val_epochs = [i * history['val_freq'] for i in range(1, len(history['val_fmax']) + 1)]
            plt.figure(figsize=(10, 6))
            plt.plot(val_epochs, history['val_fmax'], label='Fmax', marker='o')
            plt.plot(val_epochs, history['val_aupr'], label='AUPR', marker='s')
            plt.title('Full Matrix Validation Metrics')
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'metrics_curve.png'), dpi=150); plt.close()
            
            # 分布图
            if 'val_score_mean' in history:
                plt.figure(figsize=(10, 6))
                plt.plot(val_epochs, history['val_score_mean'], label='Mean', color='blue')
                plt.fill_between(val_epochs, history['val_score_min'], history['val_score_max'], color='gray', alpha=0.2, label='Range')
                plt.title('Score Distribution')
                plt.ylim(0, 1.0)
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(save_dir, 'score_dist.png'), dpi=150); plt.close()
    except Exception as e: print(f"绘图错: {e}")

@torch.no_grad()
def validate_full_matrix(model, val_loader, device, all_go_embeddings, chunk_size=1000):
    model.eval()
    go_embeddings = torch.FloatTensor(all_go_embeddings).to(device)
    evaluator = BatchEvaluator(model, go_embeddings, device, chunk_size=chunk_size)
    metrics_calc = StreamingMetricsCalculator(num_thresholds=100)
    
    total_min, total_max, total_mean = 1.0, 0.0, 0.0
    n_b = 0
    
    print(f"验证中... (Prot: {len(val_loader.dataset)})")
    for batch in tqdm(val_loader, desc="Full Val"):
        protein_embs = batch[0].to(device)
        true_labels = batch[1].to(device)
        
        logits = evaluator.evaluate_proteins(protein_embs, return_all_scores=True)
        scores = torch.sigmoid(logits)
        
        metrics_calc.update(scores, true_labels)
        
        total_min = min(total_min, scores.min().item())
        total_max = max(total_max, scores.max().item())
        total_mean += scores.mean().item()
        n_b += 1
        
    stats = {'min': total_min, 'max': total_max, 'mean': total_mean/n_b if n_b else 0}
    print(f"  [Stats] Range: [{stats['min']:.4f}, {stats['max']:.4f}], Mean: {stats['mean']:.4f}")
    return metrics_calc.compute(), stats

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, n_b = 0, 0
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        protein_embs = batch['protein_embs'].to(device)
        go_embs = batch['go_embs'].to(device)
        labels = batch['labels'].to(device).float()
        
        logits = model(protein_embs, go_embs)
        loss = criterion(logits, labels.squeeze())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item(); n_b += 1
        pbar.set_postfix({'loss': f'{total_loss/n_b:.4f}'})
    return total_loss / n_b

# ============================================================================
# 5. 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="企业级 HNM 训练")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--config', type=str, default='GNN_esm2/config/config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join('GNN_esm2/checkpoints', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f: yaml.dump(config, f)
    
    print(f"=== 启动实验: {args.exp_name} (Hard Negative Mining Enabled) ===")
    
    # 1. 加载基础数据和 Embeddings (用于挖掘和全量验证)
    print("加载基础数据...")
    processed_dir = config['data']['processed_dir']
    data = load_processed_data(processed_dir) # train_ids, val_ids, etc.
    emb_file = config['esm2']['embeddings_file']
    emb_data = np.load(emb_file) # train_embeddings, val_embeddings
    
    # 2. 准备验证集 Loader (全量)
    val_protein_embs = torch.FloatTensor(emb_data['val_embeddings'])
    val_labels_dense = torch.FloatTensor(data['val_labels'])
    val_loader_full = DataLoader(
        TensorDataset(val_protein_embs, val_labels_dense),
        batch_size=config['inference']['batch_size'] // 2, shuffle=False, num_workers=2
    )
    
    # 3. 创建训练集 Loader (使用自定义 HNM Sampler)
    train_loader, _, metadata = create_scalable_dataloaders(config, base_dir='.')
    
    # [HNM 核心] 替换 dataset 的 sampler 为 HardAwareNegativeSampler
    all_go_ids = list(metadata['go_metadata']['go_id_to_idx'].keys())
    hard_sampler = HardAwareNegativeSampler(all_go_ids, hard_sample_rate=0.5)
    # 假设 create_scalable_dataloaders 返回的是 DataLoader，其 dataset 是 ScalablePairwiseDataset
    if hasattr(train_loader.dataset, 'negative_sampler'):
        train_loader.dataset.negative_sampler = hard_sampler
        print("✓ 已注入 HardAwareNegativeSampler")
    else:
        print("⚠️ 警告: 无法注入 HNM Sampler，Dataset 类型不匹配？")

    # 4. 模型 & 优化器
    model = PairwiseScorer(
        esm_dim=config['model']['esm_dim'],
        go_dim=config['model']['go_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        fusion_type=config['model']['fusion_type']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    loss_cfg = config['training'].get('loss', {'alpha': 0.25, 'gamma': 2.0})
    criterion = FocalLoss(alpha=loss_cfg['alpha'], gamma=loss_cfg['gamma'])
    
    # 5. 训练循环
    history = {'train_loss': [], 'val_fmax': [], 'val_aupr': [], 'val_score_mean': [], 'val_score_min': [], 'val_score_max': [], 'val_freq': config['validation']['val_frequency']}
    best_fmax = 0.0
    
    # 准备挖掘所需的映射
    idx_to_go_id = metadata['go_metadata']['idx_to_go_id']
    go_id_to_idx = metadata['go_metadata']['go_id_to_idx']
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        
        # --- 训练 ---
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(loss)
        print(f"  Train Loss: {loss:.6f}")
        
        # --- 挖掘困难样本 (从第1个epoch开始，或者loss稳定后) ---
        # 每个 epoch 结束挖掘一次，为下一个 epoch 准备数据
        if epoch < config['training']['num_epochs']:
            new_hard_negatives = mine_hard_negatives(
                model=model,
                train_ids=data['train_ids'],
                seq_embeddings=emb_data['train_embeddings'],
                go_embeddings=metadata['go_embeddings'],
                idx_to_go_id=idx_to_go_id,
                labels_dict=train_loader.dataset.labels_dict,
                device=device,
                num_proteins=2000, # 每次挖掘2000个蛋白
                threshold=0.2      # 挖掘阈值，大于此值的负样本被视为困难
            )
            # 更新采样器
            if hasattr(train_loader.dataset, 'negative_sampler'):
                train_loader.dataset.negative_sampler.update_hard_negatives(new_hard_negatives)
        
        # --- 全量验证 ---
        if epoch % history['val_freq'] == 0:
            metrics, stats = validate_full_matrix(model, val_loader_full, device, metadata['go_embeddings'], chunk_size=config['validation'].get('eval_chunk_size', 1000))
            
            fmax, aupr = metrics['fmax'], metrics['aupr']
            history['val_fmax'].append(fmax); history['val_aupr'].append(aupr)
            history['val_score_mean'].append(stats['mean']); history['val_score_min'].append(stats['min']); history['val_score_max'].append(stats['max'])
            
            print(f"  Val Fmax: {fmax:.4f} | AUPR: {aupr:.4f}")
            
            if fmax > best_fmax:
                best_fmax = fmax
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'fmax': fmax, 'config': config}, os.path.join(save_dir, 'best.pth'))
                print(f"  ⭐ New Best!")
            
            torch.save({'epoch': epoch, 'model': model.state_dict()}, os.path.join(save_dir, 'last.pth'))
            with open(os.path.join(save_dir, 'history.json'), 'w') as f: json.dump(history, f)
            plot_training_curves(history, save_dir)

    print(f"\n训练完成! 最佳 Fmax: {best_fmax:.4f}")

if __name__ == "__main__":
    main()