"""
CAFA6 企业级训练脚本 (HNM + 详细日志 + 增强绘图版)
功能: 
1. Hard Negative Mining (HNM)
2. Focal Loss / Bias Init 可配
3. 详细日志监控 (LR, Val Loss, Precision, Recall)
4. 增强的训练曲线绘图 (Loss, Fmax/AUPR, Learning Rate)
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
try:
    from src.scalable_dataset import NegativeSampler
except ImportError:
    NegativeSampler = object

# ============================================================================
# 1. 困难负样本采样器 (Hard Aware Negative Sampler)
# ============================================================================
class HardAwareNegativeSampler(NegativeSampler):
    # ... (保持不变) ...
    def __init__(self, all_go_ids: List[str], hard_sample_rate: float = 0.5):
        self.all_go_ids = all_go_ids
        self.all_go_set = set(all_go_ids)
        self.hard_negatives = {} 
        self.hard_sample_rate = hard_sample_rate 

    def update_hard_negatives(self, new_hard_negatives: Dict[str, List[str]]):
        self.hard_negatives = new_hard_negatives
        print(f"  [Sampler] 困难样本池已更新: 覆盖 {len(self.hard_negatives)} 个蛋白质")

    def sample(self, protein_id: str, positive_gos: set, k: int) -> List[str]:
        samples = []
        # 1. 尝试从困难样本池采样
        hard_candidates = self.hard_negatives.get(protein_id, [])
        hard_candidates = [go for go in hard_candidates if go not in positive_gos]
        
        if hard_candidates and random.random() < self.hard_sample_rate:
            num_hard = min(len(hard_candidates), k // 2)
            if num_hard > 0:
                samples.extend(random.sample(hard_candidates, num_hard))
            
        # 2. 剩余的用随机采样补足
        num_random = k - len(samples)
        if num_random > 0:
            added = 0
            max_attempts = num_random * 5
            attempts = 0
            while added < num_random and attempts < max_attempts:
                neg = random.choice(self.all_go_ids)
                if neg not in positive_gos and neg not in samples:
                    samples.append(neg)
                    added += 1
                attempts += 1
            
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
    # ... (保持不变) ...
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_factor * (1-pt)**self.gamma * bce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# ============================================================================
# 3. 困难样本挖掘函数
# ============================================================================
@torch.no_grad()
def mine_hard_negatives(
    model, train_ids, seq_embeddings, go_embeddings, idx_to_go_id, 
    labels_dict, device, num_proteins=2000, threshold=0.2
):
    # ... (保持不变) ...
    model.eval()
    print(f"\n[Mining] 正在挖掘困难负样本 (采样 {num_proteins} 个蛋白质, 阈值 {threshold})...")
    num_samples = min(len(train_ids), num_proteins)
    sampled_indices = np.random.choice(len(train_ids), num_samples, replace=False)
    batch_pids = train_ids[sampled_indices]
    batch_embs = torch.FloatTensor(seq_embeddings[sampled_indices]).to(device)
    go_embs_tensor = torch.FloatTensor(go_embeddings).to(device)
    
    evaluator = BatchEvaluator(model, go_embs_tensor, device, chunk_size=2000)
    logits = evaluator.evaluate_proteins(batch_embs, return_all_scores=True)
    probs = torch.sigmoid(logits)
    
    hard_negatives_dict = {}
    count = 0
    probs_np = probs.cpu().numpy()
    
    for i, pid in enumerate(batch_pids):
        true_gos = set(labels_dict.get(pid, []))
        high_score_indices = np.where(probs_np[i] > threshold)[0]
        hard_gos = []
        for idx in high_score_indices:
            go_id = idx_to_go_id[idx]
            if go_id not in true_gos: hard_gos.append(go_id)
        
        if hard_gos:
            if len(hard_gos) > 50: hard_gos = np.random.choice(hard_gos, 50, replace=False).tolist()
            hard_negatives_dict[pid] = hard_gos
            count += len(hard_gos)
    print(f"[Mining] 挖掘完成! 找到 {count} 个困难负样本，涉及 {len(hard_negatives_dict)} 个蛋白质。")
    return hard_negatives_dict

# ============================================================================
# 4. 训练与验证辅助函数
# ============================================================================
def train_epoch(model, train_loader, optimizer, criterion, device):
    # ... (保持不变) ...
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

@torch.no_grad()
def validate_full_matrix(model, val_loader, device, all_go_embeddings, criterion, chunk_size=1000):
    # ... (保持不变) ...
    model.eval()
    go_embeddings = torch.FloatTensor(all_go_embeddings).to(device)
    evaluator = BatchEvaluator(model, go_embeddings, device, chunk_size=chunk_size)
    metrics_calc = StreamingMetricsCalculator(num_thresholds=100)
    
    total_min, total_max, total_mean = 1.0, 0.0, 0.0
    total_loss = 0.0
    n_b = 0
    
    for batch in tqdm(val_loader, desc="Full Val"):
        protein_embs = batch[0].to(device)
        true_labels = batch[1].to(device)
        logits = evaluator.evaluate_proteins(protein_embs, return_all_scores=True)
        
        loss = criterion(logits, true_labels)
        total_loss += loss.item()
        
        scores = torch.sigmoid(logits)
        metrics_calc.update(scores, true_labels)
        
        total_min = min(total_min, scores.min().item())
        total_max = max(total_max, scores.max().item())
        total_mean += scores.mean().item()
        n_b += 1
        
    stats = {'min': total_min, 'max': total_max, 'mean': total_mean/n_b if n_b else 0}
    return metrics_calc.compute(), stats, total_loss / n_b if n_b else 0

# [新增/修改] 增强的绘图函数
def plot_training_curves(history, save_dir):
    """绘制 Loss, Fmax/AUPR, Learning Rate 三合一图表"""
    try:
        # 创建一个 3 行 1 列的画布
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # 计算 Epoch 轴
        epochs_train = range(1, len(history['train_loss']) + 1)
        val_freq = history.get('val_freq', 1)
        epochs_val = [i * val_freq for i in range(1, len(history['val_fmax']) + 1)]

        # --- 子图 1: Loss 曲线 (Train vs Val) ---
        ax_loss = axes[0]
        ax_loss.plot(epochs_train, history['train_loss'], label='Train Loss', color='tab:blue', linewidth=2)
        if len(history['val_loss']) > 0:
            # 确保 Val Loss 对齐到正确的 Epoch
            ax_loss.plot(epochs_val, history['val_loss'], label='Val Loss', color='tab:orange', linestyle='--', marker='o', markersize=4)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_loss.set_title('Training & Validation Loss', fontsize=14)
        ax_loss.legend(fontsize=12)
        ax_loss.grid(True, alpha=0.3)

        # --- 子图 2: Metrics 曲线 (Fmax & AUPR) ---
        ax_metrics = axes[1]
        if len(history['val_fmax']) > 0:
            ax_metrics.plot(epochs_val, history['val_fmax'], label='Fmax', color='tab:green', marker='o', linewidth=2)
            ax_metrics.plot(epochs_val, history['val_aupr'], label='AUPR', color='tab:purple', marker='s', linewidth=2)
        ax_metrics.set_ylabel('Score', fontsize=12)
        ax_metrics.set_title('Validation Metrics (Fmax & AUPR)', fontsize=14)
        ax_metrics.legend(fontsize=12)
        ax_metrics.grid(True, alpha=0.3)
        ax_metrics.set_ylim(bottom=0.0, top=1.0) # 限制 metric 范围在 0-1

        # --- 子图 3: Learning Rate 曲线 ---
        ax_lr = axes[2]
        if len(history['lr']) > 0:
            ax_lr.plot(epochs_train, history['lr'], label='Learning Rate', color='tab:red', linestyle='-.')
        ax_lr.set_ylabel('Learning Rate (log scale)', fontsize=12)
        ax_lr.set_xlabel('Epoch', fontsize=14)
        ax_lr.set_title('Learning Rate Schedule', fontsize=14)
        ax_lr.set_yscale('log') # LR 通常用对数坐标显示
        ax_lr.legend(fontsize=12)
        ax_lr.grid(True, alpha=0.3, which='both')

        # 调整布局并保存
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'training_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [Plot] 训练总结图已保存至: {save_path}")
        plt.close(fig) # 关闭画布释放内存

    except Exception as e:
        print(f"Warning: 绘图失败: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 5. 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="企业级 HNM 训练 (Ablation + Log + Plot)")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--config', type=str, default='GNN_esm2/config/config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join('GNN_esm2/checkpoints', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f: yaml.dump(config, f)
    
    print(f"=== 启动实验: {args.exp_name} ===")
    print(f"配置: BiasInit={config['model'].get('use_bias_init')}, Loss={config['training']['loss']['type']}")
    
    # 1. 加载基础数据
    print("加载基础数据...")
    data = load_processed_data(config['data']['processed_dir'])
    emb_data = np.load(config['esm2']['embeddings_file'])
    
    # 2. 准备全量验证集 Loader
    val_loader_full = DataLoader(
        TensorDataset(torch.FloatTensor(emb_data['val_embeddings']), torch.FloatTensor(data['val_labels'])),
        batch_size=config['inference']['batch_size'] // 2, shuffle=False, num_workers=2
    )
    
    # 3. 准备训练集 Loader & Sampler
    train_loader, _, metadata = create_scalable_dataloaders(config, base_dir='.')
    all_go_ids = list(metadata['go_metadata']['go_id_to_idx'].keys())
    
    # [HNM] Hard Negative Mining 配置
    hnm_cfg = config['training'].get('hard_negative_mining', {'enabled': False})
    
    if hnm_cfg['enabled']:
        print(f"✓ [HNM] 启用困难负样本挖掘 (Rate: {hnm_cfg.get('hard_sample_rate', 0.5)})")
        hard_sampler = HardAwareNegativeSampler(
            all_go_ids, 
            hard_sample_rate=hnm_cfg.get('hard_sample_rate', 0.5)
        )
        if hasattr(train_loader.dataset, 'negative_sampler'):
            train_loader.dataset.negative_sampler = hard_sampler
            print("  -> 已注入 HardAwareNegativeSampler")
    else:
        print("✓ [HNM] 禁用困难负样本挖掘 (使用默认随机采样)")

    # 4. 模型初始化
    use_bias_init = config['model'].get('use_bias_init', True)
    model = PairwiseScorer(
        esm_dim=config['model']['esm_dim'],
        go_dim=config['model']['go_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        fusion_type=config['model']['fusion_type'],
        use_bias_init=use_bias_init
    ).to(device)
    
    # 5. Loss 选择
    loss_cfg = config['training'].get('loss', {'type': 'bce'})
    if loss_cfg['type'] == 'focal':
        alpha = loss_cfg['params']['alpha']
        gamma = loss_cfg['params']['gamma']
        print(f"✓ [Loss] 使用 Focal Loss (alpha={alpha}, gamma={gamma})")
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        print(f"✓ [Loss] 使用 Standard BCE Loss")
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    # 6. 训练循环
    history = {
        'train_loss': [], 'val_loss': [], 
        'val_fmax': [], 'val_aupr': [], 
        'lr': [], 'val_freq': config['validation']['val_frequency']
    }
    best_fmax = 0.0
    idx_to_go_id = metadata['go_metadata']['idx_to_go_id']
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        
        # --- 训练 ---
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        # 记录 LR
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # --- 挖掘困难样本 ---
        if (hnm_cfg['enabled'] and 
            epoch >= hnm_cfg.get('start_epoch', 1) and 
            epoch < config['training']['num_epochs'] and 
            epoch % hnm_cfg.get('mining_interval', 1) == 0):
            
            new_hard_negatives = mine_hard_negatives(
                model=model,
                train_ids=data['train_ids'],
                seq_embeddings=emb_data['train_embeddings'],
                go_embeddings=metadata['go_embeddings'],
                idx_to_go_id=idx_to_go_id,
                labels_dict=train_loader.dataset.labels_dict,
                device=device,
                num_proteins=hnm_cfg.get('num_proteins', 2000),
                threshold=hnm_cfg.get('threshold', 0.05)
            )
            if hasattr(train_loader.dataset, 'negative_sampler') and \
               isinstance(train_loader.dataset.negative_sampler, HardAwareNegativeSampler):
                train_loader.dataset.negative_sampler.update_hard_negatives(new_hard_negatives)
        
        # --- 验证 ---
        if epoch % history['val_freq'] == 0:
            metrics, stats, val_loss = validate_full_matrix(
                model, val_loader_full, device, metadata['go_embeddings'], 
                criterion, chunk_size=config['validation'].get('eval_chunk_size', 1000)
            )
            
            fmax, aupr = metrics['fmax'], metrics['aupr']
            history['val_fmax'].append(fmax); history['val_aupr'].append(aupr)
            history['val_loss'].append(val_loss)
            
            # [详细日志打印]
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
            print(f"  Val Fmax: {fmax:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, T={metrics['threshold']:.4f})")
            print(f"  Val AUPR: {aupr:.4f}")
            print(f"  Score Dist: Mean={stats['mean']:.4f}, Range=[{stats['min']:.4f}, {stats['max']:.4f}]")
            
            if fmax > best_fmax:
                best_fmax = fmax
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'fmax': fmax, 'config': config}, os.path.join(save_dir, 'best.pth'))
                print(f"  ⭐ New Best Fmax!")
            
            torch.save({'epoch': epoch, 'model': model.state_dict()}, os.path.join(save_dir, 'last.pth'))
            with open(os.path.join(save_dir, 'history.json'), 'w') as f: json.dump(history, f)
            
            # [新增] 调用增强的绘图函数
            plot_training_curves(history, save_dir)

    print(f"\n训练完成! 最佳 Fmax: {best_fmax:.4f}")

if __name__ == "__main__":
    main()