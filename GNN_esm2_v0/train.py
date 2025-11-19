"""
CAFA6训练脚本 - Pairwise模型
从cafa主目录运行: python GNN_esm2/train.py
"""

import os
import sys
import yaml
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from collections import defaultdict

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import create_dataloaders
from model import PairwiseScoringModel
from metrics import compute_fmax, compute_aupr


class Trainer:
    """Pairwise模型训练器"""
    
    def __init__(self, config: dict, save_dir: str, base_dir: str = '.'):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {self.device}")
        
        # 创建数据加载器
        print("\n" + "="*70)
        print("创建数据加载器")
        print("="*70)
        self.train_loader, self.val_loader, self.metadata = create_dataloaders(config, base_dir)
        
        # 创建模型
        print("\n" + "="*70)
        print("创建模型")
        print("="*70)
        self.model = PairwiseScoringModel(config).to(self.device)
        
        # 打印参数统计
        param_stats = self.model.get_num_params()
        print(f"\n模型参数统计:")
        for key, value in param_stats.items():
            print(f"  {key}: {value:,}")
        
        # 优化器
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor'],
            min_lr=config['training']['scheduler']['min_lr'],
            verbose=True
        )
        
        # 损失函数
        if config['training']['loss']['type'] == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif config['training']['loss']['type'] == 'focal':
            self.criterion = self._focal_loss
            self.focal_alpha = config['training']['loss']['focal_alpha']
            self.focal_gamma = config['training']['loss']['focal_gamma']
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_fmax': [],
            'val_aupr': [],
            'lr': []
        }
        
        # 最佳指标
        self.best_fmax = 0.0
        self.epochs_without_improvement = 0
    
    def _focal_loss(self, logits, targets):
        """Focal loss"""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        return focal_loss.mean()
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            seq_embs = batch['seq_embs'].to(self.device)
            go_embs = batch['go_embs'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            logits = self.model(seq_embs, go_embs)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """验证 - 评估所有(protein, go)对"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # 收集所有预测和标签
        all_predictions = []
        all_labels = []
        
        print("\n验证中...")
        for batch in tqdm(self.val_loader, desc="验证"):
            seq_embs = batch['seq_embs'].to(self.device)
            go_embs = batch['go_embs'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            logits = self.model(seq_embs, go_embs)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测
            probs = torch.sigmoid(logits).cpu().numpy()
            all_predictions.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / num_batches
        
        # 转换为numpy数组
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        # 计算指标
        fmax_result = compute_fmax(labels, predictions)
        aupr = compute_aupr(labels, predictions)
        
        metrics = {
            'fmax': fmax_result['fmax'],
            'precision': fmax_result['precision'],
            'recall': fmax_result['recall'],
            'threshold': fmax_result['threshold'],
            'aupr': aupr
        }
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, self.save_dir / 'last.pth')
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"  ✓ 保存最佳模型 (Fmax: {metrics['fmax']:.4f})")
        
        if epoch % self.config['training']['save_every'] == 0:
            torch.save(checkpoint, self.save_dir / f'epoch_{epoch}.pth')
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*70)
        print("开始训练")
        print("="*70)
        
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['early_stopping']['patience']
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate()
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['fmax'])
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_fmax'].append(val_metrics['fmax'])
            self.history['val_aupr'].append(val_metrics['aupr'])
            self.history['lr'].append(current_lr)
            
            # 打印结果
            print(f"\n  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率:   {current_lr:.6f}")
            print(f"\n  验证指标:")
            print(f"    Fmax:      {val_metrics['fmax']:.4f} "
                  f"(P={val_metrics['precision']:.4f}, "
                  f"R={val_metrics['recall']:.4f}, "
                  f"T={val_metrics['threshold']:.4f})")
            print(f"    AUPR:      {val_metrics['aupr']:.4f}")
            
            # 检查是否是最佳模型
            current_fmax = val_metrics['fmax']
            is_best = current_fmax > self.best_fmax
            
            if is_best:
                self.best_fmax = current_fmax
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # 保存checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 早停
            if self.epochs_without_improvement >= patience:
                print(f"\n早停触发！{patience}个epoch验证指标未提升")
                break
        
        # 保存历史
        with open(self.save_dir / 'history.json', 'w') as f:
            history_serializable = {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_fmax': [float(x) for x in self.history['val_fmax']],
                'val_aupr': [float(x) for x in self.history['val_aupr']],
                'lr': [float(x) for x in self.history['lr']]
            }
            json.dump(history_serializable, f, indent=2)
        
        print("\n" + "="*70)
        print("训练完成!")
        print("="*70)
        print(f"\n最佳验证Fmax: {self.best_fmax:.4f}")
        print(f"模型保存在: {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="CAFA6 Pairwise训练")
    parser.add_argument('--config', type=str,
                        default='GNN_esm2/config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称')
    
    args = parser.parse_args()
    
    # 确保在cafa主目录下运行
    if not os.path.exists('cafa-6-protein-function-prediction'):
        print("错误: 请在cafa主目录下运行此脚本")
        print("用法: python GNN_esm2/train.py [options]")
        sys.exit(1)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建实验目录
    if args.exp_name is None:
        exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_name = args.exp_name
    
    save_dir = Path(config['training']['save_dir']) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("="*70)
    print("CAFA6 蛋白质功能预测 (Pairwise模型)")
    print("="*70)
    print(f"\n实验名称: {exp_name}")
    print(f"保存目录: {save_dir}")
    
    # 训练
    trainer = Trainer(config, save_dir, base_dir='.')
    trainer.train()


if __name__ == "__main__":
    main()
