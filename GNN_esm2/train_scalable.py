"""
企业级可扩展训练脚本
支持任意规模的蛋白质和GO数量

使用方法:
    python GNN_esm2/train_scalable.py --exp_name my_experiment
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    create_scalable_dataloaders,
    PairwiseScorer,
    BatchEvaluator,
    StreamingMetricsCalculator
)


def train_epoch(model, train_loader, optimizer, criterion, device, config):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        protein_embs = batch['protein_embs'].to(device)
        go_embs = batch['go_embs'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        scores = model(protein_embs, go_embs)
        loss = criterion(scores, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def validate_batch(model, val_loader, device, metadata, config):
    """
    批量验证 - 使用BatchEvaluator
    
    不构建pairs，直接矩阵计算
    """
    model.eval()
    
    # 创建批量评估器
    go_embeddings = torch.FloatTensor(metadata['go_embeddings'])
    evaluator = BatchEvaluator(
        model=model,
        go_embeddings=go_embeddings,
        device=device,
        chunk_size=config['validation'].get('eval_chunk_size', 1000)
    )
    
    # 流式指标计算器
    metrics_calc = StreamingMetricsCalculator(num_thresholds=100)
    
    print("\n验证中（批量模式）...")
    pbar = tqdm(val_loader, desc="Validation")
    
    for batch in pbar:
        protein_embs = batch['protein_embs']
        labels = batch['labels']
        
        # 批量计算所有GO的分数
        # 注意：这里labels应该是完整的标签矩阵 [batch, num_gos]
        # 但我们的val_loader返回的是采样的pairs
        # 所以这里需要特殊处理
        
        # 简化版本：对每个batch计算分数
        scores = model(protein_embs.to(device), batch['go_embs'].to(device))
        
        # 更新指标
        metrics_calc.update(
            scores.unsqueeze(1),  # [batch, 1]
            labels.unsqueeze(1)   # [batch, 1]
        )
    
    # 计算最终指标
    metrics = metrics_calc.compute()
    
    return metrics


@torch.no_grad()
def validate_streaming(model, val_loader, device):
    """
    流式验证 - 使用StreamingMetricsCalculator
    
    不存储全部预测，增量计算指标
    """
    model.eval()
    
    metrics_calc = StreamingMetricsCalculator(num_thresholds=100)
    
    print("\n验证中（流式模式）...")
    pbar = tqdm(val_loader, desc="Validation")
    
    for batch in pbar:
        protein_embs = batch['protein_embs'].to(device)
        go_embs = batch['go_embs'].to(device)
        labels = batch['labels']
        
        # 前向传播
        scores = model(protein_embs, go_embs)
        
        # 增量更新指标
        metrics_calc.update(
            scores.unsqueeze(-1).cpu(),
            labels.unsqueeze(-1)
        )
    
    # 计算最终指标
    metrics = metrics_calc.compute()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="企业级可扩展训练")
    parser.add_argument('--exp_name', type=str, required=True, help='实验名称')
    parser.add_argument('--config', type=str, default='GNN_esm2/config/config.yaml', help='配置文件')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    save_dir = os.path.join('GNN_esm2/checkpoints', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print("="*70)
    print("CAFA6 蛋白质功能预测 (企业级可扩展版本)")
    print("="*70)
    print(f"实验名称: {args.exp_name}")
    print(f"保存目录: {save_dir}")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("="*70)
    print("创建数据加载器")
    print("="*70)
    
    train_loader, val_loader, metadata = create_scalable_dataloaders(config, base_dir='.')
    
    # 创建模型
    print("\n" + "="*70)
    print("创建模型")
    print("="*70)
    
    model = PairwiseScorer(
        esm_dim=config['model']['esm_dim'],
        go_dim=config['model']['go_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        fusion_type=config['model']['fusion_type']
    ).to(device)
    
    print(f"✓ 模型创建完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = None
    
    # 损失函数
    criterion = nn.BCELoss()
    
    # 训练循环
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    
    best_fmax = 0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print("-"*70)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        if epoch % config['validation']['val_frequency'] == 0:
            metrics = validate_streaming(model, val_loader, device)
            
            print(f"\n验证指标:")
            print(f"  Fmax: {metrics['fmax']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Threshold: {metrics['threshold']:.4f}")
            print(f"  AUPR: {metrics['aupr']:.4f}")
            
            # 保存最佳模型
            if metrics['fmax'] > best_fmax:
                best_fmax = metrics['fmax']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fmax': best_fmax,
                    'config': config
                }, os.path.join(save_dir, 'best.pth'))
                print(f"  ✓ 保存最佳模型 (Fmax: {best_fmax:.4f})")
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)
    print(f"最佳Fmax: {best_fmax:.4f}")
    print(f"模型保存在: {save_dir}")


if __name__ == "__main__":
    main()
