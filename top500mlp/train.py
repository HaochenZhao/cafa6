"""
训练脚本：三头MLP模型 + ESM2嵌入
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model_three_head_mlp import ESM2WithThreeHeadMLP, ThreeHeadMLP
from dataset import create_dataloaders
from metrics import compute_metrics_per_aspect, print_metrics


class Trainer:
    """训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metadata: Dict,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_aspect_loss: bool = True,  # 是否使用分aspect的loss
        aspect_loss_weights: Dict[str, float] = None,  # 各aspect的loss权重
        use_class_weights: bool = False,  # 是否使用类别权重（处理类别不平衡）
        save_dir: str = "checkpoints",
        patience: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metadata = metadata
        self.device = device
        self.use_aspect_loss = use_aspect_loss
        self.save_dir = save_dir
        self.patience = patience

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # Aspect loss权重
        if aspect_loss_weights is None:
            aspect_loss_weights = {'C': 1.0, 'F': 1.0, 'P': 1.0}
        self.aspect_loss_weights = aspect_loss_weights

        # 损失函数
        if use_class_weights:
            # 计算正负样本权重
            pos_weights = self._compute_pos_weights()
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        # 创建aspect索引
        self.aspect_indices = {
            'C': np.where(metadata['go_aspects'] == 'C')[0],
            'F': np.where(metadata['go_aspects'] == 'F')[0],
            'P': np.where(metadata['go_aspects'] == 'P')[0]
        }

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_fmax': [],
            'val_aupr': [],
            'best_epoch': 0
        }

        self.best_fmax = 0
        self.epochs_without_improvement = 0

    def _compute_pos_weights(self) -> torch.Tensor:
        """计算正样本权重（用于处理类别不平衡）"""
        # 统计正样本数量
        pos_counts = []
        total_samples = len(self.train_loader.dataset)

        for batch in self.train_loader:
            labels = batch['labels'].numpy()
            pos_counts.append(labels.sum(axis=0))

        pos_counts = np.sum(pos_counts, axis=0)
        neg_counts = total_samples - pos_counts

        # 计算权重: neg / pos
        pos_weights = neg_counts / (pos_counts + 1e-5)
        pos_weights = np.clip(pos_weights, 0.1, 10.0)  # 限制权重范围

        return torch.FloatTensor(pos_weights)

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            # 获取数据
            labels = batch['labels'].to(self.device)
            labels_C = batch['labels_C'].to(self.device)
            labels_F = batch['labels_F'].to(self.device)
            labels_P = batch['labels_P'].to(self.device)

            # 前向传播
            if 'sequences' in batch:
                # 动态模式（ESM2 + MLP）
                sequences = batch['sequences']
                if self.use_aspect_loss:
                    outputs = self.model(sequences, return_separate=True)
                else:
                    outputs = self.model(sequences, return_separate=False)
            else:
                # 预计算嵌入模式（仅MLP）
                embeddings = batch['embeddings'].to(self.device)
                if self.use_aspect_loss:
                    outputs = self.model(embeddings, return_separate=True)
                else:
                    outputs = self.model(embeddings, return_separate=False)

            # 计算损失
            if self.use_aspect_loss:
                # 分aspect计算loss
                loss_C = self.criterion(outputs['C'], labels_C)
                loss_F = self.criterion(outputs['F'], labels_F)
                loss_P = self.criterion(outputs['P'], labels_P)

                loss = (self.aspect_loss_weights['C'] * loss_C +
                        self.aspect_loss_weights['F'] * loss_F +
                        self.aspect_loss_weights['P'] * loss_P)
            else:
                # 整体计算loss
                loss = self.criterion(outputs['logits'], labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """验证"""
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Valid]")

        for batch in pbar:
            # 获取数据
            labels = batch['labels'].to(self.device)
            labels_C = batch['labels_C'].to(self.device)
            labels_F = batch['labels_F'].to(self.device)
            labels_P = batch['labels_P'].to(self.device)

            # 前向传播
            if 'sequences' in batch:
                # 动态模式（ESM2 + MLP）
                sequences = batch['sequences']
                if self.use_aspect_loss:
                    outputs = self.model(sequences, return_separate=True)
                else:
                    outputs = self.model(sequences, return_separate=False)
            else:
                # 预计算嵌入模式（仅MLP）
                embeddings = batch['embeddings'].to(self.device)
                if self.use_aspect_loss:
                    outputs = self.model(embeddings, return_separate=True)
                else:
                    outputs = self.model(embeddings, return_separate=False)

            # 计算损失
            if self.use_aspect_loss:
                loss_C = self.criterion(outputs['C'], labels_C)
                loss_F = self.criterion(outputs['F'], labels_F)
                loss_P = self.criterion(outputs['P'], labels_P)
                loss = (self.aspect_loss_weights['C'] * loss_C +
                        self.aspect_loss_weights['F'] * loss_F +
                        self.aspect_loss_weights['P'] * loss_P)

                # 拼接logits
                logits = torch.cat([outputs['C'], outputs['F'], outputs['P']], dim=1)
            else:
                loss = self.criterion(outputs['logits'], labels)
                logits = outputs['logits']

            total_loss += loss.item()

            # 收集预测和标签
            predictions = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_labels.append(labels_np)
            all_predictions.append(predictions)

        # 合并所有批次
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # 计算指标
        metrics = compute_metrics_per_aspect(
            all_labels,
            all_predictions,
            self.aspect_indices
        )

        avg_loss = total_loss / len(self.val_loader)
        metrics['loss'] = avg_loss

        return metrics

    def train(self, num_epochs: int):
        """训练主循环"""
        print("="*70)
        print("开始训练")
        print("="*70)
        print(f"设备: {self.device}")
        print(f"训练样本: {len(self.train_loader.dataset):,}")
        print(f"验证样本: {len(self.val_loader.dataset):,}")
        print(f"批次大小: {self.train_loader.batch_size}")
        print(f"总epochs: {num_epochs}")
        print(f"使用aspect loss: {self.use_aspect_loss}")
        if self.use_aspect_loss:
            print(f"Aspect权重: {self.aspect_loss_weights}")
        print("="*70)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)

            # 验证
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['loss']
            val_fmax = val_metrics['overall']['fmax']
            val_aupr = val_metrics['overall']['aupr']

            self.history['val_loss'].append(val_loss)
            self.history['val_fmax'].append(val_fmax)
            self.history['val_aupr'].append(val_aupr)

            # 打印结果
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Fmax:   {val_fmax:.4f}")
            print(f"  Val AUPR:   {val_aupr:.4f}")

            # 详细指标
            print_metrics(val_metrics, prefix="  ")

            # 学习率调度
            self.scheduler.step(val_fmax)

            # 保存最佳模型
            if val_fmax > self.best_fmax:
                self.best_fmax = val_fmax
                self.history['best_epoch'] = epoch
                self.epochs_without_improvement = 0

                # 保存模型
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✓ 新的最佳模型! Fmax: {val_fmax:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"  {self.epochs_without_improvement} epochs without improvement")

            # 早停
            if self.epochs_without_improvement >= self.patience:
                print(f"\n早停触发！{self.patience} epochs内没有改进")
                break

            # 定期保存
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

        # 训练结束
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("训练完成!")
        print("="*70)
        print(f"总时间: {total_time/60:.2f} 分钟")
        print(f"最佳epoch: {self.history['best_epoch']}")
        print(f"最佳Fmax: {self.best_fmax:.4f}")
        print("="*70)

        # 保存训练历史
        self.save_history()

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        # 保存最新
        path = os.path.join(self.save_dir, 'last.pth')
        torch.save(checkpoint, path)

        # 保存最佳
        if is_best:
            path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, path)

    def save_history(self):
        """保存训练历史"""
        path = os.path.join(self.save_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="训练三头MLP模型")
    parser.add_argument('--use_precomputed_embeddings', action='store_true',
                        help='使用预计算的ESM2嵌入')
    parser.add_argument('--embeddings_file', type=str, default=None,
                        help='预计算嵌入文件路径 (.npz或.h5)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='模型保存目录')

    args = parser.parse_args()

    # 配置
    config = {
        'data_dir': 'processed_data_v3',
        'batch_size': args.batch_size,
        'num_workers': 4,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'num_epochs': args.num_epochs,
        'patience': 10,
        'hidden_dims': [512, 256],
        'dropout': 0.3,
        'use_shared_layer': True,
        'use_aspect_loss': True,
        'aspect_loss_weights': {'C': 1.0, 'F': 1.0, 'P': 1.5},  # P aspect权重稍高
        'use_class_weights': False,
        'use_precomputed_embeddings': args.use_precomputed_embeddings,
        'embeddings_file': args.embeddings_file,
        'save_dir': args.save_dir if args.save_dir else f'checkpoints/exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载数据...")

    # 加载预计算嵌入（如果使用）
    train_embeddings = None
    val_embeddings = None
    if config['use_precomputed_embeddings']:
        if config['embeddings_file'] is None:
            raise ValueError("使用预计算嵌入时必须指定 --embeddings_file")

        print(f"加载预计算嵌入: {config['embeddings_file']}")

        if config['embeddings_file'].endswith('.npz'):
            emb_data = np.load(config['embeddings_file'])
            train_embeddings = emb_data['train_embeddings']
            val_embeddings = emb_data['val_embeddings']
        elif config['embeddings_file'].endswith('.h5'):
            import h5py
            with h5py.File(config['embeddings_file'], 'r') as f:
                train_embeddings = f['train_embeddings'][:]
                val_embeddings = f['val_embeddings'][:]
        else:
            raise ValueError("嵌入文件必须是 .npz 或 .h5 格式")

        print(f"✓ 嵌入加载完成")
        print(f"  训练集嵌入: {train_embeddings.shape}")
        print(f"  验证集嵌入: {val_embeddings.shape}")

    train_loader, val_loader, metadata = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_precomputed_embeddings=config['use_precomputed_embeddings'],
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings
    )

    # 创建模型
    print("\n创建模型...")
    if config['use_precomputed_embeddings']:
        # 仅MLP模式
        esm_dim = train_embeddings.shape[1]
        model = ThreeHeadMLP(
            esm_dim=esm_dim,
            hidden_dims=config['hidden_dims'],
            num_labels_per_aspect=metadata['num_labels_per_aspect'],
            dropout=config['dropout'],
            use_shared_layer=config['use_shared_layer']
        )
    else:
        # ESM2 + MLP模式
        model = ESM2WithThreeHeadMLP(
            esm_model_name="esm2_t6_8M_UR50D",
            num_labels_per_aspect=metadata['num_labels_per_aspect'],
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout'],
            use_shared_layer=config['use_shared_layer'],
            freeze_esm=True,
            pooling='mean'
        )

    model = model.to(device)

    # 打印模型信息
    print("\n模型信息:")
    if config['use_precomputed_embeddings']:
        param_stats = model.get_num_params()
        for key, value in param_stats.items():
            print(f"  {key}: {value:,}")
    else:
        param_stats = model.get_num_params()
        for key, value in param_stats.items():
            print(f"  {key}: {value:,}")

    # 保存配置
    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        metadata=metadata,
        device=device,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        use_aspect_loss=config['use_aspect_loss'],
        aspect_loss_weights=config['aspect_loss_weights'],
        use_class_weights=config['use_class_weights'],
        save_dir=config['save_dir'],
        patience=config['patience']
    )

    # 开始训练
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()
