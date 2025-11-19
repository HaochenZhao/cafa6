"""
企业级可扩展Pairwise数据集
支持任意规模的蛋白质和GO数量

核心特性:
1. IterableDataset - 动态生成，内存恒定
2. 可插拔负采样策略
3. 批量矩阵评估
4. 流式指标计算
"""

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random
from abc import ABC, abstractmethod


# ============================================================================
# 负采样引擎
# ============================================================================

class NegativeSampler(ABC):
    """负采样基类"""
    
    @abstractmethod
    def sample(self, protein_id: str, positive_gos: set, k: int) -> List[str]:
        """
        采样k个负样本
        
        参数:
            protein_id: 蛋白质ID
            positive_gos: 正样本GO集合
            k: 采样数量
        
        返回:
            负样本GO列表
        """
        pass


class UniformNegativeSampler(NegativeSampler):
    """均匀随机负采样"""
    
    def __init__(self, all_go_ids: List[str]):
        self.all_go_ids = all_go_ids
        self.all_go_set = set(all_go_ids)
    
    def sample(self, protein_id: str, positive_gos: set, k: int) -> List[str]:
        negative_pool = list(self.all_go_set - positive_gos)
        
        if len(negative_pool) == 0:
            return []
        
        k = min(k, len(negative_pool))
        return random.sample(negative_pool, k)


class StratifiedNegativeSampler(NegativeSampler):
    """分层负采样（按aspect）"""
    
    def __init__(self, all_go_ids: List[str], go_aspects: Dict[str, str]):
        """
        参数:
            all_go_ids: 所有GO ID列表
            go_aspects: {go_id: aspect} 映射
        """
        self.all_go_ids = all_go_ids
        self.go_aspects = go_aspects
        
        # 按aspect分组
        self.gos_by_aspect = {'C': [], 'F': [], 'P': []}
        for go_id in all_go_ids:
            aspect = go_aspects.get(go_id, 'P')
            self.gos_by_aspect[aspect].append(go_id)
    
    def sample(self, protein_id: str, positive_gos: set, k: int) -> List[str]:
        """确保各aspect的负样本比例平衡"""
        samples = []
        k_per_aspect = k // 3
        
        for aspect in ['C', 'F', 'P']:
            aspect_pool = list(set(self.gos_by_aspect[aspect]) - positive_gos)
            
            if len(aspect_pool) > 0:
                n = min(k_per_aspect, len(aspect_pool))
                samples.extend(random.sample(aspect_pool, n))
        
        # 如果不够k个，从全部negative pool补充
        if len(samples) < k:
            remaining = k - len(samples)
            all_negatives = list(set(self.all_go_ids) - positive_gos - set(samples))
            if len(all_negatives) > 0:
                n = min(remaining, len(all_negatives))
                samples.extend(random.sample(all_negatives, n))
        
        return samples


# ============================================================================
# 可扩展的Iterable数据集
# ============================================================================

class ScalablePairwiseDataset(IterableDataset):
    """
    可扩展的Pairwise数据集 - 动态生成样本对
    
    特性:
    - 不预构建pairs，内存使用O(1)
    - 支持无限数据流
    - 可插拔负采样策略
    - 支持分布式训练
    """
    
    def __init__(
        self,
        protein_ids: np.ndarray,
        seq_embeddings: np.ndarray,
        go_embeddings: np.ndarray,
        go_metadata: dict,
        labels_dict: Dict[str, List[str]],
        negative_sampler: NegativeSampler,
        mode: str = 'train',
        epoch_size: Optional[int] = None,
        seed: int = 42
    ):
        """
        参数:
            protein_ids: 蛋白质ID数组
            seq_embeddings: 序列embeddings [num_proteins, esm_dim]
            go_embeddings: GO embeddings [num_gos, go_dim]
            go_metadata: GO元数据
            labels_dict: {protein_id: [positive_go_ids]}
            negative_sampler: 负采样器
            mode: 'train' 或 'val'
            epoch_size: 每个epoch的样本数（None=自动）
            seed: 随机种子
        """
        super().__init__()
        
        self.protein_ids = protein_ids
        self.seq_embeddings = torch.FloatTensor(seq_embeddings)
        self.go_embeddings = torch.FloatTensor(go_embeddings)
        self.go_metadata = go_metadata
        self.labels_dict = labels_dict
        self.negative_sampler = negative_sampler
        self.mode = mode
        self.seed = seed
        
        # 构建索引
        self.protein_id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
        self.go_id_to_idx = go_metadata['go_id_to_idx']
        
        # 计算epoch大小
        if epoch_size is None:
            # 训练: 每个蛋白平均生成的样本数
            avg_labels = np.mean([len(gos) for gos in labels_dict.values()])
            samples_per_protein = int(avg_labels * (1 + negative_sampler.sample.__code__.co_argcount))
            self.epoch_size = len(protein_ids) * samples_per_protein
        else:
            self.epoch_size = epoch_size
        
        print(f"  {mode}数据集初始化:")
        print(f"    蛋白质数: {len(protein_ids):,}")
        print(f"    GO数: {len(self.go_id_to_idx):,}")
        print(f"    Epoch大小: {self.epoch_size:,} 样本对")
    
    def __iter__(self):
        """生成器 - 动态生成样本对"""
        worker_info = torch.utils.data.get_worker_info()
        
        # 多worker支持
        if worker_info is not None:
            # 分布式：每个worker处理不同的蛋白质
            per_worker = len(self.protein_ids) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.protein_ids)
            protein_subset = self.protein_ids[start:end]
            worker_seed = self.seed + worker_info.id
        else:
            protein_subset = self.protein_ids
            worker_seed = self.seed
        
        # 设置随机种子
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        
        # 生成样本
        count = 0
        max_samples = self.epoch_size // (worker_info.num_workers if worker_info else 1)
        
        while count < max_samples:
            # 随机选择一个蛋白质
            protein_id = random.choice(protein_subset)
            protein_idx = self.protein_id_to_idx[protein_id]
            protein_emb = self.seq_embeddings[protein_idx]
            
            # 获取正样本
            positive_gos = set(self.labels_dict.get(protein_id, []))
            
            # 生成正样本对
            for go_id in positive_gos:
                if go_id in self.go_id_to_idx:
                    go_idx = self.go_id_to_idx[go_id]
                    go_emb = self.go_embeddings[go_idx]
                    
                    yield {
                        'protein_emb': protein_emb,
                        'go_emb': go_emb,
                        'label': torch.tensor(1.0),
                        'protein_id': protein_id,
                        'go_id': go_id
                    }
                    count += 1
                    
                    if count >= max_samples:
                        return
            
            # 采样负样本
            k = len(positive_gos) * 3  # 负样本比例
            negative_gos = self.negative_sampler.sample(protein_id, positive_gos, k)
            
            for go_id in negative_gos:
                if go_id in self.go_id_to_idx:
                    go_idx = self.go_id_to_idx[go_id]
                    go_emb = self.go_embeddings[go_idx]
                    
                    yield {
                        'protein_emb': protein_emb,
                        'go_emb': go_emb,
                        'label': torch.tensor(0.0),
                        'protein_id': protein_id,
                        'go_id': go_id
                    }
                    count += 1
                    
                    if count >= max_samples:
                        return


def collate_fn(batch):
    """Collate function for DataLoader"""
    protein_embs = torch.stack([item['protein_emb'] for item in batch])
    go_embs = torch.stack([item['go_emb'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'protein_embs': protein_embs,
        'go_embs': go_embs,
        'labels': labels
    }


# ============================================================================
# 批量评估器
# ============================================================================

class BatchEvaluator:
    """
    批量矩阵评估 - 不构建pairs，直接矩阵运算
    
    对每个蛋白质，一次性计算对所有GO的分数
    """
    
    def __init__(
        self,
        model,
        go_embeddings: torch.Tensor,
        device: torch.device,
        chunk_size: int = 1000
    ):
        """
        参数:
            model: 预测模型
            go_embeddings: 所有GO的embeddings [num_gos, go_dim]
            device: 计算设备
            chunk_size: GO分块大小（内存优化）
        """
        self.model = model
        self.go_embeddings = go_embeddings.to(device)
        self.device = device
        self.chunk_size = chunk_size
        self.num_gos = len(go_embeddings)
    
    @torch.no_grad()
    def evaluate_proteins(
        self,
        protein_embs: torch.Tensor,
        return_all_scores: bool = False
    ) -> torch.Tensor:
        """
        评估一批蛋白质对所有GO的分数
        
        参数:
            protein_embs: [batch_size, esm_dim]
            return_all_scores: 是否返回完整分数矩阵
        
        返回:
            scores: [batch_size, num_gos]
        """
        batch_size = len(protein_embs)
        protein_embs = protein_embs.to(self.device)
        
        all_scores = []
        
        # 分块计算（避免内存爆炸）
        for i in range(0, self.num_gos, self.chunk_size):
            go_chunk = self.go_embeddings[i:i+self.chunk_size]
            chunk_size_actual = len(go_chunk)
            
            # 扩展到 [batch, chunk_size, dim]
            protein_expanded = protein_embs.unsqueeze(1).expand(
                batch_size, chunk_size_actual, -1
            )
            go_expanded = go_chunk.unsqueeze(0).expand(
                batch_size, chunk_size_actual, -1
            )
            
            # 前向传播
            chunk_scores = self.model(protein_expanded, go_expanded)  # [batch, chunk_size]
            
            all_scores.append(chunk_scores.cpu() if not return_all_scores else chunk_scores)
        
        scores = torch.cat(all_scores, dim=1)  # [batch, num_gos]
        
        return scores
    
    @torch.no_grad()
    def evaluate_single_protein(
        self,
        protein_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        评估单个蛋白质（流式）
        
        参数:
            protein_emb: [esm_dim]
        
        返回:
            scores: [num_gos]
        """
        protein_emb = protein_emb.to(self.device).unsqueeze(0)  # [1, esm_dim]
        scores = self.evaluate_proteins(protein_emb, return_all_scores=False)
        return scores.squeeze(0)  # [num_gos]


# ============================================================================
# 流式指标计算器
# ============================================================================

class StreamingMetricsCalculator:
    """
    流式指标计算 - 增量更新，不存储全部预测
    
    内存使用: O(num_thresholds) 而非 O(num_predictions)
    """
    
    def __init__(self, num_thresholds: int = 100):
        self.thresholds = np.linspace(0, 1, num_thresholds)
        self.reset()
    
    def reset(self):
        """重置统计量"""
        self.tp = np.zeros(len(self.thresholds))
        self.fp = np.zeros(len(self.thresholds))
        self.fn = np.zeros(len(self.thresholds))
        self.tn = np.zeros(len(self.thresholds))
        
        # AUPR计算（采样）
        self.sampled_scores = []
        self.sampled_labels = []
        self.sample_rate = 0.1  # 只采样10%用于AUPR
    
    def update(self, scores: torch.Tensor, labels: torch.Tensor):
        """
        增量更新统计量
        
        参数:
            scores: [batch, num_gos] 预测分数
            labels: [batch, num_gos] 真实标签
        """
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # 对每个阈值更新TP/FP/FN/TN
        for i, thresh in enumerate(self.thresholds):
            preds = (scores >= thresh).astype(int)
            
            self.tp[i] += ((preds == 1) & (labels == 1)).sum()
            self.fp[i] += ((preds == 1) & (labels == 0)).sum()
            self.fn[i] += ((preds == 0) & (labels == 1)).sum()
            self.tn[i] += ((preds == 0) & (labels == 0)).sum()
        
        # 采样用于AUPR
        if random.random() < self.sample_rate:
            self.sampled_scores.append(scores.flatten())
            self.sampled_labels.append(labels.flatten())
    
    def compute(self) -> dict:
        """
        从累加的统计量计算最终指标
        
        返回:
            metrics字典
        """
        # 计算precision和recall
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Fmax
        fmax_idx = f1.argmax()
        fmax = f1[fmax_idx]
        best_thresh = self.thresholds[fmax_idx]
        best_precision = precision[fmax_idx]
        best_recall = recall[fmax_idx]
        
        # AUPR（从采样的数据计算）
        aupr = 0.0
        if self.sampled_scores:
            from sklearn.metrics import average_precision_score
            
            all_scores = np.concatenate(self.sampled_scores)
            all_labels = np.concatenate(self.sampled_labels)
            
            if len(np.unique(all_labels)) > 1:  # 需要正负样本都有
                aupr = average_precision_score(all_labels, all_scores)
        
        return {
            'fmax': float(fmax),
            'precision': float(best_precision),
            'recall': float(best_recall),
            'threshold': float(best_thresh),
            'aupr': float(aupr)
        }


# ============================================================================
# 工具函数
# ============================================================================

def create_labels_dict(protein_ids: np.ndarray, labels: np.ndarray, go_terms: List[str]) -> Dict[str, List[str]]:
    """
    从标签矩阵创建labels_dict
    
    参数:
        protein_ids: [num_proteins]
        labels: [num_proteins, num_gos]
        go_terms: GO terms列表
    
    返回:
        {protein_id: [positive_go_ids]}
    """
    labels_dict = {}
    
    for i, protein_id in enumerate(protein_ids):
        positive_indices = np.where(labels[i] == 1)[0]
        positive_gos = [go_terms[idx] for idx in positive_indices]
        labels_dict[protein_id] = positive_gos
    
    return labels_dict


def estimate_epoch_size(labels_dict: Dict[str, List[str]], negative_ratio: int = 3) -> int:
    """
    估算epoch大小
    
    参数:
        labels_dict: {protein_id: [positive_gos]}
        negative_ratio: 负样本比例
    
    返回:
        估算的样本对数
    """
    total_positives = sum(len(gos) for gos in labels_dict.values())
    total_negatives = total_positives * negative_ratio
    return total_positives + total_negatives
