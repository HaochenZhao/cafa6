"""
评估指标模块
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def compute_fmax(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray = None) -> dict:
    """
    计算Fmax指标
    
    参数:
        labels: [num_samples] 真实标签
        scores: [num_samples] 预测分数
        thresholds: 阈值数组（可选）
    
    返回:
        metrics字典
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    best_f1 = 0
    best_thresh = 0
    best_precision = 0
    best_recall = 0
    
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_precision = precision
            best_recall = recall
    
    return {
        'fmax': float(best_f1),
        'precision': float(best_precision),
        'recall': float(best_recall),
        'threshold': float(best_thresh)
    }


def compute_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    计算AUPR (Area Under Precision-Recall Curve)
    
    参数:
        labels: [num_samples] 真实标签
        scores: [num_samples] 预测分数
    
    返回:
        AUPR分数
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    return average_precision_score(labels, scores)
