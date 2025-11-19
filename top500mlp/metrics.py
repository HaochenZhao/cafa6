"""
评估指标：Fmax, AUPR等
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from typing import Dict, Tuple


def compute_fmax(labels: np.ndarray, predictions: np.ndarray, beta: float = 1.0) -> Dict:
    """
    计算Fmax (Maximum F-score)

    参数:
        labels: 真实标签 [num_samples, num_labels]
        predictions: 预测分数 [num_samples, num_labels]
        beta: F-score的beta值，默认1.0 (F1-score)

    返回:
        包含fmax, precision, recall, threshold的字典
    """
    # 展平为1D数组
    labels_flat = labels.flatten()
    predictions_flat = predictions.flatten()

    # 计算precision-recall曲线
    precisions, recalls, thresholds = precision_recall_curve(labels_flat, predictions_flat)

    # 计算F-score
    beta_squared = beta ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        f_scores = (1 + beta_squared) * (precisions * recalls) / (beta_squared * precisions + recalls)
        f_scores = np.nan_to_num(f_scores)

    # 找到最大F-score
    max_idx = np.argmax(f_scores)
    fmax = f_scores[max_idx]
    best_precision = precisions[max_idx]
    best_recall = recalls[max_idx]
    best_threshold = thresholds[max_idx] if max_idx < len(thresholds) else 0.5

    return {
        'fmax': fmax,
        'precision': best_precision,
        'recall': best_recall,
        'threshold': best_threshold
    }


def compute_aupr(labels: np.ndarray, predictions: np.ndarray) -> float:
    """
    计算AUPR (Area Under Precision-Recall curve)

    参数:
        labels: 真实标签 [num_samples, num_labels]
        predictions: 预测分数 [num_samples, num_labels]

    返回:
        AUPR分数
    """
    labels_flat = labels.flatten()
    predictions_flat = predictions.flatten()

    return average_precision_score(labels_flat, predictions_flat)


def compute_metrics_per_aspect(
    labels: np.ndarray,
    predictions: np.ndarray,
    aspect_indices: Dict[str, np.ndarray]
) -> Dict[str, Dict]:
    """
    分别计算每个aspect的指标

    参数:
        labels: 真实标签 [num_samples, num_labels]
        predictions: 预测分数 [num_samples, num_labels]
        aspect_indices: {'C': indices, 'F': indices, 'P': indices}

    返回:
        每个aspect的指标字典
    """
    results = {}

    for aspect in ['C', 'F', 'P']:
        indices = aspect_indices[aspect]
        aspect_labels = labels[:, indices]
        aspect_preds = predictions[:, indices]

        # 计算Fmax和AUPR
        fmax_metrics = compute_fmax(aspect_labels, aspect_preds)
        aupr = compute_aupr(aspect_labels, aspect_preds)

        results[aspect] = {
            'fmax': fmax_metrics['fmax'],
            'precision': fmax_metrics['precision'],
            'recall': fmax_metrics['recall'],
            'threshold': fmax_metrics['threshold'],
            'aupr': aupr
        }

    # 计算总体指标
    overall_fmax = compute_fmax(labels, predictions)
    overall_aupr = compute_aupr(labels, predictions)

    results['overall'] = {
        'fmax': overall_fmax['fmax'],
        'precision': overall_fmax['precision'],
        'recall': overall_fmax['recall'],
        'threshold': overall_fmax['threshold'],
        'aupr': overall_aupr
    }

    return results


def compute_smin(labels: np.ndarray, predictions: np.ndarray, ia_weights: np.ndarray = None) -> float:
    """
    计算Smin (Minimum Semantic Distance)
    这是CAFA竞赛的官方评估指标之一

    参数:
        labels: 真实标签 [num_samples, num_labels]
        predictions: 预测分数 [num_samples, num_labels]
        ia_weights: IA权重 [num_labels]，如果为None则使用均匀权重

    返回:
        Smin分数
    """
    if ia_weights is None:
        ia_weights = np.ones(labels.shape[1])

    # 将预测转换为二值标签（需要阈值）
    # 这里使用多个阈值并选择最小的semantic distance
    thresholds = np.arange(0.01, 1.0, 0.01)
    min_dist = float('inf')

    for threshold in thresholds:
        pred_binary = (predictions >= threshold).astype(float)

        # 计算残差 (RU) 和未覆盖 (MI)
        # RU: 预测为正但实际为负
        # MI: 预测为负但实际为正
        ru = np.maximum(0, pred_binary - labels)  # remaining uncertainty
        mi = np.maximum(0, labels - pred_binary)  # misinformation

        # 加权求和
        ru_weighted = (ru * ia_weights).sum(axis=1)
        mi_weighted = (mi * ia_weights).sum(axis=1)

        # Semantic distance
        semantic_dist = np.sqrt(ru_weighted**2 + mi_weighted**2)

        # 平均距离
        avg_dist = semantic_dist.mean()

        if avg_dist < min_dist:
            min_dist = avg_dist

    return min_dist


def print_metrics(metrics: Dict[str, Dict], epoch: int = None, prefix: str = ""):
    """
    打印指标

    参数:
        metrics: 指标字典
        epoch: epoch编号
        prefix: 前缀字符串
    """
    if epoch is not None:
        print(f"\n{prefix}Epoch {epoch} 评估结果:")
    else:
        print(f"\n{prefix}评估结果:")

    print("="*70)

    # 打印各aspect的指标
    for aspect in ['C', 'F', 'P']:
        if aspect in metrics:
            m = metrics[aspect]
            print(f"{aspect}:")
            print(f"  Fmax:      {m['fmax']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, T={m['threshold']:.4f})")
            print(f"  AUPR:      {m['aupr']:.4f}")

    # 打印总体指标
    if 'overall' in metrics:
        m = metrics['overall']
        print(f"\nOverall:")
        print(f"  Fmax:      {m['fmax']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, T={m['threshold']:.4f})")
        print(f"  AUPR:      {m['aupr']:.4f}")

    if 'smin' in metrics:
        print(f"  Smin:      {metrics['smin']:.4f}")

    print("="*70)


if __name__ == "__main__":
    # 测试评估指标
    print("="*70)
    print("测试评估指标")
    print("="*70)

    # 创建模拟数据
    np.random.seed(42)
    num_samples = 100
    num_labels = 1500

    # 模拟稀疏标签
    labels = np.zeros((num_samples, num_labels))
    for i in range(num_samples):
        num_pos = np.random.randint(1, 10)
        pos_indices = np.random.choice(num_labels, num_pos, replace=False)
        labels[i, pos_indices] = 1

    # 模拟预测（添加一些噪声）
    predictions = labels + np.random.randn(num_samples, num_labels) * 0.2
    predictions = np.clip(predictions, 0, 1)

    # 创建aspect索引
    aspect_indices = {
        'C': np.arange(0, 500),
        'F': np.arange(500, 1000),
        'P': np.arange(1000, 1500)
    }

    # 计算指标
    print("\n计算指标...")
    metrics = compute_metrics_per_aspect(labels, predictions, aspect_indices)

    # 打印结果
    print_metrics(metrics)

    print("\n指标测试完成！")
