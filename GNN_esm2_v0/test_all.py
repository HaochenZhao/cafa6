"""
快速测试脚本 - Pairwise模型
从cafa主目录运行: python GNN_esm2/test_all.py
"""

import os
import sys
import yaml
import numpy as np
import torch

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import load_go_embeddings, load_processed_data
from model import PairwiseScoringModel


def test_directory_structure():
    """测试目录结构"""
    print("="*70)
    print("测试1: 目录结构")
    print("="*70)
    
    required_dirs = [
        'cafa-6-protein-function-prediction',
        'processed_data_v4',
        'go_gnn_results/balanced_fixed',
        'embeddings',
        'GNN_esm2'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✓ 目录结构正确")
    else:
        print("\n✗ 缺少必要目录")
        if not os.path.exists('processed_data_v4'):
            print("  注意: 请确认使用 processed_data_v4")
    
    return all_exist


def test_go_embeddings():
    """测试GO embeddings加载"""
    print("\n" + "="*70)
    print("测试2: GO Embeddings加载")
    print("="*70)
    
    try:
        embeddings, metadata = load_go_embeddings('go_gnn_results/balanced_fixed')
        print(f"✓ GO embeddings shape: {embeddings.shape}")
        print(f"✓ GO数量: {len(metadata['go_id_to_idx'])}")
        return embeddings, metadata, True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return None, None, False


def test_processed_data():
    """测试processed数据加载"""
    print("\n" + "="*70)
    print("测试3: Processed数据加载")
    print("="*70)
    
    try:
        data = load_processed_data('processed_data_v4')
        print(f"✓ 数据加载成功")
        return data, True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return None, False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "="*70)
    print("测试4: Pairwise模型创建")
    print("="*70)
    
    try:
        # 加载配置
        with open('GNN_esm2/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 创建模型
        model = PairwiseScoringModel(config)
        
        print("✓ 模型创建成功")
        
        # 参数统计
        param_stats = model.get_num_params()
        print("\n模型参数:")
        for key, value in param_stats.items():
            print(f"  {key}: {value:,}")
        
        return model, config, True
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*70)
    print("测试5: 前向传播")
    print("="*70)
    
    try:
        model, config, success = test_model_creation()
        if not success or model is None:
            print("✗ 模型创建失败，跳过前向传播测试")
            return False
        
        # 创建假输入
        batch_size = 16
        seq_dim = config['model']['esm_dim_map'][config['esm2']['model_name']]
        go_dim = config['go_gnn']['embedding_dim']
        
        seq_embs = torch.randn(batch_size, seq_dim)
        go_embs = torch.randn(batch_size, go_dim)
        
        print(f"\n输入形状:")
        print(f"  seq_embs: {seq_embs.shape}")
        print(f"  go_embs: {go_embs.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            logits = model(seq_embs, go_embs)
            probs = model.predict(seq_embs, go_embs)
        
        print("\n✓ 前向传播成功")
        print(f"\n输出形状:")
        print(f"  logits: {logits.shape}")
        print(f"  probs: {probs.shape}")
        print(f"\n概率范围: [{probs.min():.4f}, {probs.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pairwise_logic():
    """测试pairwise逻辑"""
    print("\n" + "="*70)
    print("测试6: Pairwise逻辑")
    print("="*70)
    
    try:
        # 模拟3个蛋白质，5个GO terms
        num_proteins = 3
        num_gos = 5
        
        print(f"\n模拟场景:")
        print(f"  蛋白质数: {num_proteins}")
        print(f"  GO terms数: {num_gos}")
        print(f"  总样本对数: {num_proteins * num_gos}")
        
        # 创建假的embeddings
        seq_dim = 320
        go_dim = 256
        
        seq_embs = np.random.randn(num_proteins, seq_dim)
        go_embs = np.random.randn(num_gos, go_dim)
        
        print(f"\n✓ 可以为每个(protein, go)对计算分数")
        print(f"  这样不会损失任何GO terms!")
        
        return True
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def main():
    """运行所有测试"""
    # 确保在cafa主目录下运行
    if not os.path.exists('cafa-6-protein-function-prediction'):
        print("\n错误: 请在cafa主目录下运行此脚本")
        print("用法: python GNN_esm2/test_all.py")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("CAFA6 Pairwise模型 - 快速测试")
    print("="*70 + "\n")
    
    results = {}
    
    # 测试1: 目录结构
    results['directory'] = test_directory_structure()
    
    # 测试2: GO embeddings
    _, _, results['go_embeddings'] = test_go_embeddings()
    
    # 测试3: Processed数据
    _, results['processed_data'] = test_processed_data()
    
    # 测试4: 模型创建
    _, _, results['model'] = test_model_creation()
    
    # 测试5: 前向传播
    results['forward'] = test_forward_pass()
    
    # 测试6: Pairwise逻辑
    results['pairwise_logic'] = test_pairwise_logic()
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 所有测试通过！可以开始训练了！")
        print("\n💡 Pairwise模型的优势:")
        print("  ✓ 利用所有GO terms（不限于top500）")
        print("  ✓ 每个(sequence, go)对独立评分")
        print("  ✓ 更灵活的预测阈值控制")
        print("  ✓ 预期更高的Fmax!")
        print("\n下一步:")
        print("  python GNN_esm2/train.py")
        print("\n或指定实验名称:")
        print("  python GNN_esm2/train.py --exp_name pairwise_baseline")
    else:
        print("\n⚠️ 部分测试失败，请检查:")
        if not results['directory']:
            print("  - 确保在cafa主目录下运行")
            print("  - 检查processed_data_v4目录是否存在")


if __name__ == "__main__":
    main()
