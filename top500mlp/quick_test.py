"""
快速测试脚本：测试模型和数据加载是否正常工作
不需要GPU，不需要fair-esm
"""

import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model_three_head_mlp import ThreeHeadMLP
from dataset import load_cafa6_data, CAFA6Dataset, collate_fn_dynamic
from torch.utils.data import DataLoader
from metrics import compute_metrics_per_aspect, print_metrics


def test_model():
    """测试模型（不需要ESM2）"""
    print("="*70)
    print("测试1: 模型前向传播")
    print("="*70)

    # 创建模型（仅MLP部分）
    model = ThreeHeadMLP(
        esm_dim=320,
        hidden_dims=[512, 256],
        num_labels_per_aspect={'C': 500, 'F': 500, 'P': 500},
        dropout=0.3,
        use_shared_layer=True
    )

    # 模拟输入
    batch_size = 4
    fake_embeddings = torch.randn(batch_size, 320)

    print("\n模拟ESM2嵌入形状:", fake_embeddings.shape)

    # 测试前向传播 - 合并输出
    outputs_combined = model(fake_embeddings, return_separate=False)
    print("✓ 合并输出形状:", outputs_combined['logits'].shape)
    assert outputs_combined['logits'].shape == (batch_size, 1500), "输出形状错误"

    # 测试前向传播 - 分离输出
    outputs_separate = model(fake_embeddings, return_separate=True)
    print("✓ 分离输出形状:")
    for aspect in ['C', 'F', 'P']:
        print(f"    {aspect}: {outputs_separate[aspect].shape}")
        assert outputs_separate[aspect].shape == (batch_size, 500), f"{aspect}输出形状错误"

    # 参数统计
    print("\n✓ 参数统计:")
    params = model.get_num_params()
    for key, value in params.items():
        print(f"    {key}: {value:,}")

    print("\n✓ 模型测试通过！\n")
    return True


def test_data_loading():
    """测试数据加载"""
    print("="*70)
    print("测试2: 数据加载")
    print("="*70)

    try:
        # 加载数据
        data = load_cafa6_data("processed_data_v3")

        print("\n✓ 数据加载成功")
        print(f"  训练集: {len(data['train']['sequences']):,} 样本")
        print(f"  验证集: {len(data['val']['sequences']):,} 样本")
        print(f"  测试集: {len(data['test']['sequences']):,} 样本")
        print(f"  GO术语: {len(data['go_terms'])}")

        # 创建数据集
        train_dataset = CAFA6Dataset(
            sequences=data['train']['sequences'][:100],  # 只用前100个测试
            labels=data['train']['labels'][:100],
            protein_ids=data['train']['ids'][:100],
            go_aspects=data['go_aspects'],
            use_precomputed_embeddings=False
        )

        print(f"\n✓ 数据集创建成功: {len(train_dataset)} 样本")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_dynamic
        )

        print("✓ 数据加载器创建成功")

        # 测试一个批次
        batch = next(iter(train_loader))
        print("\n✓ 批次测试:")
        print(f"  序列数: {len(batch['sequences'])}")
        print(f"  标签形状: {batch['labels'].shape}")
        print(f"  C标签形状: {batch['labels_C'].shape}")
        print(f"  F标签形状: {batch['labels_F'].shape}")
        print(f"  P标签形状: {batch['labels_P'].shape}")
        print(f"  第一个序列长度: {len(batch['sequences'][0])}")
        print(f"  第一个序列前50字符: {batch['sequences'][0][:50]}")

        print("\n✓ 数据加载测试通过！\n")
        return True

    except FileNotFoundError as e:
        print(f"\n✗ 数据文件未找到: {e}")
        print("  请先运行: python save_processed_data_v3.py")
        return False


def test_metrics():
    """测试评估指标"""
    print("="*70)
    print("测试3: 评估指标")
    print("="*70)

    # 创建模拟数据
    np.random.seed(42)
    num_samples = 50
    num_labels = 1500

    # 模拟稀疏标签
    labels = np.zeros((num_samples, num_labels))
    for i in range(num_samples):
        num_pos = np.random.randint(1, 10)
        pos_indices = np.random.choice(num_labels, num_pos, replace=False)
        labels[i, pos_indices] = 1

    # 模拟预测
    predictions = labels + np.random.randn(num_samples, num_labels) * 0.2
    predictions = np.clip(predictions, 0, 1)

    print("\n生成模拟数据:")
    print(f"  样本数: {num_samples}")
    print(f"  标签数: {num_labels}")
    print(f"  平均每样本标签数: {labels.sum(axis=1).mean():.2f}")

    # aspect索引
    aspect_indices = {
        'C': np.arange(0, 500),
        'F': np.arange(500, 1000),
        'P': np.arange(1000, 1500)
    }

    # 计算指标
    print("\n计算评估指标...")
    metrics = compute_metrics_per_aspect(labels, predictions, aspect_indices)

    # 打印结果
    print_metrics(metrics)

    print("✓ 评估指标测试通过！\n")
    return True


def test_training_loop():
    """测试训练循环（小规模）"""
    print("="*70)
    print("测试4: 训练循环（迷你版本）")
    print("="*70)

    try:
        # 加载少量数据
        data = load_cafa6_data("processed_data_v3")

        # 只用前20个样本
        train_dataset = CAFA6Dataset(
            sequences=data['train']['sequences'][:20],
            labels=data['train']['labels'][:20],
            protein_ids=data['train']['ids'][:20],
            go_aspects=data['go_aspects'],
            use_precomputed_embeddings=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_dynamic
        )

        # 创建模型（仅MLP，用随机嵌入模拟ESM2）
        model = ThreeHeadMLP(
            esm_dim=320,
            hidden_dims=[128, 64],  # 小一点
            num_labels_per_aspect={'C': 500, 'F': 500, 'P': 500},
            dropout=0.1,
            use_shared_layer=True
        )

        device = torch.device('cpu')
        model = model.to(device)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        print("\n开始迷你训练（2个epoch）...")

        # 训练2个epoch
        for epoch in range(1, 3):
            model.train()
            total_loss = 0

            for batch in train_loader:
                # 模拟ESM2嵌入（随机）
                batch_size = len(batch['sequences'])
                fake_embeddings = torch.randn(batch_size, 320).to(device)

                labels_C = batch['labels_C'].to(device)
                labels_F = batch['labels_F'].to(device)
                labels_P = batch['labels_P'].to(device)

                # 前向传播
                outputs = model(fake_embeddings, return_separate=True)

                # 计算损失
                loss_C = criterion(outputs['C'], labels_C)
                loss_F = criterion(outputs['F'], labels_F)
                loss_P = criterion(outputs['P'], labels_P)
                loss = loss_C + loss_F + loss_P

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")

        print("\n✓ 训练循环测试通过！\n")
        return True

    except Exception as e:
        print(f"\n✗ 训练循环测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("CAFA6 三头MLP - 快速测试")
    print("="*70 + "\n")

    results = {}

    # 测试1: 模型
    results['model'] = test_model()

    # 测试2: 数据加载
    results['data'] = test_data_loading()

    # 测试3: 评估指标
    results['metrics'] = test_metrics()

    # 测试4: 训练循环
    results['training'] = test_training_loop()

    # 总结
    print("="*70)
    print("测试总结")
    print("="*70)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:12s}: {status}")

    all_passed = all(results.values())
    print("="*70)

    if all_passed:
        print("\n🎉 所有测试通过！可以开始正式训练了！")
        print("\n下一步:")
        print("  1. 确保安装了 fair-esm: pip install fair-esm")
        print("  2. 确保有GPU（可选，但强烈建议）")
        print("  3. 运行训练: python train.py")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")

    print()


if __name__ == "__main__":
    main()
