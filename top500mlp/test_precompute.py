"""
测试预计算嵌入流程（不需要实际运行ESM2）
"""

import numpy as np
import torch
from model_three_head_mlp import ThreeHeadMLP


def create_fake_embeddings(data_dir: str = "processed_data_v3", embed_dim: int = 320):
    """
    创建假的嵌入数据用于测试（不需要ESM2）
    """
    print("="*70)
    print("创建假嵌入用于测试")
    print("="*70)

    # 加载数据获取样本数量
    import os
    npz_file = os.path.join(data_dir, "cafa6_data.npz")
    data = np.load(npz_file, allow_pickle=True)

    num_train = len(data['train_sequences'])
    num_val = len(data['val_sequences'])
    num_test = len(data['test_sequences'])

    print(f"\n数据集大小:")
    print(f"  训练集: {num_train:,}")
    print(f"  验证集: {num_val:,}")
    print(f"  测试集: {num_test:,}")

    # 创建随机嵌入
    print(f"\n创建随机嵌入 (维度={embed_dim})...")
    train_embeddings = np.random.randn(num_train, embed_dim).astype(np.float32)
    val_embeddings = np.random.randn(num_val, embed_dim).astype(np.float32)
    test_embeddings = np.random.randn(num_test, embed_dim).astype(np.float32)

    # 标准化（模拟真实嵌入）
    train_embeddings = (train_embeddings - train_embeddings.mean()) / train_embeddings.std()
    val_embeddings = (val_embeddings - val_embeddings.mean()) / val_embeddings.std()
    test_embeddings = (test_embeddings - test_embeddings.mean()) / test_embeddings.std()

    print("✓ 嵌入创建完成")
    print(f"  训练集: {train_embeddings.shape}")
    print(f"  验证集: {val_embeddings.shape}")
    print(f"  测试集: {test_embeddings.shape}")

    # 保存
    output_file = "embeddings/fake_embeddings_for_test.npz"
    os.makedirs("embeddings", exist_ok=True)

    print(f"\n保存到: {output_file}")
    np.savez_compressed(
        output_file,
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings
    )

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ 保存完成")
    print(f"  文件大小: {file_size_mb:.2f} MB")

    return output_file


def test_precomputed_training():
    """
    测试使用预计算嵌入的训练流程
    """
    print("\n" + "="*70)
    print("测试预计算嵌入训练流程")
    print("="*70)

    # 创建假嵌入
    embeddings_file = create_fake_embeddings()

    # 加载嵌入
    print("\n" + "="*70)
    print("加载嵌入")
    print("="*70)

    emb_data = np.load(embeddings_file)
    train_embeddings = emb_data['train_embeddings']
    val_embeddings = emb_data['val_embeddings']

    print(f"✓ 嵌入加载完成")
    print(f"  训练集嵌入: {train_embeddings.shape}")
    print(f"  验证集嵌入: {val_embeddings.shape}")

    # 创建数据加载器
    print("\n" + "="*70)
    print("创建数据加载器")
    print("="*70)

    # 只取前100个样本用于快速测试
    # 需要手动创建小数据集
    import os
    from dataset import CAFA6Dataset, collate_fn_precomputed
    from torch.utils.data import DataLoader

    # 加载完整数据
    npz_file = os.path.join("processed_data_v3", "cafa6_data.npz")
    data = np.load(npz_file, allow_pickle=True)

    # 只取前100个样本
    num_test_samples = 100
    train_dataset = CAFA6Dataset(
        sequences=data['train_sequences'][:num_test_samples],
        labels=data['train_labels'][:num_test_samples],
        protein_ids=data['train_ids'][:num_test_samples],
        go_aspects=data['go_aspects'],
        use_precomputed_embeddings=True,
        embeddings=train_embeddings[:num_test_samples]
    )

    val_dataset = CAFA6Dataset(
        sequences=data['val_sequences'][:num_test_samples],
        labels=data['val_labels'][:num_test_samples],
        protein_ids=data['val_ids'][:num_test_samples],
        go_aspects=data['go_aspects'],
        use_precomputed_embeddings=True,
        embeddings=val_embeddings[:num_test_samples]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_precomputed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_precomputed
    )

    # 创建元数据
    metadata = {
        'go_terms': data['go_terms'].tolist(),
        'ia_weights': data['ia_weights'],
        'go_aspects': data['go_aspects'],
        'num_labels': len(data['go_terms']),
        'num_labels_per_aspect': {
            'C': (data['go_aspects'] == 'C').sum(),
            'F': (data['go_aspects'] == 'F').sum(),
            'P': (data['go_aspects'] == 'P').sum()
        }
    }

    print("✓ 数据加载器创建成功")

    # 创建模型
    print("\n" + "="*70)
    print("创建模型")
    print("="*70)

    esm_dim = train_embeddings.shape[1]
    model = ThreeHeadMLP(
        esm_dim=esm_dim,
        hidden_dims=[128, 64],  # 小一点用于测试
        num_labels_per_aspect=metadata['num_labels_per_aspect'],
        dropout=0.1,
        use_shared_layer=True
    )

    print("✓ 模型创建成功")
    param_stats = model.get_num_params()
    for key, value in param_stats.items():
        print(f"  {key}: {value:,}")

    # 测试前向传播
    print("\n" + "="*70)
    print("测试前向传播")
    print("="*70)

    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    batch = next(iter(train_loader))
    embeddings = batch['embeddings'].to(device)
    labels_C = batch['labels_C'].to(device)
    labels_F = batch['labels_F'].to(device)
    labels_P = batch['labels_P'].to(device)

    print(f"批次大小: {embeddings.shape[0]}")
    print(f"嵌入维度: {embeddings.shape[1]}")

    with torch.no_grad():
        outputs = model(embeddings, return_separate=True)

    print("\n✓ 前向传播成功")
    print(f"  C输出: {outputs['C'].shape}")
    print(f"  F输出: {outputs['F'].shape}")
    print(f"  P输出: {outputs['P'].shape}")

    # 测试损失计算
    print("\n" + "="*70)
    print("测试损失计算")
    print("="*70)

    criterion = torch.nn.BCEWithLogitsLoss()

    loss_C = criterion(outputs['C'], labels_C)
    loss_F = criterion(outputs['F'], labels_F)
    loss_P = criterion(outputs['P'], labels_P)
    total_loss = loss_C + loss_F + loss_P

    print(f"✓ 损失计算成功")
    print(f"  Loss C: {loss_C.item():.4f}")
    print(f"  Loss F: {loss_F.item():.4f}")
    print(f"  Loss P: {loss_P.item():.4f}")
    print(f"  Total: {total_loss.item():.4f}")

    # 测试训练步骤
    print("\n" + "="*70)
    print("测试训练步骤（2个batch）")
    print("="*70)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i, batch in enumerate(train_loader):
        if i >= 2:  # 只测试2个batch
            break

        embeddings = batch['embeddings'].to(device)
        labels_C = batch['labels_C'].to(device)
        labels_F = batch['labels_F'].to(device)
        labels_P = batch['labels_P'].to(device)

        # 前向传播
        outputs = model(embeddings, return_separate=True)

        # 计算损失
        loss_C = criterion(outputs['C'], labels_C)
        loss_F = criterion(outputs['F'], labels_F)
        loss_P = criterion(outputs['P'], labels_P)
        loss = loss_C + loss_F + loss_P

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Batch {i+1}: Loss = {loss.item():.4f}")

    print("\n✓ 训练步骤测试成功")

    print("\n" + "="*70)
    print("所有测试通过！")
    print("="*70)
    print("\n下一步:")
    print("  1. 运行真实的预计算: python precompute_embeddings.py")
    print("  2. 使用预计算嵌入训练: python train.py --use_precomputed_embeddings \\")
    print("       --embeddings_file embeddings/embeddings_t6_8M_mean.npz")


if __name__ == "__main__":
    test_precomputed_training()
