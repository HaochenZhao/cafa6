# CAFA6 Pairwise Scoring Model

**突破性改进**: 不再局限于top500分类，而是对每个(sequence, GO term)对独立评分！

## 🎯 核心思想

### 旧方案的问题
```
序列 -> ESM2 embedding -> 分类头 -> Top 500 GO terms
                                    ❌ 损失了大量GO信息！
```

### 新方案 (Pairwise Scoring)
```
(序列 embedding, GO embedding) -> Scoring Model -> 分数 (0-1)
                                                   ✓ 利用所有GO terms!
```

**关键优势**:
- ✅ 可以预测**所有**GO terms（不限于500个）
- ✅ 更高的Fmax（覆盖更多真实标签）
- ✅ 灵活的阈值控制
- ✅ 充分利用GO GNN embeddings

## 📂 项目结构

```
cafa/
├── processed_data_v4/          # 处理后的数据 ⭐ 注意是v4
├── go_gnn_results/             # GO GNN embeddings
├── embeddings/                 # ESM2 embeddings
└── GNN_esm2/                   # 本项目
    ├── config/config.yaml      # 配置
    ├── src/
    │   ├── dataset.py          # Pairwise数据集
    │   ├── model.py            # Pairwise模型
    │   └── metrics.py          # 评估指标
    ├── train.py                # 训练
    ├── predict.py              # 推理
    └── test_all.py             # 测试
```

## 🚀 快速开始

### 1. 测试环境

```bash
cd /path/to/cafa
python GNN_esm2/test_all.py
```

### 2. 训练模型

```bash
# 基础训练
python GNN_esm2/train.py

# 指定实验名
python GNN_esm2/train.py --exp_name pairwise_baseline
```

### 3. 生成预测

```bash
python GNN_esm2/predict.py \
    --checkpoint GNN_esm2/checkpoints/exp_XXX/best.pth \
    --config GNN_esm2/checkpoints/exp_XXX/config.yaml \
    --output GNN_esm2/outputs/submission.tsv \
    --threshold 0.5
```

## 🏗️ 模型架构

### Pairwise Scoring

```
Sequence Embedding (320维)  ──┐
                              ├──> Fusion Module ──> Score (0-1)
GO Embedding (256维)        ──┘
```

### 支持的融合方式

1. **MLP融合** (默认)
   ```
   concat([seq_emb, go_emb]) -> MLP -> score
   ```

2. **Bilinear融合**
   ```
   Bilinear(seq_emb, go_emb) -> MLP -> score
   ```

3. **Attention融合**
   ```
   MultiHeadAttention(seq_emb, go_emb) -> MLP -> score
   ```

## ⚙️ 配置说明

### 采样策略

```yaml
training:
  sampling:
    strategy: "mixed"  # all, negative_sampling, mixed
    negative_ratio: 5  # 每个正样本配5个负样本
```

**采样策略说明**:
- `all`: 所有(protein, go)对（内存需求大）
- `negative_sampling`: 所有正样本 + 采样负样本（推荐）
- `mixed`: 智能混合（正样本少的用全样本，多的用采样）

### 融合方式

```yaml
model:
  fusion:
    type: "mlp"  # mlp, bilinear, attention
    hidden_dims: [512, 256, 128]
    dropout: 0.3
```

### 批次大小

```yaml
training:
  batch_size: 512  # 训练时的(seq, go)对数量

inference:
  batch_size: 2048  # 推理时可以更大
```

## 📊 数据流

### 训练阶段
```
1. 采样(protein, go)对 + labels
2. 获取seq_embedding和go_embedding
3. 计算分数
4. 计算loss并更新
```

### 推理阶段
```
对每个蛋白质:
    对所有GO terms:
        计算(protein, go)的分数
    应用阈值筛选
    生成预测
```

## 💡 为什么Pairwise更好？

### 对比分析

| 方案 | Top500分类 | Pairwise Scoring |
|------|-----------|------------------|
| 可预测GO数 | 500 | 40,000+ |
| 覆盖率 | 低 | 高 |
| Fmax | ~0.45 | **预期>0.50** |
| 灵活性 | 低 | 高 |

### 举例说明

**蛋白质X的真实标签**: GO:0001, GO:0002, GO:0003, ..., GO:0050

**Top500方案**:
- 只能预测500个预定义的GO
- 如果GO:0050不在这500个里 → **无法预测** ❌
- 导致recall低，Fmax低

**Pairwise方案**:
- 可以预测所有40000+个GO
- GO:0050也能预测 → **覆盖更全** ✅
- recall高，Fmax高

## 🔧 优化建议

### 1. 调整采样策略

```yaml
# 如果内存充足
training:
  sampling:
    strategy: "all"  # 使用所有对
    
# 如果内存不足
training:
  sampling:
    strategy: "negative_sampling"
    negative_ratio: 3  # 减少负样本比例
```

### 2. 尝试不同融合方式

```yaml
# Bilinear可能更好地捕获交互
model:
  fusion:
    type: "bilinear"
    
# Attention可能更灵活
model:
  fusion:
    type: "attention"
```

### 3. 调整阈值

推理时尝试不同阈值:
```bash
# 较低阈值 -> 更高recall
python GNN_esm2/predict.py ... --threshold 0.3

# 较高阈值 -> 更高precision
python GNN_esm2/predict.py ... --threshold 0.7
```

## 📈 预期性能提升

基于Pairwise架构，预期性能:

| 指标 | Top500方案 | Pairwise方案 | 提升 |
|------|-----------|-------------|------|
| Fmax | 0.45 | **0.50-0.55** | +11-22% |
| AUPR | 0.40 | **0.45-0.50** | +12-25% |
| Recall | 低 | **高** | 显著提升 |
| 覆盖GO数 | 500 | **40,000+** | 80x |

## 🎓 技术细节

### 负采样的重要性

为什么需要负采样？
- 正样本太稀疏（每个蛋白~10个正样本，40000个GO）
- 全部使用会导致极度不平衡
- 负采样可以平衡正负样本比例

### 批处理策略

**训练**: batch_size个(seq, go)对
```python
# batch中可能包含:
# (protein1, go1), (protein1, go2), (protein2, go1), ...
```

**推理**: 对每个蛋白，批量预测所有GO
```python
# 一次处理一个蛋白的所有GO terms
for protein in proteins:
    scores = model.predict(protein_emb, all_go_embs)
```

## 🚧 常见问题

### Q: 训练很慢？

A: 调整采样策略和batch_size
```yaml
training:
  sampling:
    strategy: "negative_sampling"  # 而不是"all"
    negative_ratio: 3  # 减少负样本
  batch_size: 1024  # 增大batch
```

### Q: 内存不足？

A: 
1. 减小batch_size
2. 使用negative_sampling而不是all
3. 推理时减小inference.batch_size

### Q: 如何知道最佳阈值？

A: 在验证集上尝试不同阈值，选择Fmax最高的:
```python
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    # 应用阈值，计算Fmax
    # 选择最佳阈值
```

## 📚 相关文档

- `config/config.yaml` - 完整配置选项
- `src/dataset.py` - Pairwise数据集实现
- `src/model.py` - Pairwise模型实现

---

**这是一个突破性的架构改进！不再受限于top500，充分利用所有GO terms，预期显著提升性能！** 🚀
