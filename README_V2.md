# CAFA6 数据处理工具包 V2 ⭐

**智能数据处理：过滤IA=0 + 智能层次化映射**

## 🎯 V2版本核心改进

### 1. ✅ 过滤IA=0的GO术语
- IA=0的GO在评估时不计分，应该排除
- 从40,122个GO → 22,078个有效GO
- 选Top-K更有意义

### 2. ✅ 智能层次化映射
- **V1策略**：添加所有祖先GO（噪声多）
- **V2策略**：只映射到最近的Top-K祖先（更精确）

### 3. ✅ 包含IA权重
- 保存每个GO术语的IA值
- 方便评估时使用CAFA官方指标

---

## 🚀 快速开始

### 一键运行
```bash
python quick_start_v2.py
```

### 在代码中使用
```python
from save_processed_data_v2 import save_processed_data, load_processed_data

# 首次：处理并保存数据
save_processed_data(
    output_dir="processed_data_v2",
    top_k_go=500,
    use_hierarchy=True  # 智能层次化
)

# 后续：快速读取（1-2秒）
data = load_processed_data("processed_data_v2")

# 使用数据
train_sequences = data['train']['sequences']
train_labels = data['train']['labels']
ia_weights = data['ia_weights']  # ⭐ V2新增
```

---

## 📊 智能层次化策略

### 场景1：GO直接在Top-K中
```
蛋白质A: GO:0005515 (Top-1)
         ↓
直接使用: GO:0005515 ✅
```

### 场景2：GO不在Top-K，但祖先在
```
蛋白质B: GO:9999999 (排名800，不在Top-500)
         ↓ 查找祖先
         GO:0008150 (排名50) ← 最近的Top-K祖先
         ↓ 停止向上搜索
         GO:0003674 (根节点) ← 不添加

结果：只映射到 GO:0008150 ✅
```

**关键**：使用BFS找到最近的Top-K祖先后立即停止，避免添加过于通用的祖先。

---

## 📈 效果对比

| 指标 | V1（简单层次化） | V2（智能层次化） |
|------|----------------|----------------|
| **过滤IA=0** | ❌ | ✅ |
| **层次化策略** | 添加所有祖先 | 只添加最近祖先 |
| **平均标签数** | ~8-10 | ~5-7 |
| **噪声** | 高 | 低 |
| **无标签样本** | ~2-3% | ~2-5% |
| **评估准确性** | 中 | 高 |

---

## 💾 数据格式

```python
data = {
    'train': {
        'sequences': array(['MKTF...', ...]),
        'labels': array([[0,1,0,...], ...]),  # (N, 500)
        'ids': array(['A0A0C5B5G6', ...])
    },
    'val': {...},
    'test': {...},
    'go_terms': ['GO:0005515', ...],  # 500个GO（IA>0）
    'ia_weights': array([0.201, 0.953, ...])  # ⭐ V2新增
}
```

---

## 🔧 配置选项

```python
save_processed_data(
    output_dir="processed_data_v2",
    top_k_go=500,           # Top-K GO术语数量
    val_ratio=0.1,          # 验证集比例
    use_hierarchy=True      # 是否使用智能层次化
)
```

**推荐配置**：
- **CAFA竞赛**: `top_k_go=500, use_hierarchy=True`
- **快速实验**: `top_k_go=100, use_hierarchy=False`
- **高覆盖率**: `top_k_go=1000, use_hierarchy=True`

---

## 📊 映射统计示例

```
GO映射统计:
  总标注数: 537,027
  直接匹配Top-K: 450,000 (83.8%)  ← 原本就在Top-500
  层次化映射: 50,000 (9.3%)       ← 通过层次化映射到Top-500
  丢失: 37,027 (6.9%)              ← 太罕见，无法映射
```

**解读**：
- 83.8%的标注直接在Top-500中
- 9.3%通过智能层次化映射到Top-500
- 只有6.9%丢失（非常罕见的GO）

---

## 📝 评估时使用IA权重

```python
import numpy as np

def weighted_f1_score(y_true, y_pred, ia_weights):
    """使用IA加权的F1分数（CAFA官方指标）"""

    tp = ((y_true == 1) & (y_pred == 1)).astype(float)
    fp = ((y_true == 0) & (y_pred == 1)).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).astype(float)

    # IA加权
    tp_weighted = (tp * ia_weights).sum()
    fp_weighted = (fp * ia_weights).sum()
    fn_weighted = (fn * ia_weights).sum()

    # 计算precision和recall
    precision = tp_weighted / (tp_weighted + fp_weighted) if (tp_weighted + fp_weighted) > 0 else 0
    recall = tp_weighted / (tp_weighted + fn_weighted) if (tp_weighted + fn_weighted) > 0 else 0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall

# 使用
data = load_processed_data("processed_data_v2")
f1, prec, rec = weighted_f1_score(
    y_true=data['val']['labels'],
    y_pred=predictions,
    ia_weights=data['ia_weights']
)

print(f"Weighted F1: {f1:.4f}")
```

---

## 📁 文件说明

### V2核心文件
| 文件 | 说明 |
|------|------|
| **save_processed_data_v2.py** | V2数据处理核心代码 ⭐ |
| **quick_start_v2.py** | V2快速开始脚本 |
| **v2改进说明.md** | 详细改进说明 |
| **README_V2.md** | 本文档 |

### V1文件（保留用于对比）
| 文件 | 说明 |
|------|------|
| save_processed_data.py | V1版本 |
| hierarchical_data_loader.py | V1层次化加载器 |
| simple_data_loader.py | 简单加载器 |

---

## 🎯 使用建议

### 什么时候用V2？
- ✅ CAFA竞赛提交
- ✅ 需要准确评估
- ✅ 生产环境
- ✅ 研究论文

### 什么时候用V1？
- 学习和理解层次化概念
- 快速原型实验
- 不在乎评估准确性

**推荐**：优先使用V2版本！

---

## 📊 完整工作流程

```
1. 读取数据
   ├── IA.tsv (过滤IA=0)
   ├── go-basic.obo (GO层次结构)
   ├── train_sequences.fasta
   └── train_terms.tsv
          ↓
2. 智能处理
   ├── 过滤IA=0的GO术语
   ├── 选择Top-K GO（从IA>0中）
   ├── 智能层次化映射
   └── 构建标签矩阵
          ↓
3. 保存数据
   ├── cafa6_data.npz (主数据+IA权重)
   ├── cafa6_metadata.pkl (元数据)
   └── data_statistics.txt (统计)
          ↓
4. 快速读取 (1-2秒)
          ↓
5. 训练和评估
```

---

## 🎓 理论基础

### 为什么要过滤IA=0？
- IA=0表示该GO在CAFA评估中不计分
- 包含它们会浪费计算资源
- 可能影响模型学习有效特征

### 为什么智能层次化更好？
- **V1问题**：添加所有祖先会引入太多通用GO（如根节点）
- **V2优势**：只添加最近祖先，保留合适的抽象层次
- **生物学意义**：最近祖先提供的信息最相关

### IA权重的作用？
- 评估时对不同GO赋予不同重要性
- 高IA（罕见GO）：预测正确得分高
- 低IA（常见GO）：预测正确得分低
- 避免模型只预测常见GO来"刷分"

---

## 📚 相关文档

- [v2改进说明.md](v2改进说明.md) - 详细改进说明
- [快速使用指南.md](快速使用指南.md) - V1使用指南
- [hierarchical_method_explanation.md](hierarchical_method_explanation.md) - 层次化原理

---

## 💡 常见问题

### Q: V2比V1慢吗？
A: 处理时稍慢（需要BFS查找祖先），但读取速度一样快（1-2秒）。

### Q: 无标签样本会增加吗？
A: 可能略微增加（2-3% → 2-5%），但这是合理的（过滤掉了不该有的GO）。

### Q: 必须用IA权重评估吗？
A: 竞赛提交时必须用，训练时可选。

### Q: 如何选择Top-K数量？
A:
- Top-500：平衡性能和覆盖率
- Top-1000：更高覆盖率，但计算量增加
- Top-100：快速实验

---

## 🎉 总结

**V2版本是更成熟、更符合CAFA标准的数据处理方案！**

核心改进：
1. ✅ 过滤IA=0（更准确）
2. ✅ 智能层次化（更精确）
3. ✅ 包含IA权重（便于评估）
4. ✅ 详细统计（可追溯）

**推荐所有CAFA6参赛者使用V2版本！**

---

**快速开始** → 运行 `python quick_start_v2.py`
