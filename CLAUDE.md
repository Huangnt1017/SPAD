# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Common Commands

### Training

```powershell
# 设置环境变量 (VSCode 打开项目时已自动设置)
$env:PYTHONPATH = "D:\PYproject\SPAD"
$python = "D:\anaconda3\envs\torchnew\python.exe"

# 训练 baseline (示例: PointNet++)
& $python scripts/train.py --model pointnet2 --batch-size 32 --epochs 100

# 训练自研模型 (GCN 变体)
& $python scripts/train.py --model graph_residual_gcn --batch-size 32 --epochs 100

# 从 checkpoint 继续训练
& $python scripts/train.py --model dgcnn --resume checkpoints/CLS/dgcnn_last.pth
```

### Testing / Evaluation

```powershell
# 测试集评估 (分类 + 3D Box AP)
& $python scripts/test.py --checkpoint checkpoints/CLS/dgcnn_best.pth

# 单样本推理 + 可视化
& $python scripts/test1.py --checkpoint checkpoints/CLS/dgcnn_best.pth
```

### Available Models

`dgcnn`, `pointnet`, `pointnet2`, `pointnet2msg`, `pointbert`, `pointmae`, `pointrwkv`, `pointtransformer`, `pointtransv2`, `pointtransv3`, `pointmlp`, `pointmlpelite`, `spt`, `upp`, `graph_residual`, `graph_residual_gcn`, `3detr`

### Key Training Parameters

| 参数 | 默认值 | 说明 |
|:-----|:------:|:-----|
| `--batch-size` | 32 | 批大小 |
| `--epochs` | 100 | 训练轮数 |
| `--num-points` | 1024 | 每样本固定点数 |
| `--lr` / `--min-lr` | 1e-3 / 1e-5 | 余弦退火学习率 |
| `--cls-loss-weight` | 1.0 | 分类 loss 权重 |
| `--box-loss-weight` | 10.0 | Box loss 权重 |
| `--no-auto-balance` | 默认 | 禁用 Kendall 自适应权重 |
| `--seed` | 42 | 随机种子 |
| `--augment-train` | 默认开启 | 训练集数据增强 |

---

## Project Architecture

```
SPAD/
├── baseline/          # 13 个对比 baseline 模型
├── model/             # 自研模型 (graph_residual.py, graph_res_GCN.py)
├── scripts/
│   ├── train.py       # 统一训练入口
│   ├── test.py        # 测试集评估 (分类 + 3D Box AP)
│   └── test1.py       # 单样本推理 + 可视化
├── utils/
│   ├── heads.py       # 统一分类头 + 中心点回归头 (所有模型必须使用)
│   ├── loss.py        # Soft-histogram depth loss + 框几何工具
│   ├── data.py        # DataLoader + 归一化
│   ├── data_augment.py    # 数据增强
│   ├── checkpoint.py  # checkpoint 保存/加载
│   └── transformer_blocks.py  # ViT 原语集中定义
├── SNN_based_method/  # 脉冲神经网络变体 (独立子项目)
├── logs/              # 训练/测试日志输出，按 CLS/SNN 分组
└── checkpoints/       # checkpoint 输出，按 CLS/SNN 分组
```

---

## Mandatory Conventions (所有模型必须遵守)

### 1. Box Head: 直接回归 (Center-Only)

```python
box_preds = self.box_head(f_pooled)  # (B, 3) 中心点 [cx, cy, cz]
```

**禁止** centroid-offset (`pred = centroid + offset`)。例外: 3DETR (query-based 架构)。

### 2. 统一头部架构 (utils/heads.py)

```python
from utils.heads import build_standard_cls_head, build_standard_box_head

# 分类头: 3 层 MLP (pooled → 256 → 128 → num_classes)
self.cls_head = build_standard_cls_head(pooled_dim, num_classes, dropout=0.3)

# 中心点回归头: 3 层 MLP (pooled → 256 → 128 → 3)
self.box_head = build_standard_box_head(pooled_dim, box_dim=3, dropout=0.3)
```

**标准结构:**
- **cls_head**: `Linear(pooled→256) → BN1d → LeakyReLU(0.2) → Dropout(0.3) → Linear(256→128) → BN1d → LeakyReLU(0.2) → Dropout(0.3) → Linear(128→num_classes)`
- **box_head**: 同上，最后一层 `Linear(128→3)`
- **唯一变量仅为 backbone 架构**

**例外:**
- **SPT**: Conv1d + SpikeNode，维度对齐 (512→256→128→out)
- **3DETR**: per-query GenericMLP

### 3. Loss: Soft-Histogram Depth Loss

```python
criterion = PointCloudMultiTaskLoss(
    cls_weight=1.0,
    box_weight=10.0,
    auto_balance=False,  # 固定权重
)
```

公式: $\mathcal{L}_{depth} = \sum_d \sum_k w_k \cdot (\hat{c}_d - (c_d^{gt} + k \cdot \delta_d))^2$

参数: `sh_k=2` (窗口半径), `sh_sigma=1.5` (高斯宽度)

---

## Code Standards

### 注释 (必须中文)

1. **形状变换前**必须注释: `reshape / view / permute / transpose / flatten / squeeze / unsqueeze / cat / stack`
   - 写明形状走向: `# (B, N, 4) → (B, 4, N)`
2. **公共函数/类**必须有 docstring (`Args / Returns / Raises`)
3. **张量重的代码**docstring 里写关键 shape
4. **临时逻辑**用 `TODO(name) YYYY-MM-DD: 动作` 标记

### 命名 (PEP 8)

- 模块/函数/变量: `snake_case`
- 公共类: `CapWords`
- 常量: `UPPER_SNAKE_CASE`
- 布尔: `is_xxx / has_xxx / enable_xxx`
- **禁止**模糊名: `data / result / temp / value / item / dict / list / d / r / t / tmp`

### 类型 & 异常

- 公共函数加 type hints
- 入口验证 shape/dtype/range，错误信息说出实际值与期望值
- 不静默吞异常

---

## Point Cloud Pipeline

数据契约:
```
raw → parsed → dataset 样本 → batch tensor → model 输入 → loss 目标 → 评估输出
```

训练失败排查顺序:
1. 数据解析 / batching
2. 目标编码 (bbox 归一化 / 中心点提取)
3. loss 兼容性 (模型输出 shape vs loss 期望)
4. **最后**才动模型架构

---

## `__main__` Memory Test Convention

baseline 文件含 `if __name__ == "__main__":` 时:
- 打印 GPU 型号与显存
- 扫 `[4, 8, 16, 32]` batch size，每个: 重建模型 → CUDA → forward+backward → 打印峰值显存
- 捕获 `torch.cuda.OutOfMemoryError`，报失败 batch 后停止
- 支持 tuple / dict / tensor 三种返回构造临时 loss

---

## Current Model Status (2026-06-02)

| 模型 | Box Head | Head 架构 | Loss | 权重 |
|:-----|:--------:|:---------:|:----:|:----:|
| 13 个 baseline | 直接回归 [B,3] | 统一 MLP | Soft-histogram | 固定 |
| graph_residual.py | 直接回归 [B,3] | 统一 MLP | Soft-histogram | 固定 |
| graph_res_GCN.py | 直接回归 [B,3] | 统一 MLP | Soft-histogram | 固定 |
| SPT | 直接回归 [B,3] | Conv1d + SpikeNode | Soft-histogram | 固定 |
| 3DETR | centroid-offset [B,6] | per-query GenericMLP | Soft-histogram | 固定 |

---

## Paper Experiment Guidelines

1. **统一训练条件**: 相同 loss / 数据划分 (seed=42) / 超参
2. **评估指标**: Top-1/Top-3 Accuracy + 3D Box IoU
3. **对比维度**: backbone 是唯一变量
4. **消融实验**: graph_residual vs graph_res_GCN (DGCNN EdgeConv vs SAGEConv)
