# SPAD 项目编码规范

本项目所有 Python 代码改动**必须**遵守以下规则。完整 skill 定义见 `.claude/skills/`。

---

## 项目架构概览

```
SPAD/
├── baseline/          # 13 个对比 baseline 模型 (DGCNN, PointNet++, PointMLP, ...)
├── model/             # 自研模型
│   ├── graph_residual.py      # 原版: DGCNN Conv2d EdgeConv + 全局注意力
│   ├── graph_res_GCN.py       # PyG 变体: SAGEConv + SE gate
│   └── readme.md              # 架构蓝图 + 优化路线
├── scripts/
│   ├── train.py       # 统一训练入口
│   ├── test.py        # 测试集评估 (分类 + 3D Box AP)
│   ├── test1.py       # 单样本推理 + 可视化
│   └── plot_train_log.py  # 训练曲线绘制
├── utils/
│   ├── loss.py        # 多任务损失 + 框几何工具
│   ├── heads.py       # 统一分类头 + 中心点回归头构建
│   ├── data.py        # DataLoader + 归一化
│   ├── data_augment.py    # 数据增强
│   └── checkpoint.py  # checkpoint 保存/加载
└── CLAUDE.md          # 本文件
```

---

## 统一约定 (所有模型必须遵守)

### Box Head: 直接回归 (Center-Only)

所有模型 (baseline + 自研) 的 box_head **必须**使用直接回归:

```python
box_preds = self.box_head(f_pooled)  # (B, 3) 中心点 [cx, cy, cz]
```

**禁止** centroid-offset (`pred = centroid + offset`)，原因:
- 12 个 baseline 全部使用直接回归
- 论文要求 backbone 是唯一变量，box_head 策略必须统一
- 3DETR 是例外 (centroid-offset + 6 维输出)，但其架构本身就是 query-based

### 统一头部架构 (utils/heads.py)

所有模型 (baseline + 自研) 的 cls_head 和 box_head **必须**使用 `utils/heads.py` 中的统一构建函数:

```python
from utils.heads import build_standard_cls_head, build_standard_box_head

# 分类头: 3 层 MLP (pooled → 256 → 128 → num_classes)
self.cls_head = build_standard_cls_head(pooled_dim, num_classes, dropout=0.3)

# 中心点回归头: 3 层 MLP (pooled → 256 → 128 → 3)
self.box_head = build_standard_box_head(pooled_dim, box_dim=3, dropout=0.3)
```

**统一标准:**
- **cls_head**: `Linear(pooled→256, bias=False) → BN1d → LeakyReLU(0.2) → Dropout → Linear(256→128) → BN1d → LeakyReLU(0.2) → Dropout → Linear(128→num_classes)`
- **box_head**: `Linear(pooled→256, bias=False) → BN1d → LeakyReLU(0.2) → Dropout → Linear(256→128) → BN1d → LeakyReLU(0.2) → Linear(128→3)`
- 所有模型使用相同的中间维度 (256 → 128)、激活函数 (LeakyReLU 0.2)、Dropout 率 (0.3)
- **唯一变量仅为 backbone 架构**，确保论文对比公平

**特殊架构例外:**
- **SPT** (脉冲神经网络): 使用 Conv1d + SpikeNode 替代 Linear + LeakyReLU，但中间维度对齐标准 (512→256→128→out)
- **3DETR** (query-based): per-query GenericMLP 头，架构本质上无法适配统一头

### Loss: Soft-Histogram Depth Loss

```python
criterion = PointCloudMultiTaskLoss(
    cls_weight=1.0,
    box_weight=1.0,
    auto_balance=False,  # 固定权重, 不用 Kendall
)
```

公式: $\mathcal{L}_{depth} = \sum_d \sum_k w_k \cdot (\hat{c}_d - (c_d^{gt} + k \cdot \delta_d))^2$

- 直接建模 SPAD 物理过程 (时间 bin 量化 + 高斯脉冲展宽)
- 替代旧版 Log-Cauchy (数学近似，物理意义弱)
- 参数: `sh_k=2` (窗口半径), `sh_sigma=1.5` (高斯宽度)

### Loss 权重: 固定 λ_cls · L_cls + λ_depth · L_depth

默认 `cls_weight=1.0`, `box_weight=1.0`, `auto_balance=False`。

**不推荐** Kendall 自适应权重，原因:
- 训练 loss 可能变负 (log-variance 项无约束增长)
- 固定权重更稳定，论文对比更公平

---

## 注释规范

### 语言
- **修改、新增或修改代码注释一律用清晰的技术中文**。
- 不使用模糊词或翻译腔；术语保留英文（如 `forward`、`tensor`、`logits`、`bbox`）。

### 义务（不可省略）
1. **形状变换前**必须注释：`reshape / view / permute / transpose / flatten / squeeze / unsqueeze / cat / stack`
   - 写明形状走向，如 `# (B, N, 4) → (B, 4, N)`。
2. **公共函数/类**必须有 docstring，包含 `Args / Returns / Raises`（仅当相关时）。
3. **张量重的代码**docstring 里写关键 shape，例如 `pred: [B, 3] 中心点 / gt: [B, 6] 角点`。
4. **数据流走向**——尤其是数组在管道中形状变化的关键节点——必须注释。
5. **非显然的常量**说明其来源/含义。
6. **临时逻辑**用 `TODO(name) YYYY-MM-DD: 下一步动作` 标记。
7. **不要写"什么"型注释**（代码已说明）；写"为什么/怎么/边界条件"。

---

## 命名（PEP 8）

- 模块/函数/方法/变量/属性: `snake_case`
- 公共类: `CapWords`；私有类: `_CapWords`
- 模块级常量: `UPPER_SNAKE_CASE`
- 布尔: `is_xxx / has_xxx / enable_xxx`
- **禁止**模糊名: `data / result / temp / value / item / dict / list / d / r / t / tmp`
  - 小作用域单字母可: `i j k x y z`
- 用语义名: `points / features / labels / logits / valid_mask / sample_indices / pred_centers`

---

## 类型 & 异常

- 新加的公共函数和重要内部辅助加 type hints。
- 入口/边界先验证 shape/dtype/range/必需键，错误信息要可执行（说出实际值与期望值）。
- 不静默吞异常；要么带上下文处理，要么再抛。

---

## 改动边界

- 保留现有公共接口行为；只在任务明确要求时改。
- 最小局部改动 > 大范围重写；不顺手重构无关组件。
- 重复逻辑提取为可复用辅助；优先纯函数。
- 当改动代码逻辑后及时更新注释。

---

## Point Cloud 管道工作流

改动前先理清数据契约：

```
raw → parsed → dataset 样本 → batch tensor → model 输入 → loss 目标 → 评估输出
```

训练失败时排查顺序：
1. 数据解析 / batching
2. 目标编码 (bbox 归一化 / 中心点提取)
3. loss 兼容性 (模型输出 shape vs loss 期望)
4. **最后**才动模型架构

增强若改几何，确认 label 同步对齐。

---

## `__main__` 显存测试规约

当 baseline 文件含 `if __name__ == "__main__":` 时：
- 打印 GPU 型号与显存。
- 批量扫 `[4, 8, 16, 32]`，对每个 size：重建模型 → 上 CUDA → forward+backward → 打印峰值显存。
- 捕获 `torch.cuda.OutOfMemoryError`，报失败 batch 后停止扫描。
- 支持 tuple / dict / tensor 三种返回构造临时 loss。

---

## 长文件快速排错

1. 先扫 imports / 类与函数签名 / 配置常量 / 装饰器 / 继承 / TODO / debug print。
2. 给出≤3 个高嫌疑区。
3. 只读这几个窄窗，定位根因。
4. 给出聚焦修改，不做大改。

---

## 完成定义

- 端到端 sanity 跑通，或明确指出剩余阻塞。
- 注释+docstring 解释了意图、数据流、shape 期望。
- 命名/类型/异常处理符合上面规则。
- 无关文件无关行为不动。

---

## 当前模型状态 (截至 2026-06-02)

| 模型 | Box Head | cls_head / box_head 架构 | Loss | 权重策略 |
|:-----|:--------:|:------------------------:|:----:|:--------:|
| 12 个 baseline (DGCNN, PointNet++, ...) | 直接回归 [B,3] | 统一 MLP (utils/heads.py) | Soft-histogram | 固定 λ=1.0 |
| graph_residual.py (DGCNN EdgeConv) | 直接回归 [B,3] | 统一 MLP (utils/heads.py) | Soft-histogram | 固定 λ=1.0 |
| graph_res_GCN.py (PyG SAGEConv) | 直接回归 [B,3] | 统一 MLP (utils/heads.py) | Soft-histogram | 固定 λ=1.0 |
| SPT (脉冲神经网络) | 直接回归 [B,3] | Conv1d + SpikeNode (维度对齐) | Soft-histogram | 固定 λ=1.0 |
| 3DETR (例外) | centroid-offset [B,6] | per-query GenericMLP | Soft-histogram | 固定 λ=1.0 |

**所有模型统一训练条件 → backbone 是唯一变量 → 论文对比公平。**

---

## 训练命令模板

```powershell
# 设置环境变量 (VSCode 打开项目时已自动设置)
$env:PYTHONPATH = "D:\PYproject\SPAD"

# 训练自研模型 (GCN 变体)
& "D:\anaconda3\envs\torchnew\python.exe" "D:\PYproject\SPAD\scripts\train.py" \
    --model graph_residual_gcn --batch-size 32 --epochs 100

# 训练 baseline (示例: PointNet++)
& "D:\anaconda3\envs\torchnew\python.exe" "D:\PYproject\SPAD\scripts\train.py" \
    --model pointnet2 --batch-size 32 --epochs 100
```

常用参数:
- `--model <name>`: dgcnn / pointnet / pointnet2 / pointmlp / graph_residual / graph_residual_gcn / ...
- `--batch-size 32 --epochs 100`
- `--num-points 1024` (每样本固定点数)
- `--lr 1e-3 --min-lr 1e-5` (余弦退火)
- `--cls-loss-weight 1.0 --box-loss-weight 1.0` (固定权重)
- `--no-auto-balance` (默认，禁用 Kendall)
- `--amp / --tf32` (默认开启)

---

## 论文实验建议

1. **统一训练条件**: 所有模型使用相同 loss / 数据集划分 (seed=42) / 超参
2. **评估指标**:
   - 分类: Top-1 / Top-3 Accuracy
   - 3D Box: IoU (固定半宽重建后计算)
3. **对比维度**: backbone 架构是唯一变量，其余全部固定
4. **消融实验**: graph_residual vs graph_res_GCN (DGCNN EdgeConv vs SAGEConv)
