# SPAD 点云多任务 Loss 行为说明

本文档说明当前 `utils/loss.py` 中 `PointCloudMultiTaskLoss` 的数学定义、为什么日志中的 `total_loss` 可以为负、以及如何解读分类与 3D 中心回归的量级。

## 1. 当前训练目标

当前训练是一个两任务联合优化问题：

1. 分类任务：预测 26 类目标标签。
2. 几何任务：预测目标 3D box 的中心点 `[cx, cy, cz]`。

对应的原始分项 loss 为：

```text
L_cls = CrossEntropy(logits, class_label)

L_box = sum_d log(1 + ((c_pred_d - c_gt_d) / h_d)^2)
        d in {x, y, z}
```

其中 `h_d` 是目标框在每个维度上的固定半宽，使用归一化空间中的数值：

```text
h_x = 7.5 / 63  ~= 0.11905
h_y = 7.5 / 63  ~= 0.11905
h_z = 2.5 / 109 ~= 0.02294
```

这里的 `L_box` 是 Log-Cauchy 形式的中心点回归损失。它不是直接回归 6 维角点框，而是把 GT 角点框转成中心点，然后只监督中心点误差。评估和可视化时再用固定半宽把中心点还原为 6 维 box。

## 2. Box Loss 的尺度含义

`L_box` 中的误差不是直接使用归一化坐标差，而是先除以各维半宽：

```text
delta_norm_d = (c_pred_d - c_gt_d) / h_d
```

这样做的含义是：误差被转换成“相对于目标物理尺寸的误差”。

例如：

```text
x/y 方向半宽约 0.119
z 方向半宽约 0.023
```

所以同样是 `0.01` 的归一化坐标误差，在 z 轴上比在 x/y 轴上更严重。因为真实目标在 z 方向更薄，z 中心偏一点对 3D IoU 的影响更大。

这不是无意的量级不一致，而是刻意把不同维度按物理不确定度归一。也就是说，`h_d` 越小，该维度越需要精确。

Log-Cauchy 形式：

```text
log(1 + delta_norm^2)
```

有三个性质：

1. 小误差时近似平方损失：`log(1 + x^2) ~= x^2`。
2. 大误差时增长变慢：`log(1 + x^2) ~= 2 log(|x|)`。
3. 梯度不会像纯 IoU 在不重叠时那样直接消失。

因此它比普通 MSE 更稳健，也比仅用 IoU 更容易在初期提供有效梯度。

## 3. Kendall 自动权重的参数化

默认情况下，代码使用 Kendall et al. 的同方差不确定性自动平衡。总 loss 是：

```text
L_total =
    exp(-s_cls) * L_cls + s_cls
  + exp(-s_box) * L_box + s_box
```

其中：

```text
s_cls = log(sigma_cls^2)
s_box = log(sigma_box^2)
```

代码中对应两个可学习参数：

```python
self.log_var_cls
self.log_var_box
```

注意：优化的是 `log(sigma^2)`，不是直接优化 `sigma^2`。

因此：

```text
sigma^2 = exp(s)
```

无论 `s` 是正数、零还是负数，`exp(s)` 始终大于 0。也就是说，代码不会得到负方差；看到 `log_var` 为负，只代表估计出的任务方差小于 1。

## 4. 为什么 Total Loss 会变成负数

`total_loss` 可以为负，这是 Kendall 形式的正常数学结果。

看单个任务项：

```text
f(s) = exp(-s) * L + s
```

对 `s` 求导：

```text
df/ds = -exp(-s) * L + 1
```

令导数为 0：

```text
exp(-s) * L = 1
s = log(L)
```

代回原函数：

```text
f(log L) = 1 + log(L)
```

因此当原始分项 loss 满足：

```text
L < exp(-1) ~= 0.3679
```

这一项的最优值就可能为负。

这说明：

1. 负的 `total_loss` 不代表 CrossEntropy 为负。
2. 负的 `total_loss` 不代表 box loss 为负。
3. 负的 `total_loss` 不代表方差为负。
4. 负的 `total_loss` 不会天然导致反向传播错误。

它只是说明某些分项 loss 已经很小，而 Kendall 的 `+ log_var` 项把优化目标整体平移到了负区间。

## 5. 日志中各字段应该如何看

训练日志中主要字段含义如下：

```text
train_loss / val_loss
```

这是带 Kendall 自动权重后的优化目标，即 `L_total`。它可以为负，不适合直接当作“非负误差大小”理解。

```text
train_top1 / val_top1
train_top3 / val_top3
```

分类准确率指标。它们不受 loss 正负号影响。

```text
train_box_gauss / val_box_gauss
```

这是原始 box 分项损失 `L_box`，不包含 Kendall 的 `exp(-s_box)` 和 `+s_box`。它应该非负，更适合观察中心点回归是否在收敛。

```text
train_box_iou / val_box_iou
```

这是用预测中心点和固定半宽重建 3D box 后计算的 IoU，仅用于监控，不参与反向传播。

因此判断训练是否正常时，不应只看 `train_loss` 是否为负，而应同时看：

```text
val_top1
val_top3
val_box_gauss
val_box_iou
```

## 6. 什么时候需要关闭 Kendall 自动权重

当前默认使用：

```text
--auto-balance
```

如果需要让总 loss 保持传统非负加权和，可以显式关闭：

```text
--no-auto-balance
```

此时总 loss 变为：

```text
L_total = cls_loss_weight * L_cls + box_loss_weight * L_box
```

对应参数为：

```text
--cls-loss-weight
--box-loss-weight
```

这种模式下，只要两个权重非负，`total_loss` 就不会为负。

但这会改变实验设定。若目标是复现实验中的 Kendall 自动任务平衡，则不应因为 `total_loss` 为负而关闭它。

## 7. 当前实现的合理性结论

当前 loss 的关键行为可以总结为：

1. `log_var_cls` 和 `log_var_box` 可以为负，但真实方差 `exp(log_var)` 始终为正。
2. `total_loss` 可以为负，这是 Kendall 自动权重公式的正常结果。
3. `box_gauss_loss` 本身非负，适合观察几何回归误差。
4. `box_iou_mean` 只是监控指标，不参与训练梯度。
5. box 回归按固定目标半宽做尺度归一，z 轴权重更高是有物理含义的。

因此，仅凭日志中的 `train_loss` 或 `val_loss` 变负，不能判定 loss 设计错误。更可靠的判断依据是：分类准确率是否提升、`box_gauss` 是否下降、`box_iou` 是否上升，以及验证集表现是否稳定。
