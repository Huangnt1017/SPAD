# 分类定位消融实验、执行与结果母版

> 当前状态：**核心 A0--A3 已完成 8/8 次 100 epoch 训练及统一无增强测试。结构消融当前完成 B1_seed42/43 与 B3/B6/B7_seed42；B3_seed43 的最新可恢复 `_last.pth` 明确记录 epoch 93，训练日志只进入 epoch 94，未发现 epoch 97 checkpoint。B4、B6/B7_seed43 与 B8 尚未完成；核心结构补齐队列已于 2026-07-21 13:20:32 启动，当前 B3 正从 epoch 93 恢复训练，B8 不在本次队列。**

最后更新：2026-07-21

## 1. 方案结论

新的分层方案比“所有开关都做三 seed”更好，原因是：

1. **核心归因更集中**：B1、B3、B4、B6 分别隔离坐标图、坐标残差、聚合器和特征残差；B7 再检查两条显式坐标增强路径的合并贡献与交互。
2. **计算量更可控**：SE 和是否包含自身点降为单 seed 附录，不挤占核心预算；目标性权重只保留 0、0.25、0.5、1.0 四个有解释力的点。
3. **仍保留同 seed 配对**：核心主矩阵、核心结构消融和权重敏感性均使用 seed42/43，可计算逐 seed paired delta。
4. **增加算子级对照**：B8 在完全相同的 KNN、双分支、融合与残差结构下，仅把 GraphSAGE 换成参数量完全一致的 EdgeCNN，避免把图卷积差异与模型容量混淆。
5. **避免为形式上的第三 seed 延误主线**：seed44 已有资产可单独报告，但不再阻塞主结果。

统计边界：`n=2` 只能报告两个独立运行、mean/std 和同 seed paired delta；**不做显著性声明，不把较小 std 解释为稳定性已经充分证明**。如论文评审明确要求更强统计证据，再补第三 seed，而不是预先把全部结构项扩大到三 seed。

## 2. A0--A3 主矩阵：2 seed

### 2.1 因子定义

| ID | Backbone | 有效 box head | 有效 `lambda_obj` | 主要比较 | 主 seed |
|---|---|---|---:|---|---|
| A0 | DGCNN | 标准 MLP | 0 | 统一协议下的骨干参照 | 42/43 |
| A1 | GraphResidual-GCN 全开 | 标准 MLP | 0 | `A1-A0`：骨干整体差异 | 42/43 |
| A2 | GraphResidual-GCN 全开 | centroid | 0 | `A2-A1`：定位头参数化差异 | 42/43 |
| A3 | GraphResidual-GCN 全开 | centroid | 0.5 | `A3-A2`：目标性 BCE 增量 | 42/43 |

说明：baseline/MLP 头没有 `seg_logits`，因此旧参数中即使出现 `seg_loss_weight=0.5`，有效 `lambda_obj` 仍为 0。`MLP + lambda_obj=0.5` 与 `MLP + lambda_obj=0` 在计算图上等价，不占正式训练名额。

### 2.2 主矩阵资产台账

| ID | seed42 | seed43 | 主统计 |
|---|---|---|---|
| A0 | 已有 100 epoch 资产 | 已有 100 epoch 资产 | 两 seed |
| A1 | 已有 100 epoch 资产 | 已有 100 epoch 资产 | 两 seed |
| A2 | 已有 100 epoch 资产 | 已有 100 epoch 资产 | 两 seed |
| A3 | 已有 100 epoch 资产 | 已有 100 epoch 资产 | 两 seed |

注册表已显式登记所有 8 个 best/last checkpoint 和训练日志，`B0` 与 `lambda_obj=0/0.5` 锚点会复用这些资产，不会误重训。

### 2.3 seed44 的处理

| ID | 状态 | 用途 |
|---|---|---|
| A0-seed44 | 已有资产 | 额外稳健性观察，不进主 mean/std |
| A1-seed44 | 已有资产 | 额外稳健性观察，不进主 mean/std |
| A2-seed44 | 已有资产 | 额外稳健性观察，不进主 mean/std |
| A3-seed44 | 未要求 | 用户后续如需可手动启动；不属于当前完成条件 |

禁止把 seed44 与 seed42/43 混成“主三 seed”统计。若未来完成 A3-seed44，应单列 robustness 表，或在补齐 A0--A3 完整第三 seed 后重新定义统计方案。

## 3. 核心结构消融：B0/B1/B3/B4/B6/B7 × 2 seed

全部固定：

```text
model = graph_residual_gcn_ablation
box_head = mlp
lambda_obj = 0
epochs = 100
train seeds = 42, 43
```

| ID | 相对 B0 的变化 | 开关 | 回答问题 | seed |
|---|---|---|---|---|
| B0 | 全开结构，复用同 seed A1 | 默认全开 | 结构锚点 | 42/43 |
| B1 | 无坐标图 | `--gcn-no-physical-branch` | 静态物理坐标图 GraphSAGE 分支是否有增量 | 42/43 |
| B3 | 无坐标残差 | `--gcn-no-coord-residual` | 坐标门控、编码器和受控残差整体是否有增量 | 42/43 |
| B4 | mean 聚合 | `--gcn-aggregation mean` | max 相对 mean 聚合是否必要 | 42/43 |
| B6 | 无特征残差 | `--gcn-no-feature-residual` | 显式 feature residual 是否有增量 | 42/43 |
| B7 | 同时关闭两条显式坐标增强路径 | `--gcn-no-physical-branch --gcn-no-coord-residual` | 两条坐标路径的合并贡献、互补或冗余 | 42/43 |

新增正式训练数：B0 复用 A1，不新增；B1/B3/B4/B6/B7 各两次，共 **10 次新训练**。

### 3.1 当前进度

截至 2026-07-21：

- `B0_seed42/43` 复用同 seed 的 A1 checkpoint 和统一无增强测试资产；
- `B1_seed42/43`、`B3/B6/B7_seed42` 已完成 100 epoch；核心结构完整 checkpoint 覆盖 `7/12`，新增训练完成 `5/10`；
- 最新正式训练是 `B3_no_coord_residual_seed43`：`_last.pth` 修改时间为 2026-07-20 19:00:17，checkpoint 元数据为 `epoch=93`；console 仅出现 3 条 epoch 94 起始进度，没有 epoch 95--100，也没有 epoch 97 checkpoint，因此恢复点必须按 epoch 93 计算；
- 2026-07-21 13:20:32 已启动核心结构补齐后台队列：总启动器 PID `41768`，第一阶段队列 PID `4856`，B3 训练 PID `14588`；13:20:41 日志确认从 epoch 93 checkpoint 正确恢复并进入 epoch 94。固定环境为 PyTorch `2.7.1+cu128`、CUDA `12.8`、RTX 4070 SUPER；
- 尚未启动 `B4_seed42/43`、`B6/B7_seed43`；B1/B3/B6/B7 的新增资产仍待统一无增强测试；
- B8 EdgeCNN 两 seed 已完成注册表接入、参数匹配、GPU smoke 和 dry-run，尚未正式训练。

2026-07-20 的
[`ablation_training_plan_20260720_10h.md`](ablation_training_plan_20260720_10h.md)
保留为上一轮队列快照；最新约 10 小时执行顺序和命令以本文第 9 节为准。

### 3.2 归因规则

- `B1-B0`：坐标图 GraphSAGE 分支整体贡献；不能声称只解释某一层。
- `B3-B0`：`coord_gate + coord_res + coord_encoder + coord_scale` 整体贡献。
- `B4-B0`：max 与 mean 聚合选择的影响。
- `B6-B0`：显式 feature residual 的贡献。
- `B7-B0`：两条显式坐标增强路径的合并贡献。
- `B7` 不能替代 `B1` 和 `B3`：只有三者同时保留，才能区分单路径贡献与联合关闭后的交互。

### 3.3 GraphSAGE 与 EdgeCNN 算子对照

B8 单列为 `operator` family，不改变原 B0--B7 的硬删除语义：

| ID | GraphSAGE 锚点 | 对照算子 | 固定项 | seed |
|---|---|---|---|---|
| B8 | 同 seed B0/A1 | EdgeCNN：`Linear(2Cin,Cout)` edge MLP + 同聚合 | KNN、双分支、SE、fusion、feature/coord residual、head 和训练协议全部不变 | 42/43 |

EdgeCNN 每条 `j→i` 边使用 `[x_j-x_i, x_i]`，其参数量与同通道 `SAGEConv` 完全一致。报告 `B8-B0` 的同 seed 差值，才能把“GraphSAGE 消息传递”与“CNN 式边卷积”区别开，而不混入参数量差异。自动分析入口：

```powershell
& "D:\Anaconda3\envs\torchnew\python.exe" scripts\analyze_gcn_vs_edge_cnn.py
```

## 4. 单 seed 附录：B2/B5

| ID | 变化 | 开关 | seed | 解释边界 |
|---|---|---|---:|---|
| B2 | 无 SE | `--gcn-no-se-gate` | 42 | 附录观察通道门控，不作核心结论 |
| B5 | KNN 包含自身点 | `--gcn-include-self` | 42 | 附录观察邻域根节点语义，不作核心结论 |

共 **2 次新训练**。如果单 seed 结果幅度很大、与核心结论直接相关，再决定是否追加 seed43；当前计划不预先扩展。

## 5. 目标性权重敏感性

固定 A2/A3 的完整 GCN + centroid 结构，只变化 `lambda_obj`：

| `lambda_obj` | seed | 资产策略 |
|---:|---|---|
| 0 | 42/43 | 复用 A2-seed42/43 |
| 0.25 | 42/43 | 新训练 |
| 0.5 | 42/43 | 复用 A3-seed42/43 |
| 1.0 | 42/43 | 新训练 |

因此敏感性表有 8 行，但只新增 **4 次训练**。0.25 用于观察低于默认值时的变化，1.0 用于观察更强目标性监督；不再保留 0.1、0.75 等稠密网格，避免把测试集变成调参集。

## 6. 总训练预算

| 阶段 | 表中 run | 复用 | 新训练总数 | 已完成新训练 | 剩余新训练 |
|---|---:|---:|---:|---:|---:|
| A0--A3 主矩阵 | 8 | 8 已有 | 0 | 0 | 0 |
| 核心结构 B0/B1/B3/B4/B6/B7 | 12 | B0 两次 | 10 | 5 | 5（含 B3_seed43 续训） |
| 算子对照 B8 EdgeCNN | 2 | 0 | 2 | 0 | 2 |
| 附录 B2/B5 | 2 | 0 | 2 | 0 | 2 |
| `lambda_obj` 敏感性 | 8 | 4 | 4 | 0 | 4 |
| **合计** | **32** | **14** | **18** | **5** | **13** |

seed44 robustness 不计入上述主计划预算。B8 作为独立算子对照，默认在原核心结构补齐后单独训练，不自动并入已经暂停的阶段 A 队列。

## 7. 冻结数据与训练协议

### 7.1 环境和路径

```text
Python: D:\Anaconda3\envs\torchnew\python.exe
Project root: D:\PYproject\SPAD
Chapter-3 dataset: D:\PYproject\SPADdata\2025-04-30-dpc
Formal three-page source: D:\PYproject\SPADdata\20250430\2025-04-30-pc
GPU: NVIDIA GeForce RTX 4070 SUPER, about 12 GB
PyTorch/CUDA runtime: 2.7.1+cu128 / 12.8
```

用户已确认：分类与消融实验使用 `2025-04-30-dpc`。为与已有 8526 样本结果连续，本系列继续只使用该目录；正式三页源 `2025-04-30-pc` 必须另建系列，禁止混表。

### 7.2 不可变划分

```text
Raw train/val/test: 5116 / 1705 / 1705
Train augmented views: 15348 (num_aug=3)
Split seed: 42
Main training seeds: 42, 43
Evaluation point-sampling seed: 42
Split cache: D:\PYproject\SPADdata\2025-04-30-dpc\.split_cache.json
Split cache SHA256: AB94E67744AC3C73FC45A2D3E3E389773661E3EEBA85A6F8EF2C3025220A9F22
```

成员列表 SHA256：

```text
train: A7A2B049DCDE91D3756EBEAAD36411F263EC65CF99734A83D839E45C5EFD4F2A
val:   46B642920FCC2C0652742E8348A5E5C7A2F6585C59DBEE4B775CB7B301DC84F4
test:  A126290B8B078C1A3DD706BEB397887CD505EC0EA4BEFB0B44FF7B7A4BD3D654
```

每项训练开始前和结束后必须校验 split cache SHA256；不得删除、重建或覆盖该 cache。

### 7.3 优化配置

| 项目 | 固定值 |
|---|---:|
| epochs | 100 |
| physical batch / grad accumulation | 32 / 1 |
| points/sample | 1024 |
| train augmentation views | 3 |
| optimizer | AdamW |
| initial/min LR | `1e-3` / `1e-5` |
| weight decay | `1e-4` |
| label smoothing | 0.1 |
| classification/depth weight | 1.0 / 10.0 |
| auto balance | off |
| AMP / TF32 | off / off |
| EMA | off |
| gradient checkpoint | true |
| train-time validation | 一个确定性增强视图 |
| final test | 无增强，eval seed 42 |

## 8. Batch 与 A3-seed44 结论

- 现有日志没有 OOM 证据；同结构显存约 3.5 GB/12 GB，GPU 利用率约 98%。
- A2-seed43 用时 `1:53:09`，A3-seed43 用时 `1:53:59`，目标性分支只增加约 50 秒。
- 约两小时主要来自 15348 个训练视图、动态 kNN、FP32、逐 epoch 验证和 gradient checkpoint，不是超显存导致。
- `batch16 + accum2` 主要用于降显存，通常比 `batch32 + accum1` 更慢；“拆分 batch”不能作为默认加速方案。
- A3-seed44 不自动启动。用户未来若确实需要，只在无其他 GPU 训练时手动执行 robustness 单项。

安全 dry-run：

```powershell
$SPAD_PYTHON = "D:\Anaconda3\envs\torchnew\python.exe"
Set-Location "D:\PYproject\SPAD"
$env:PYTHONPATH = "D:\PYproject\SPAD"

& $SPAD_PYTHON scripts\run_ablation_matrix.py `
  --families robustness `
  --experiments A3_seed44 `
  --run-tag dry_a3_seed44_optional
```

如需先做 batch 基准，只在无现有 CUDA 训练时运行 `scripts/benchmark_a3_batch.py`；默认模式为 dry-run。

## 9. 推荐执行顺序

### 9.1 下一训练窗：核心结构补齐 + B8（约 10 小时）

按 checkpoint 元数据和已有实测耗时安排：

| 阶段 | 实验 | 恢复/新增 | 预计纯训练时间 | 累计 |
|---|---|---|---:|---:|
| 0 | `B3_no_coord_residual_seed43` | 从 epoch 93 `_last.pth` 恢复到 100 | `0:07--0:12` | `0:07--0:12` |
| 1 | `B6_no_feature_residual_seed43`、`B7_no_coordinate_pathways_seed43` | 新训练两次 | 约 `3:14` | `3:21--3:26` |
| 2 | `B4_mean_aggregation_seed42/43` | 新训练两次 | 约 `3:40` | `7:01--7:06` |
| 3 | `B8_edge_cnn_seed42/43` | 新训练两次 | 约 `3:50--4:10` | `10:51--11:16` |

因此“约 10 小时”按完整配对口径实际应预留约 `10:50--11:20` 纯训练墙钟；随后统一无增强测试约 `0:25--0:40`，审计、汇总与文档同步约 `0:10`。若硬限制为 10 小时，应优先完整结束阶段 0--2 和 `B8_seed42`，`B8_seed43` 允许依靠每 epoch `_last.pth` 在下一窗口续训，不修改 batch、精度、增强或 epoch 协议来抢时间。

执行优先级不变：先补齐原核心结构，再完成 B8 两 seed 算子配对；之后才安排 `lambda_obj=0.25/1.0` 和附录 B2/B5。每个阶段完成后立即更新 `logs/training_results_since_202607.md`，不要等全部阶段结束才记录。

### 9.2 已复核的 dry-run

2026-07-21 已使用固定解释器执行以下三段 dry-run，均通过解释器、数据根目录、split hash 和实验注册检查：

- 阶段 0--1：B3 正确识别 epoch 93 `_last.pth` 并生成 `--resume`；B6/B7_seed43 为全新训练；
- 阶段 2：B4_seed42/43 均为 `gcn_operator=sage, aggregation=mean`；
- 阶段 3：B8_seed42/43 均为 `gcn_operator=edge_cnn, aggregation=max`，其余结构开关与 B0/A1 一致。

### 9.3 已启动的核心结构正式队列

用户于 2026-07-21 明确下令启动阶段 0--2；已于 13:20:32 通过隐藏 PowerShell 启动器串行执行，B8 不包含在本次队列中。

```text
总启动器 PID: 41768
第一阶段队列 PID: 4856
当前 B3 训练 PID: 14588
总队列目录: outputs/ABL/training_queues/structure_core_completion_20260721_132032
阶段 0--1 run_tag: structure_seed43_completion_20260721_132032
阶段 2 run_tag: structure_b4_pair_20260721_132032
```

实际执行顺序：

1. `B3_no_coord_residual_seed43`：从 epoch 93 `_last.pth` 恢复；
2. `B6_no_feature_residual_seed43`；
3. `B7_no_coordinate_pathways_seed43`；
4. `B4_mean_aggregation_seed42`；
5. `B4_mean_aggregation_seed43`。

阶段 0--1 成功退出后，启动器才会进入 B4 两 seed；任一阶段异常都会停止后续阶段，避免错误继续扩散。B8 仍等待用户后续命令，不会由当前启动器自动开始。

## 10. 自动汇总与当前结果

```powershell
$SPAD_PYTHON = "D:\Anaconda3\envs\torchnew\python.exe"
Set-Location "D:\PYproject\SPAD"
$env:PYTHONPATH = "D:\PYproject\SPAD"

& $SPAD_PYTHON scripts\summarize_ablation.py --families core
& $SPAD_PYTHON scripts\update_ablation_docs.py
```

<!-- ABLATION_CORE_RESULTS_START -->
### A0--A3 两 seed 最终结果

> 同步时间：2026-07-20T19:36:33  
> 训练完成：8/8；统一测试与完整审计：8/8；split SHA256 `AB94E67744AC3C73FC45A2D3E3E389773661E3EEBA85A6F8EF2C3025220A9F22`；eval seed=42、无增强。

| ID | seed | 有效 head | 有效 λobj | epoch | best epoch | 训练耗时 | Top-1 | F1 | z-MAE | center-MAE | mIoU | AP50 | AP50:95 | 状态 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A0_seed42 | 42 | mlp | 0.00 | 100 | 97 | 2:50:32 | 97.60% | 97.48% | 0.009312 | 0.012457 | 0.5897 | 53.29% | 16.35% | 完成 |
| A1_seed42 | 42 | mlp | 0.00 | 100 | 94 | 1:58:27 | 98.06% | 97.90% | 0.006094 | 0.011924 | 0.6623 | 80.22% | 28.33% | 完成 |
| A2_seed42 | 42 | centroid | 0.00 | 100 | 96 | 1:53:58 | 98.24% | 98.12% | 0.006592 | 0.011745 | 0.7237 | 93.81% | 42.61% | 完成 |
| A3_seed42 | 42 | centroid | 0.50 | 100 | 74 | 2:01:06 | 97.77% | 97.68% | 0.006727 | 0.011719 | 0.7115 | 94.92% | 41.51% | 完成 |
| A0_seed43 | 43 | mlp | 0.00 | 100 | 90 | 2:40:48 | 97.48% | 97.30% | 0.007757 | 0.011948 | 0.6221 | 64.80% | 20.62% | 完成 |
| A1_seed43 | 43 | mlp | 0.00 | 100 | 98 | 1:51:42 | 98.12% | 98.01% | 0.007336 | 0.012432 | 0.6275 | 68.29% | 22.54% | 完成 |
| A2_seed43 | 43 | centroid | 0.00 | 100 | 78 | 1:53:09 | 97.54% | 97.42% | 0.006238 | 0.010720 | 0.7420 | 93.18% | 46.12% | 完成 |
| A3_seed43 | 43 | centroid | 0.50 | 100 | 100 | 1:53:59 | 97.83% | 97.72% | 0.006391 | 0.010700 | 0.7312 | 95.64% | 45.24% | 完成 |

#### 两 seed mean ± std

| ID | n | Top-1 | F1 | z-MAE | center-MAE | mIoU | AP50 | AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 2/2 | 97.54 ± 0.08% | 97.39 ± 0.13% | 0.0085 ± 0.0011 | 0.0122 ± 0.0004 | 0.6059 ± 0.0229 | 59.04 ± 8.14% | 18.49 ± 3.02% |
| A1 | 2/2 | 98.09 ± 0.04% | 97.95 ± 0.08% | 0.0067 ± 0.0009 | 0.0122 ± 0.0004 | 0.6449 ± 0.0246 | 74.25 ± 8.44% | 25.44 ± 4.09% |
| A2 | 2/2 | 97.89 ± 0.50% | 97.77 ± 0.50% | 0.0064 ± 0.0003 | 0.0112 ± 0.0007 | 0.7328 ± 0.0130 | 93.49 ± 0.45% | 44.36 ± 2.48% |
| A3 | 2/2 | 97.80 ± 0.04% | 97.70 ± 0.03% | 0.0066 ± 0.0002 | 0.0112 ± 0.0007 | 0.7214 ± 0.0139 | 95.28 ± 0.51% | 43.38 ± 2.64% |

#### 同 seed 配对差值（上组减下组）

- **backbone (A1-A0)**：Top-1 0.56%，F1 0.56%，z-MAE -0.001820，center-MAE -0.000025，mIoU 0.0390，AP50 15.21%，AP50:95 6.95%.
- **centroid_head (A2-A1)**：Top-1 -0.21%，F1 -0.18%，z-MAE -0.000300，center-MAE -0.000945，mIoU 0.0879，AP50 19.24%，AP50:95 18.93%.
- **objectness_bce (A3-A2)**：Top-1 -0.09%，F1 -0.07%，z-MAE 0.000144，center-MAE -0.000024，mIoU -0.0115，AP50 1.79%，AP50:95 -0.99%.
- **full_method (A3-A0)**：Top-1 0.26%，F1 0.31%，z-MAE -0.001976，center-MAE -0.000993，mIoU 0.1155，AP50 36.24%，AP50:95 24.89%.

8 个核心 run 已全部通过 checkpoint、统一无增强测试、有效配置和 split hash 审计。
<!-- ABLATION_CORE_RESULTS_END -->

## 11. 代码与文档职责

| 文件 | 职责 |
|---|---|
| `scripts/ablation_registry.py` | 唯一实验编号、seed、结构开关、权重和复用资产来源 |
| `scripts/run_ablation_matrix.py` | 注册表驱动训练队列；默认 dry-run |
| `scripts/run_ablation_evaluation.py` | 统一无增强测试 |
| `scripts/summarize_ablation.py` | 逐 run、两 seed mean/std、paired delta 与审计 |
| `scripts/smoke_ablation_matrix.py` | 结构前向/反向 smoke，不替代正式训练 |
| `scripts/audit_ablation_assets.py` | checkpoint、日志、指标与配置资产审计 |
| `scripts/update_ablation_docs.py` | 把 core 8-run 快照同步到两份 Markdown |
| `scripts/benchmark_a3_batch.py` | A3 batch/显存基准；默认 dry-run |
| `tests/test_ablation_registry.py` | 核心/附录/敏感性矩阵和全开模型等价性测试 |
| `logs/training_results_since_202607.md` | 长期训练结果与当前消融进度 |

family 名称：

```text
core
robustness
structure_core
structure_appendix
lambda
```

旧 `structure` 仍作为兼容别名，展开为 `structure_core + structure_appendix`。

## 12. 完成判据

### 12.1 A0--A3 主矩阵

- [x] A0--A3 均有 seed42/43 两次 100 epoch 正式训练；
- [x] 8 个 core best checkpoint 全部完成统一无增强测试；
- [x] 每个指标 JSON 记录 eval seed、`augment_eval=false`、box 坐标空间和 split hash；
- [x] 输出两 seed mean/std 和同 seed paired delta；
- [x] 将最终结果同步到 `logs/training_results_since_202607.md`；
- [x] seed44 与主统计分离，A3-seed44 不阻塞完成。

### 12.2 核心结构消融

- [ ] B0/B1/B3/B4/B6/B7 均覆盖 seed42/43；
- [ ] B8 EdgeCNN 覆盖 seed42/43，并与同 seed B0 做参数匹配 paired delta；
- [x] B0 对应同 seed A1，且不重复训练；
- [ ] B1/B3/B4/B6 每项只改变一个注册开关；
- [ ] B7 同时且仅同时关闭 physical branch 和 coord residual；
- [ ] 输出逐 seed、mean/std 和相对 B0 的 paired delta；
- [ ] 不对 `n=2` 作显著性声明。

### 12.3 附录与敏感性

- [ ] B2/B5 完成 seed42 单次训练和统一测试；
- [ ] `lambda_obj=0.25/1.0` 完成 seed42/43；
- [x] `lambda_obj=0/0.5` 分别复用 A2/A3；
- [ ] 权重选择不依据测试集反复调参。

## 13. 论文表述边界

- A0--A3 支撑骨干整体、定位头参数化、目标性 BCE 三层归因。
- 只有核心 B 系列完成后，才能分别声称坐标图、坐标残差、聚合器或特征残差有贡献。
- B2/B5 是单 seed 附录，不应写成强核心结论。
- B7 用于解释两条坐标增强路径的联合效果，不等于分别证明两条路径都有效。
- 所有结果必须同时报告分类与定位指标，不能只选择有利指标。
- 不使用测试集筛选 `lambda_obj`、结构开关或 best epoch。

## 14. 变更记录

- 2026-07-21 13:20：用户确认启动核心结构阶段 0--2；后台队列已从 B3_seed43 epoch 93 恢复，随后串行执行 B6/B7_seed43 与 B4_seed42/43。B8 未纳入本次启动。
- 2026-07-21：按 checkpoint 元数据复核最新训练；B3_seed43 可恢复点为 epoch 93（日志仅进入 epoch 94，并非 97）。将 B8 两 seed 纳入下一约 10 小时训练窗，补充三阶段 dry-run、耗时边界和待确认正式命令；本次未启动训练。
- 2026-07-20：新增 B8 参数匹配 EdgeCNN 算子对照、自动分析脚本和 GPU smoke；不改变既有硬删除消融定义。
- 2026-07-20：确认分类/消融数据根目录为 `D:\PYproject\SPADdata\2025-04-30-dpc`；修复统一测试汇总识别并同步 A0--A3 结果。
- 2026-07-20：登记 B1/B3/B6/B7_seed42 完成状态，并制定补齐剩余 6 个核心结构实验的约 10 小时计划。
- 2026-07-17：把 A0--A3 主矩阵由三 seed 精简为 seed42/43；seed44 转为额外 robustness，不进主 mean/std。
- 2026-07-17：核心结构改为 B0/B1/B3/B4/B6/B7 × seed42/43；B2 无 SE、B5 包含自身改为 seed42 附录。
- 2026-07-17：新增 B7，同时关闭坐标图与坐标残差两条显式坐标增强路径。
- 2026-07-17：目标性敏感性精简为 `lambda_obj ∈ {0,0.25,0.5,1.0}` × seed42/43，只新增四次训练。
- 2026-07-17：确认 A3 训练约两小时不是 OOM；batch 拆分主要省显存，通常不能加速。A3-seed44 保持用户可选手动启动，不自动执行。
