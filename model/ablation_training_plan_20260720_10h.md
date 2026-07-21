# 2026-07-20 结构消融约 10 小时训练计划

> 状态：**既有硬删除设计确认保留；B1_seed43 完成，B3_seed43 保存至 epoch 93，B6/B7 与 B4 尚未启动。B8 EdgeCNN 算子对照代码已就绪，但不自动并入本队列。**
>
> 目标：先补齐原核心结构 `B0/B1/B3/B4/B6/B7 × seed42/43`；B8 EdgeCNN × seed42/43 作为独立算子对照另行启动。

## 1. 固定协议

```text
Python: D:\Anaconda3\envs\torchnew\python.exe
Project root: D:\PYproject\SPAD
Classification data root: D:\PYproject\SPADdata\2025-04-30-dpc
Split cache SHA256: AB94E67744AC3C73FC45A2D3E3E389773661E3EEBA85A6F8EF2C3025220A9F22
GPU: NVIDIA GeForce RTX 4070 SUPER, about 12 GB
Epochs: 100
Batch / accumulation: 32 / 1
Points / train augmentation count: 1024 / 3
Precision: FP32, AMP off, TF32 off
Train/validation augmentation: on
Model/head/lambda_obj: graph_residual_gcn_ablation / mlp / 0
```

分类和消融实验继续使用 `2025-04-30-dpc`。`20250430\2025-04-30-pc` 是正式三页源点云目录，不与本分类实验系列混表。

## 2. 当前缺口

| 实验 | 目的 | 预计训练时间 | 优先级 |
|---|---|---:|---:|
| B1_no_physical_seed43 | 与现有 B1_seed42 组成两 seed 配对 | 1:49 | 1 |
| B3_no_coord_residual_seed43 | 与现有 B3_seed42 组成两 seed 配对 | 1:39 | 1 |
| B6_no_feature_residual_seed43 | 与现有 B6_seed42 组成两 seed 配对 | 1:47 | 1 |
| B7_no_coordinate_pathways_seed43 | 与现有 B7_seed42 组成两 seed 配对 | 1:27 | 1 |
| B4_mean_aggregation_seed42 | 新增 mean 聚合首个 seed | 约 1:50 | 2 |
| B4_mean_aggregation_seed43 | 补齐 mean 聚合配对 | 约 1:50 | 2 |

B1/B3/B6/B7 的估时直接采用 seed42 实测；B4 尚无正式运行，按同规模结构取约 1 小时 50 分钟。

## 3. 执行分段与时间预算

### 阶段 A：优先补齐已有四项的 seed43

预计 `6:42`，完成后 B1/B3/B6/B7 都可立即形成两 seed 统计。

执行顺序：

1. B1_no_physical_seed43
2. B3_no_coord_residual_seed43
3. B6_no_feature_residual_seed43
4. B7_no_coordinate_pathways_seed43

### 阶段 B：补齐 B4 mean 聚合两 seed

预计 `3:40`：

1. B4_mean_aggregation_seed42
2. B4_mean_aggregation_seed43

### 总预算

```text
阶段 A：约 6:42
阶段 B：约 3:40
纯训练：约 10:22
统一无增强测试：约 0:20--0:40
审计与汇总：约 0:10
总墙钟：约 10:50--11:10
```

若只能严格使用约 10 小时，优先完整执行阶段 A，再启动阶段 B。训练队列每个 epoch 保存 `_last.pth`，B4 若在时间窗口结束前未完成，可从最新 `_last.pth` 恢复；不建议为了压缩时间修改 batch、精度、增强或 epoch 协议。

## 4. 启动前 dry-run

```powershell
$SPAD_PYTHON = "D:\Anaconda3\envs\torchnew\python.exe"
Set-Location "D:\PYproject\SPAD"
$env:PYTHONPATH = "D:\PYproject\SPAD"

# 阶段 A：已有 seed42 项的 seed43 配对
& $SPAD_PYTHON scripts\run_ablation_matrix.py `
  --families structure_core `
  --experiments B1_no_physical_seed43,B3_no_coord_residual_seed43,B6_no_feature_residual_seed43,B7_no_coordinate_pathways_seed43 `
  --run-tag dry_structure_seed43_pair_completion_20260720

# 阶段 B：B4 mean 聚合两 seed
& $SPAD_PYTHON scripts\run_ablation_matrix.py `
  --families structure_core `
  --experiments B4_mean_aggregation_seed42,B4_mean_aggregation_seed43 `
  --run-tag dry_structure_b4_pair_20260720
```

2026-07-20 已执行上述 dry-run；6 个实验均显示 `PLANNED`，数据根目录、seed 和结构开关正确。

## 5. 正式训练命令

阶段 A 已由用户明确确认并启动：

```text
run_tag: structure_seed43_pair_completion_20260720
queue started: 2026-07-20 15:52:30
current first run: B1_no_physical_seed43
queue status: paused_for_design_review
B1 result: completed, epoch 100
B3 recoverable state: epoch 93 last checkpoint
B6/B7: not started
```

阶段 B 尚未启动。对应命令如下：

```powershell
$SPAD_PYTHON = "D:\Anaconda3\envs\torchnew\python.exe"
Set-Location "D:\PYproject\SPAD"
$env:PYTHONPATH = "D:\PYproject\SPAD"

# 阶段 A
& $SPAD_PYTHON scripts\run_ablation_matrix.py `
  --execute `
  --families structure_core `
  --experiments B1_no_physical_seed43,B3_no_coord_residual_seed43,B6_no_feature_residual_seed43,B7_no_coordinate_pathways_seed43 `
  --run-tag structure_seed43_pair_completion_20260720

# 阶段 B；阶段 A 返回成功后再执行
& $SPAD_PYTHON scripts\run_ablation_matrix.py `
  --execute `
  --families structure_core `
  --experiments B4_mean_aggregation_seed42,B4_mean_aggregation_seed43 `
  --run-tag structure_b4_pair_20260720
```

默认不设置 `--max-hours`，避免在最后一个 epoch 中间强制终止。如果必须设墙钟上限，可给阶段 A 设置约 `7.25` 小时、阶段 B 设置约 `4.25` 小时，并保留至少 `0.25` 小时停止余量。

## 6. 训练结束后的统一测试与汇总

```powershell
$SPAD_PYTHON = "D:\Anaconda3\envs\torchnew\python.exe"
Set-Location "D:\PYproject\SPAD"
$env:PYTHONPATH = "D:\PYproject\SPAD"

# 对全部核心结构项执行统一无增强测试；已有匹配结果会跳过。
& $SPAD_PYTHON scripts\run_ablation_evaluation.py `
  --execute `
  --families structure_core `
  --batch-size 32 `
  --device cuda `
  --eval-seed 42

# 先生成 B 系列结构汇总并记录到长期结果文档。
& $SPAD_PYTHON scripts\summarize_ablation.py --families structure_core

# 再恢复 A0--A3 的规范自动汇总并同步长期 Markdown。
& $SPAD_PYTHON scripts\summarize_ablation.py --families core
& $SPAD_PYTHON scripts\update_ablation_docs.py

# 重新生成资产审计。
& $SPAD_PYTHON scripts\audit_ablation_assets.py --families core,structure_core
```

最终验收：

- 核心结构 checkpoint 覆盖 `12/12`；
- B1/B3/B4/B6/B7 两个 seed 均有统一无增强测试 JSON；
- 汇报逐 run、mean ± sample std、同 seed paired delta；
- 不对 `n=2` 做显著性声明；
- 训练日志、best/last checkpoint、测试指标和 split hash 一一对应。

## 7. 独立算子对照 B8（未启动）

B8 不修改原 B0--B7 的实验定义，只替换局部算子：

```text
B0: GraphSAGE, 1,331,745 parameters
B8: EdgeCNN edge MLP, 1,331,745 parameters
Controlled: KNN / dual branches / SE / fusion / residuals / head / protocol
Changed: GraphSAGE -> EdgeCNN
```

安全 dry-run：

```powershell
$SPAD_PYTHON = "D:\Anaconda3\envs	orchnew\python.exe"
Set-Location "D:\PYproject\SPAD"

& $SPAD_PYTHON scriptsun_ablation_matrix.py `
  --families operator `
  --experiments B8_edge_cnn_seed42,B8_edge_cnn_seed43 `
  --run-tag dry_b8_edge_cnn_20260720

& $SPAD_PYTHON scriptsnalyze_gcn_vs_edge_cnn.py
```

正式训练预计增加约 3.5--4 小时，需用户另行确认后启动。

