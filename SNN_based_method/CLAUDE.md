# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

详细的设计理论、数据形状约定和逐模块说明请见 `reademe.md`。本文件只收录首日投产所需的最小信息。

## 环境

- Conda 环境：`torchnew`（Python 路径 `D:\Anaconda3\envs\torchnew\python.exe`）
- SpikingJelly：使用环境中安装的官方 `spikingjelly.activation_based`，**不要**使用本地的 `spikingjelly1`
- `SNN.py` 只是兼容入口，实际转发到 `SNN_new.py`。任何模型改动都请编辑 `SNN_new.py`
- 所有可调参数的唯一真源：`SNN_config.SNNConfig`（dataclass）。不要在其他地方硬编码常量

## 常用命令

以下命令都在仓库根目录 `D:\PYproject\SPAD\` 下执行。

```powershell
$python = "D:\Anaconda3\envs\torchnew\python.exe"

# 训练（默认使用 0825 + 0826；首次运行会自动生成预计算 label 池）
& $python SNN_based_method/scripts/train.py

# 显式指定超参的训练
& $python SNN_based_method/scripts/train.py `
  --pages-per-group 128 --batch-size 8 --grad-accum-steps 8 `
  --num-workers 8 --persistent-workers --prefetch-factor 4 `
  --raw-load-mode group --cuda-prefetch --pin-memory `
  --progress-interval 50 --tf32 --cudnn-benchmark --spike-backend auto

# 从 run 目录续训（自动读取 last.pth + config.json）
& $python SNN_based_method/scripts/train.py --resume-runDir checkpoints/SNN/train_YYYYMMDD_HHMMSS --epochs 40

# 批量测试（默认使用 0917 数据）
& $python SNN_based_method/scripts/test.py --checkpoint checkpoints/SNN/train_YYYYMMDD_HHMMSS/best.pth

# 单个 raw group 推理（会保存模型 vs 最大值法的对比图）
& $python SNN_based_method/scripts/test1.py --checkpoint <best.pth> --raw-path <path.raw> --group-index 0 --save-prediction

# Dry-run：预览将生成哪些预计算 label（不写文件）
& $python SNN_based_method/scripts/generate_precomputed_labels.py --data-paths <...> --csv-paths <...> --pages-per-group 128 --dry-run

# 编码可视化（频率响应 / 多帧聚合 / 雾 vs 目标区分度）
& $python SNN_based_method/visualize_encoding.py --t-max 128
```

易踩的几点：
- 续训时的 `--epochs` 是**总目标 epoch 数**，不是"再训练多少轮"。checkpoint 已训到 epoch 20，想再训 20 轮，应传 `--epochs 40`
- `--trace-steps 5` 会打印训练循环中 DataLoader 与 GPU 各自的等待点，调 worker 数之前先跑这个
- `--augment-train` 默认开启，当前默认 `num_aug=2` 且 `keep_original_sample=False`，训练集样本数 = 原始的 2 倍

## 架构概览

系统的任务是从浓雾 SPAD ToF 数据里估计深度与强度。核心思路是学习一个逐光子的 gate，再用该 gate 对原始 ToF 做物理量纲不变的加权矩。

### 单次 forward 流水线

```
raw ToF [B, 4096, P]
  → reshape 为 [P, B, 64, 64]，构建 valid mask（1 ≤ tof ≤ time_threshold）
  → ToF 编码：正弦 [P, B, 17, 64, 64] 或 LUT [P, B, D, 64, 64]
  → Stem（Conv1x1 + BN + PLIF + Conv3x3 + BN）→ [T, B, C, 64, 64]
  → N × SpikeBlock（PLIF → MultiScaleDSConv dilation 1/2/4 → PLIF → Conv1x1 + BN + residual）
  → GateHead → gate [P, B, 1, 64, 64]（逐光子 sigmoid）
  → Gated Moment：
        depth_coarse     = Σ(gate · tof · valid) / Σ(gate · valid)
        intensity_coarse = Σ(gate · valid) / P
        confidence       = weight_sum / (weight_sum + 1)
  → 双分支 SpatialRefineHead（depth_net + intensity_net，输入均为
        [depth_coarse/range, intensity_coarse, confidence]）
  → 输出 [B, 2, 64, 64]  （ch0 = depth，单位 ToF bin；ch1 = intensity，范围 [0,1]）
```

### 设计取舍

- **编码**：使用正弦编码而不是标量 `tof/128`，是为了让网络区分**时间位置**与**幅度**。非均匀频率 `[1,2,4,6,8,12,16,24]` 的精细分辨率比等差方案高 2.3 倍
- **等宽 backbone**（T/C/64×64 全程不变）：不做时间 U-Net —— P 帧是独立采样的，PLIF 沿 P 维自然累积证据
- **Gated Moment**：depth 保留在 ToF bin 的物理量纲里，gate 只负责挑出信号光子 vs 雾后向散射。**严禁**让网络直接从编码特征输出 depth，那会破坏物理量纲
- **双分支精修头**：depth 和 intensity 物理含义不同（时间位置 vs 选中光子占比），共享单头会相互掣肘。两个分支都用 `confidence` 缩放残差，避免低光子区域凭空补结构
- **分块处理**：P 被切为 `chunk_size` 大小的块；chunk 之间调用 `functional.detach_net`（保留膜电位状态、切断 BPTT 图），forward 结束时调用 `functional.reset_net`（防止跨 batch 状态泄漏）。**不要删除 `reset_net`**

### Loss

```
L = 0.3·L_GT + 0.5·L_SSIM + 0.15·L_var + 0.02·L_sparse + 0.03·L_smooth + LUT 正则项
```

- `SSIMLoss`：7×7 高斯窗口，输入通过 `depth_range=128` 归一化到 `[0,1]`
- `ImageMetrics`（MAE/RMSE/SSIM/PSNR）：仅用于评估，不参与梯度
- `w_gt` 与 `w_ssim` 默认不做 mask，因为 label 来自干净数据的弱监督
- LUT 正则项（`w_lut_smooth`、`w_lut_norm`）只在 `encoding_mode="lut"` 时生效

### 预生成 label 池

`use_precomputed_labels=True`（默认）时，训练会自动生成 `<dataset>/label/<pages_per_group>/<class>/<class>_0..4.npy`。规则：
- 只有 `fog_level=0` 的 raw 提供 label
- 每个类别 5 个 label，取自同一个 clean raw 的最后 5 个完整 group
- pool 以 `pages_per_group` 为键 —— 改 P 会生成独立 pool，不会混用
- ToF-shift 增强会同步平移 label 的 depth 通道

## 当前默认值（来自 `SNNConfig`）

| 参数 | 取值 | 说明 |
|---|---|---|
| `time_threshold` | 128 | 超过该值的 ToF 置 0 |
| `pages_per_group` | 128 | 每个样本的 page 数 P |
| `batch_size` | 8 | |
| `grad_accum_steps` | 8 | 等效 batch = 64 |
| `C` | 16 | backbone 通道数 |
| `chunk_size` | 32 | 显存旋钮；P=128 → 4 个 chunk |
| `num_blocks` | 2 | |
| `encoding_mode` | `sinusoidal` | `n_freq=8` → 17 通道 |
| `spike_backend` | `auto` | 优先 cupy，回退 torch |

## 数据布局

```
D:\PYproject\SPADdata\
  0825/                          # 训练集
    0825-group.csv               # 必须含 file_path 列；label 池还需 fog_level + target_class
    label/128/<class>/<class>_0..4.npy
  0826/                          # 训练集
  0917/                          # 测试集（test.py 自动选用）
    917group.csv

产物输出：
  logs/SNN/train_YYYYMMDD_HHMMSS.log
  checkpoints/SNN/train_YYYYMMDD_HHMMSS/{best.pth, last.pth, epoch_XXX.pth, config.json}
  logs/SNN/test_YYYYMMDD_HHMMSS/{config.json, summary.json, predictions/}
```

## 产物约定

训练与评估的所有输出都落在 `SNNConfig.log_dir` 与 `SNNConfig.checkpoint_dir` 指定的路径下。每次运行的配置会 JSON 序列化到对应 run 目录，因此即使丢失启动命令行，run 文件夹本身也能完整复现该次实验。
