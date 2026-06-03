# SPAD SNN 训练说明

本文档只记录当前 `SNN_based_method` 的实际用法。模型、数据加载、loss、日志和 checkpoint 均由 `SNNConfig` 统一管理。

## 1. 当前入口

```text
SNN_based_method/
  SNN_config.py         配置入口
  SNN.py                兼容旧导入, 转发到 SNN_new.py
  SNN_new.py            当前模型, 使用官方 spikingjelly.activation_based
  loss.py               成像 loss 和指标
  scripts/
    train.py            训练入口
    test.py             批量测试入口
    test1.py            单 raw group 推理入口
    data.py             raw/csv 数据集与 DataLoader
    augment.py          raw group 级数据增强
    runtime.py          CLI、日志、checkpoint、公用运行时工具
```

本项目使用环境中安装的官方 `spikingjelly.activation_based`，不使用本地 `spikingjelly1`。

## 2. 数据路径

默认训练数据只使用：

```text
D:\PYproject\SPADdata\0825
D:\PYproject\SPADdata\0826
```

默认训练 CSV：

```text
D:\PYproject\SPADdata\0825\0825-group.csv
D:\PYproject\SPADdata\0826\0826-group.csv
```

默认测试数据只使用：

```text
D:\PYproject\SPADdata\0917
D:\PYproject\SPADdata\0917\917group.csv
```

CSV 需要包含 `file_path` 列。训练脚本在无 `--data-paths` 参数时自动使用 0825/0826；测试脚本在未显式指定测试路径时自动切到 0917。

## 3. 输出位置

训练日志：

```text
D:\PYproject\SPAD\logs\SNN\train_YYYYMMDD_HHMMSS.log
```

训练 checkpoint：

```text
D:\PYproject\SPAD\checkpoints\SNN\train_YYYYMMDD_HHMMSS\
  best.pth
  last.pth
  epoch_XXX.pth
  config.json
```

测试结果：

```text
D:\PYproject\SPAD\logs\SNN\test_YYYYMMDD_HHMMSS\
  config.json
  summary.json
  predictions\*.npy    # 仅 --save-predictions 时生成
```

训练日志只输出一个 `.log` 文件。开头记录主要配置，之后每个 epoch 输出：

```text
train_loss / val_loss / best_val_loss / lr
train_items
val_items
val_metrics
checkpoint 保存路径
```

## 4. 数据形状

```text
raw group:      [4096, P]
DataLoader:     frames=[P, B, 1, 64, 64]
model input:    [B, 4096, P]
model output:   [B, 2, 64, 64]
label:          [B, 2, 64, 64]
```

输出通道：

```text
ch0 = depth, 单位为 ToF bin, 范围 [0, time_threshold]
ch1 = intensity, 范围 [0, 1]
```

`time_threshold` 默认是 `128`。输入中小于 `1` 或大于 `time_threshold` 的 ToF 会置 `0`，`0` 表示无效 photon。

## 5. 推荐训练命令

直接使用默认 0825/0826 训练：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py `
  --pages-per-group 128 `
  --batch-size 4 `
  --grad-accum-steps 8 `
  --num-workers 6 `
  --persistent-workers `
  --prefetch-factor 4 `
  --raw-load-mode group `
  --cuda-prefetch `
  --pin-memory `
  --progress-interval 50 `
  --tf32 `
  --cudnn-benchmark `
  --spike-backend auto
```

显式指定训练路径：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py `
  --data-paths D:\PYproject\SPADdata\0825 D:\PYproject\SPADdata\0826 `
  --csv-paths D:\PYproject\SPADdata\0825\0825-group.csv D:\PYproject\SPADdata\0826\0826-group.csv `
  --pages-per-group 128 `
  --batch-size 4 `
  --grad-accum-steps 8
```

从 checkpoint 继续训练：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py `
  --checkpoint D:\PYproject\SPAD\checkpoints\SNN\train_YYYYMMDD_HHMMSS\last.pth
```

如需定位 DataLoader 或 GPU 等待点：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py --trace-steps 5
```

## 6. 测试命令

批量测试默认使用 0917：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\test.py `
  --checkpoint D:\PYproject\SPAD\checkpoints\SNN\train_YYYYMMDD_HHMMSS\best.pth
```

保存预测结果：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\test.py `
  --checkpoint D:\PYproject\SPAD\checkpoints\SNN\train_YYYYMMDD_HHMMSS\best.pth `
  --save-predictions
```

单个 raw group 推理：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\test1.py `
  --checkpoint D:\PYproject\SPAD\checkpoints\SNN\train_YYYYMMDD_HHMMSS\best.pth `
  --raw-path D:\PYproject\SPADdata\0917\example.raw `
  --group-index 0 `
  --save-prediction
```

## 7. 关键配置

常用数据参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `pages_per_group` | `128` | 每个样本使用的 page 数 `P` |
| `time_threshold` | `128` | 大于该 ToF 的值置 `0` |
| `raw_load_mode` | `group` | 直接读取当前 group, 减少整文件读取开销 |
| `split_ratios` | `0.8,0.2,0.0` | 训练/验证/测试划分 |
| `batch_size` | `4` | 单步 batch |
| `grad_accum_steps` | `8` | 梯度累积, 等效 batch 为 `batch_size * grad_accum_steps` |

常用加速参数：

| 参数 | 推荐 | 说明 |
|---|---:|---|
| `num_workers` | `4` 到 `8` | DataLoader worker 数 |
| `persistent_workers` | 开 | 减少 epoch 间 worker 重建 |
| `prefetch_factor` | `4` | 每个 worker 预取 batch 数 |
| `pin_memory` | 开 | 加速 CPU 到 CUDA 拷贝 |
| `cuda_prefetch` | 开 | 用独立 CUDA stream 预取下一批 |
| `tf32` | 开 | Ampere/Ada GPU 上加速卷积和矩阵计算 |
| `cudnn_benchmark` | 开 | 固定输入尺寸下选择更快卷积实现 |
| `progress_interval` | `20` 到 `50` | 降低进度条导致的 CPU-GPU 同步 |

常用模型参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `model_backend` | `new` | 官方 `activation_based` 实现 |
| `spike_backend` | `auto` | CUDA 可用时优先 cupy, 否则 torch |
| `encoding_mode` | `sinusoidal` | ToF 编码, 可选 `lut` |
| `C` | `32` | 主干通道数 |
| `chunk_size` | `32` | 时间维分块大小, 影响显存 |
| `num_blocks` | `2` | SpikeBlock 数量 |
| `refine_mid` | `8` | 深度/强度精修头中间通道 |
| `return_sequence` | `True` | 训练 var/sparse loss 时需要 |

## 8. 数据增强

训练增强只作用于训练集。

ToF shift：

```powershell
--augment-train --tof-shift-max 15 --tof-shift-prob 1.0
```

逻辑：

```text
1. 在原始 raw group 上操作, 先于 time_threshold 裁剪
2. 对所有非零 ToF 加同一个随机整数 delta, delta 属于 [-15, 15]
3. 增强后小于 1 或大于 time_threshold 的值置 0
4. 输入 group 和 label group 同步 shift
```

PageDropout：

```powershell
--page-dropout --page-dropout-prob 0.1
```

逻辑：随机把整页 raw page 置 `0`，只改变输入 photon 密度，不改变标签。

Page shuffle：

```powershell
--shuffle-pages
# 或
--page-shuffle
```

逻辑：随机打乱单个样本内部的 `P` 维 page 顺序。标签由未打乱的 group 统计生成，不受 page 顺序影响。

## 9. Loss 和指标

训练 loss：

```text
L = w_gt * L_GT
  + w_ssim * L_SSIM
  + w_var * L_var
  + w_sparse * L_sparse
  + w_smooth * L_smooth
  + LUT 正则项
```

当前默认权重：

```text
w_gt=0.3
w_ssim=0.5
w_var=0.15
w_sparse=0.02
w_smooth=0.03
w_lut_smooth=0.01
w_lut_norm=0.005
```

`GT L1` 和 `SSIM` 默认不使用 mask，因为标签来自较干净数据的弱监督图。loss 和指标内部会把 depth 除以 `depth_range` 归一化到 `[0,1]` 计算；输出图里的 depth 仍是 ToF bin。若要把验证日志中的 depth MAE 换算成 bin 误差，可近似乘以 `depth_range`，当前默认是 `128`。

日志中 `train_items` 和 `val_items` 会记录各个 loss 分项，`val_metrics` 会记录 depth/intensity 的 MAE、RMSE、SSIM、PSNR。

## 10. 模型状态

`SPADSpikeNet.forward()` 内部保留两类状态操作：

```text
functional.detach_net(self)  chunk 之间截断梯度, 保留状态前向延续
functional.reset_net(self)   单次 forward 结束后重置脉冲神经元状态
```

不要删除 `reset_net`。它用于避免不同 batch 之间脉冲神经元膜电位串扰。

当前精修头已经拆成两个分支：

```text
depth_net      精修 depth
intensity_net  精修 intensity
```

两个分支都使用 coarse depth、coarse intensity 和 confidence 作为输入，并由 confidence 控制残差幅度。

## 11. 显存和速度建议

优先调这些参数：

```text
pages_per_group: 64 / 128 / 256
chunk_size:      16 / 32 / 64
batch_size:      2 / 4 / 8
grad_accum_steps 根据 batch_size 调整等效 batch
```

如果 CUDA 利用率呈锯齿状，通常优先检查：

```text
1. DataLoader 是否等待 raw 读取: 增大 num_workers, 使用 raw_load_mode=group
2. CPU 到 GPU 拷贝是否等待: 开 pin_memory 和 cuda_prefetch
3. 进度条是否频繁同步: 增大 progress_interval
4. 当前训练是否已经重启: 代码修改后旧进程不会自动使用新逻辑
5. 是否有其它进程占用显存或 GPU
```

如果 OOM，优先降低 `batch_size` 或 `chunk_size`。如需进一步降低显存，可尝试 `--amp`，但需要观察 loss 和 SSIM 是否稳定。
