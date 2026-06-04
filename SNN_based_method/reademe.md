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
    generate_precomputed_labels.py
                        预生成类别级 label 池
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

CSV 若用于预生成 label，还需要包含：

```text
fog_level
target_class
```

`fog_level=0` 的 raw 被视为 clean label 来源，`target_class` 用于把同类样本映射到同一个 label 池。

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

预生成 label 池：

```text
D:\PYproject\SPADdata\0825\label\128\A\A_0.npy ... A_4.npy
D:\PYproject\SPADdata\0826\label\128\A\A_0.npy ... A_4.npy
```

其中 `128` 是当前 `pages_per_group`。如果训练改成 `--pages-per-group 64`，会检查或生成 `label\64\...`，不会混用不同 P 的 label。

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

## 5. 模型结构与设计理论

当前模型是 `SPADSpikeNet`，目标是在浓雾 SPAD ToF 数据中选择更可能属于目标回波的 photon，再由选中的 photon 估计深度和强度。

整体结构：

```text
[B, 4096, P] raw ToF
  -> reshape 为 [P, B, 64, 64]
  -> valid mask: 1 <= tof <= time_threshold
  -> ToF 编码: sinusoidal [P, B, 17, 64, 64] 或 LUT [P, B, D, 64, 64]
  -> Stem: Conv1x1 + BN + PLIF + Conv3x3 + BN
  -> SpikeBlock x num_blocks
       PLIF -> 多尺度深度可分离卷积 -> PLIF -> Conv1x1 + BN -> residual
  -> GateHead
       PLIF -> Conv1x1 + BN -> PLIF -> Conv1x1 -> sigmoid
  -> gate [P, B, 1, 64, 64]
  -> Gated Moment
       depth_coarse     = sum(gate * tof * valid) / sum(gate * valid)
       intensity_coarse = sum(gate * valid) / P
       confidence       = weight_sum / (weight_sum + 1)
  -> 双头 SpatialRefineHead
       depth_net      精修 depth
       intensity_net  精修 intensity
  -> output [B, 2, 64, 64]
```

核心设计思路：

```text
1. 不直接把所有 photon 做直方图峰值，而是学习一个逐 photon 的 gate。
2. gate 表示当前 ToF 是否更像目标回波，而不是雾后向散射或无效噪声。
3. depth 由原始 ToF 的加权矩计算，避免编码特征本身改变物理量纲。
4. intensity 由被 gate 选中的有效 photon 比例得到，反映目标回波强度。
5. SNN 的膜电位沿 P 维累计证据，适合处理稀疏事件和多次采样 photon。
6. 最后的 CNN refine 只做空间局部残差修正，不替代 Gated Moment 的物理估计。
```

这里的 `P` 是同一个 raw group 内的 page 数。它不是图像高度或宽度，而是每个像素被重复采样的次数。训练时可以通过 `pages_per_group` 改变 `P`，模型 forward 会按实际 `P` 分 chunk 处理。

## 6. 关键模块分析

### 6.1 ToF 编码

默认使用正弦编码：

```text
valid 通道:       1 个
sin/cos 通道:     n_freq * 2 个
默认 n_freq=8:    C_enc = 17
```

设计原因：不能直接把 `tof / time_threshold` 当成单通道强度输入。标量 ToF 会让网络把更大的时间 bin 误解为更强的刺激，而实际任务需要区分“时间位置”。正弦编码把 ToF 变成多频位置特征，使卷积可以组合出不同的时间窗口选择器。

可选 `encoding_mode=lut`。LUT 使用 `nn.Embedding` 学习每个 ToF bin 的特征，表达能力更强，但需要 `w_lut_smooth` 和 `w_lut_norm` 约束相邻 bin 平滑和范数一致，避免小数据下过拟合或相邻 bin 表示剧烈跳变。

### 6.2 Stem

Stem 把编码通道映射到主干通道 `C`：

```text
[T, B, C_enc, H, W]
  -> Conv1x1 + BN
  -> PLIF/LIF/IF
  -> Conv3x3 + BN
  -> [T, B, C, H, W]
```

`Conv1x1` 用于融合 ToF 编码通道，`Conv3x3` 引入局部空间上下文。脉冲神经元位于两层卷积之间，用膜电位对 page 维上的稀疏事件进行累积。

### 6.3 SpikeBlock

每个 `SpikeBlock` 是等宽残差块：

```text
x
  -> spike_in
  -> MultiScaleDSConv(dilation=1/2/4)
  -> spike_mid
  -> Conv1x1 + BN
  -> + residual
```

`MultiScaleDSConv` 使用三路 3x3 depthwise convolution：

```text
dilation=1: 3x3 局部细节
dilation=2: 5x5 等效感受野
dilation=4: 9x9 等效感受野
```

这些 dilation 只作用在空间维，不会跳过 page，也不会改变 P 维顺序。当前实现避免显式拼接 `[B, 3C, H, W]`，用分块 pointwise 权重累加三路结果，降低训练峰值显存。

### 6.4 GateHead 和 Gated Moment

`GateHead` 输出逐 photon 的 soft gate：

```text
gate = sigmoid(logits), shape=[P, B, 1, H, W]
```

gate 不直接输出深度，而是参与加权矩估计：

```text
weight_sum = sum(gate * valid)
depth      = sum(gate * tof * valid) / (weight_sum + 1e-6)
intensity  = weight_sum / P
confidence = weight_sum / (weight_sum + 1)
```

这样保留了 ToF 的物理含义：depth 仍然由原始 ToF bin 计算，gate 只负责选择哪些 photon 更可信。`confidence` 用于表示该像素被选中的 photon 数是否足够，低置信区域不允许 refine 产生过大的残差。

### 6.5 双头精修

当前 refine 已拆成两个分支：

```text
输入: [depth_coarse / depth_range, intensity_coarse, confidence]

depth_net:
  输出 depth 残差, 只修正深度

intensity_net:
  输出 intensity 残差, 只修正强度
```

拆成两个头的原因是 depth 和 intensity 的物理含义不同：depth 是 ToF 位置，intensity 是选中 photon 占比。共享一个输出头容易让两个目标互相牵制；分离后每个分支可以学习自己的局部修正规律。两个分支的残差都会乘以 `confidence`，避免在 photon 很少的位置凭空补结构。

### 6.6 Chunk 状态

大 `P` 会显著增加显存，所以模型按 `chunk_size` 分块：

```text
P=128, chunk_size=32 -> 4 个 chunk
```

chunk 内保留完整脉冲状态和梯度；chunk 间执行：

```text
functional.detach_net(self)
```

这会保留膜电位前向状态，但截断跨 chunk 的反向传播图，降低显存和长序列 BPTT 风险。单次 forward 结束后执行：

```text
functional.reset_net(self)
```

它用于清空脉冲神经元状态，避免不同 batch 之间膜电位串扰。

## 7. 推荐训练命令

直接点击或无参数运行 `train.py` 即可使用默认 0825/0826 训练数据。训练开始前会自动检查当前 `pages_per_group` 对应的 label 池，缺失时先生成，完整存在时直接训练：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py
```

显式指定常用训练参数：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py `
  --pages-per-group 128 `
  --batch-size 8 `
  --grad-accum-steps 8 `
  --num-workers 8 `
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

显式指定训练路径和 CSV：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py `
  --data-paths D:\PYproject\SPADdata\0825 D:\PYproject\SPADdata\0826 `
  --csv-paths D:\PYproject\SPADdata\0825\0825-group.csv D:\PYproject\SPADdata\0826\0826-group.csv `
  --pages-per-group 128 `
  --batch-size 8 `
  --grad-accum-steps 8
```

从训练 run 文件夹继续训练，自动读取其中的 `last.pth` 和 `config.json`：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py `
  --resume-run-dir D:\PYproject\SPAD\checkpoints\SNN\train_YYYYMMDD_HHMMSS `
  --epochs 40
```

也可以直接指定 checkpoint：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py `
  --checkpoint D:\PYproject\SPAD\checkpoints\SNN\train_YYYYMMDD_HHMMSS\last.pth `
  --epochs 40
```

续训时 `--epochs` 表示总目标 epoch，不是额外训练多少轮。例如 checkpoint 已保存到 epoch 20，若要继续训练 20 轮，应设置 `--epochs 40`。

如需定位 DataLoader 或 GPU 等待点：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py --trace-steps 5
```

## 8. 预生成 label 池

默认训练启用 `use_precomputed_labels=True`。训练脚本会在构建 DataLoader 前检查 label 池：

```text
<dataset>\label\<pages_per_group>\<class>\<class>_0.npy
...
<dataset>\label\<pages_per_group>\<class>\<class>_4.npy
```

当前规则：

```text
1. 只使用 CSV 中 fog_level=0 的 raw 作为 label 来源。
2. 每个数据集、每个 target_class 只取一个 clean raw。
3. 每个 class 只取该 clean raw 的最后 5 个完整 group。
4. 每个 label 保存为 float32, shape=(2,64,64)。
5. 训练时同一 target_class 的样本随机抽取这 5 个 label 之一。
6. 若训练增强做 ToF shift，预生成 label 的 depth 通道会同步平移。
```

通道约定：

```text
label[0] = depth, 单位 ToF bin
label[1] = intensity, 范围 [0,1]
```

手动 dry-run 查看将生成哪些 label，不写文件：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\generate_precomputed_labels.py `
  --data-paths D:\PYproject\SPADdata\0825 D:\PYproject\SPADdata\0826 `
  --csv-paths D:\PYproject\SPADdata\0825\0825-group.csv D:\PYproject\SPADdata\0826\0826-group.csv `
  --pages-per-group 128 `
  --dry-run
```

手动生成 label：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\generate_precomputed_labels.py `
  --data-paths D:\PYproject\SPADdata\0825 D:\PYproject\SPADdata\0826 `
  --csv-paths D:\PYproject\SPADdata\0825\0825-group.csv D:\PYproject\SPADdata\0826\0826-group.csv `
  --pages-per-group 128
```

无参数运行 `generate_precomputed_labels.py` 默认是 dry-run，避免误写数据。实际训练不需要手动先运行这个脚本；`train.py` 会自动检查和生成。

## 9. 测试命令

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

使用 `--save-prediction` 时，`test1.py` 会保存模型输出和 `visualize_encoding.py` 中最大值法 baseline 的图片对比：

```text
D:\PYproject\SPAD\logs\SNN\test1_YYYYMMDD_HHMMSS\
  config.json
  summary.json
  images\
    model_depth.png
    model_intensity.png
    max_method_depth.png
    max_method_intensity.png
    model_vs_max_method.png
```

`summary.json` 同时记录模型输出与最大值法的 depth/intensity 差异统计。

## 10. 关键配置

常用数据参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `pages_per_group` | `128` | 每个样本使用的 page 数 `P` |
| `time_threshold` | `128` | 大于该 ToF 的值置 `0` |
| `raw_load_mode` | `group` | 直接读取当前 group, 减少整文件读取开销 |
| `split_ratios` | `0.8,0.2,0.0` | 训练/验证/测试划分 |
| `batch_size` | `8` | 单步 batch |
| `grad_accum_steps` | `8` | 梯度累积, 等效 batch 为 `batch_size * grad_accum_steps` |
| `use_precomputed_labels` | `True` | 优先读取预生成 label 池 |
| `precomputed_label_dir_name` | `label` | label 池父目录名 |
| `precomputed_labels_per_class` | `5` | 每个类别随机抽取的 label 数量 |

常用加速参数：

| 参数 | 推荐 | 说明 |
|---|---:|---|
| `num_workers` | `8` | DataLoader worker 数 |
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
| `C` | `16` | 主干通道数 |
| `chunk_size` | `32` | 时间维分块大小, 影响显存 |
| `num_blocks` | `2` | SpikeBlock 数量 |
| `refine_mid` | `8` | 深度/强度精修头中间通道 |
| `return_sequence` | `True` | 训练 var/sparse loss 时需要 |

## 11. 数据增强

训练增强只作用于训练集。

ToF shift：

```powershell
--augment-train --num-aug 2 --no-keep-original-sample --tof-shift-max 20 --tof-shift-prob 0.9
```

逻辑：

```text
1. 当前默认 augment_train=True, num_aug=2, keep_original_sample=False
2. 默认每个训练样本只保留 2 份增强样本, 不保留 aug_index=0 原始样本
3. 使用 --keep-original-sample 可额外保留原始样本
4. 若 num_aug=2 且保留原始样本, 训练集样本数变为原始训练集的 3 倍
5. 若 num_aug=2 且不保留原始样本, 训练集样本数变为原始训练集的 2 倍
6. 增强发生在原始 raw group 上, 先于 time_threshold 裁剪
7. 对所有非零 ToF 加同一个随机整数 delta, delta 属于 [-20, 20]
8. 增强后小于 1 或大于 time_threshold 的值置 0
9. 输入 group 和 label group 同步 shift
```

PageDropout：

```powershell
--page-dropout --page-dropout-prob 0.1
```

逻辑：随机把整页 raw page 置 `0`，只改变输入 photon 密度，不改变标签。当前默认 `page_dropout=False`，`page_dropout_prob=0`。

Page shuffle：

```powershell
--shuffle-pages
# 或
--page-shuffle
```

逻辑：随机打乱单个样本内部的 `P` 维 page 顺序。标签由未打乱的 group 统计生成，不受 page 顺序影响。

## 12. Loss 和指标

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

## 13. 模型状态

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

## 14. 显存和速度建议

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
