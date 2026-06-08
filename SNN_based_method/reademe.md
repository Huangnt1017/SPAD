# SPAD SNN 训练说明

本文档只记录当前 `SNN_based_method` 的实际用法。模型、数据加载、loss、日志和 checkpoint 均由 `SNNConfig` 统一管理。

仓库根目录下的 `baseline/`、`model/`、`scripts/`、`utils/` 是点云分类/3D box 工程；`SNN_based_method/` 是独立的 SPAD ToF 成像子项目。当前 SNN 训练只共享仓库根路径、`logs/SNN`、`checkpoints/SNN` 和外部数据目录 `D:\PYproject\SPADdata`，不调用顶层 `scripts/train.py`，也不使用本地 `spikingjelly1/`。

## 1. 当前入口

```text
SNN_based_method/
  config/
    SNN_config.py       配置入口
  model/
    SNN_new.py          默认 SNN 后端, 使用官方 spikingjelly.activation_based
    ANN_gated_moment.py 非脉冲 ANN gate-moment baseline
    SNN_c_RNN.py        显式 RNN 等价版
    SNN_c_LSTM.py       显式 ConvLSTM 版
    SNN_c_GRU.py        显式 ConvGRU 版
  utils/
    loss.py             成像 loss 和指标
    data.py             raw/csv 数据集与 DataLoader
    augment.py          raw group 级数据增强
    runtime.py          CLI、日志、checkpoint、公用运行时工具
  scripts/
    train.py            训练入口
    train_lif_to_plif.py
                        LIF checkpoint 迁移到 PLIF 微调入口
    test.py             批量测试入口
    test1.py            单 raw group 推理入口
    run_experiment_grid.py
                        对比实验矩阵生成/执行入口
    collect_experiment_results.py
                        实验结果 CSV/JSON 汇总入口
    analyze_spike_and_tau.py
                        PLIF tau、参数量和 spike 指标分析入口
    visualize_encoding.py
                        ToF 编码可视化入口
    generate_precomputed_labels.py
                        预生成类别级 label 池
    vis_label.py        预生成 label 池可视化入口
  experiments/
    chapter_grid.json   论文章节对比实验矩阵示例
```

本项目使用环境中安装的官方 `spikingjelly.activation_based`，不使用本地 `spikingjelly1`。

当前可通过 `SNNConfig.model_backend` 或命令行 `--model-backend` 选择 5 个后端：

```text
new      : 默认后端, 官方 activation_based SNN
ann_gate : 非脉冲 ANN gate-moment baseline
rnn      : 显式神经元递推等价版
lstm     : 显式 ConvLSTM 版
gru      : 显式 ConvGRU 版
```

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

工程工具运行后会生成 `artifacts` 目录，例如：

```text
D:\PYproject\SPAD\SNN_based_method\artifacts\
  chapter4_results.csv
  chapter4_results.json
  chapter4_tau_spike.csv
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

当前默认模型是 `SPADSpikeNet`（`model_backend=new`）。此外，同一套输入输出协议还提供
`ANNGatedMomentNet`、`SNN_c_RNN`、`SNN_c_LSTM` 和 `SNN_c_GRU` 四个对比后端。
`ann_gate` 是非脉冲 ANN gate-moment baseline，用于区分性能收益来自 gate-moment
结构本身，还是来自 SNN/PLIF 的脉冲时序机制。除 `ann_gate` 外，其余后端共享同一套
ToF 编码、空间卷积主干、gate 聚合和精修头，差异主要集中在时间递推核心。

整体结构：

```text
[B, 4096, P] raw ToF
  -> reshape 为 [P, B, 64, 64]
  -> valid mask: 1 <= tof <= time_threshold
  -> ToF 编码: sinusoidal [P, B, 17, 64, 64] 或 LUT [P, B, D, 64, 64]
  -> 时序 Stem
       new : Conv1x1 + BN + PLIF/LIF/IF + Conv3x3 + BN
       ann_gate: Conv1x1 + BN + GELU + Conv3x3 + BN + GELU
       rnn : Conv1x1 + BN + 显式神经元递推 + Conv3x3 + BN
       lstm: Conv1x1 + BN + ConvLSTM + Conv3x3 + BN
       gru : Conv1x1 + BN + ConvGRU + Conv3x3 + BN
  -> 时序主干块 x num_blocks
       new : SpikeBlock
       ann_gate: ANNGatedBlock
       rnn : SpikeBlockRNN
       lstm: LSTMBlock
       gru : GRUBlock
  -> 时序 GateHead
       new : PLIF -> Conv1x1 + BN -> PLIF -> Conv1x1 -> sigmoid
       ann_gate: Conv1x1 + BN + GELU -> Conv1x1 -> sigmoid
       rnn : 显式神经元递推 -> Conv1x1 + BN -> 显式神经元递推 -> Conv1x1 -> sigmoid
       lstm: ConvLSTM -> Conv1x1 + BN -> ConvLSTM -> Conv1x1 -> sigmoid
       gru : ConvGRU -> Conv1x1 + BN -> ConvGRU -> Conv1x1 -> sigmoid
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

不同后端的时间建模差异：

```text
new      : 通过脉冲神经元膜电位和 reset 隐式保存时序状态
ann_gate : 不使用脉冲状态, 对每个 page 逐帧提取特征并输出连续 gate
rnn      : 把 IF/LIF/PLIF 的膜电位更新显式展开成 RNN hidden state
lstm     : 用显式 ConvLSTM 的 (h_t, c_t) 状态建模时间依赖
gru      : 用显式 ConvGRU 的 h_t 状态建模时间依赖
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

以下先以默认 `new` 后端为例说明。Stem 把编码通道映射到主干通道 `C`：

```text
[T, B, C_enc, H, W]
  -> Conv1x1 + BN
  -> PLIF/LIF/IF
  -> Conv3x3 + BN
  -> [T, B, C, H, W]
```

`Conv1x1` 用于融合 ToF 编码通道，`Conv3x3` 引入局部空间上下文。默认 SNN 后端把
脉冲神经元放在两层卷积之间，用膜电位对 page 维上的稀疏事件进行累积。`rnn`、`lstm`
和 `gru` 后端则在同一位置分别使用显式 RNN / ConvLSTM / ConvGRU 单元。

### 6.3 SpikeBlock

默认后端的每个 `SpikeBlock` 是等宽残差块：

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

显式递推后端保持同样的多尺度卷积与残差结构，只把块内两处脉冲节点替换为对应的
RNN / ConvLSTM / ConvGRU 单元，因此不同后端之间的大部分差异都集中在时间状态更新。

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
P=128, chunk_size=64 -> 2 个 chunk
```

`new` 后端在 chunk 内保留完整脉冲状态和梯度；chunk 间执行：

```text
functional.detach_net(self)
```

这会保留膜电位前向状态，但截断跨 chunk 的反向传播图，降低显存和长序列 BPTT 风险。单次 forward 结束后执行：

```text
functional.reset_net(self)
```

它用于清空脉冲神经元状态，避免不同 batch 之间膜电位串扰。

`rnn`、`lstm`、`gru` 后端不依赖 `spikingjelly.functional.reset_net`。它们把时序状态显式保存在
本次 forward 的局部 `state` 变量中，chunk 间通过递归 `detach` 截断 BPTT，forward 结束后
自然释放，不会跨 batch 残留。

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

切换显式时序后端：

```powershell
# 非脉冲 ANN gate-moment baseline
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py --model-backend ann_gate

# 显式 RNN 等价版
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py --model-backend rnn

# 显式 ConvLSTM 版
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py --model-backend lstm

# 显式 ConvGRU 版
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py --model-backend gru
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

可视化已生成的 label 池：

```powershell
D:\Anaconda3\envs\torchnew\python.exe -c "from SNN_based_method.scripts.vis_label import visualize_labels; visualize_labels(r'D:\PYproject\SPADdata\0825\label', '128', 'K')"
```

`vis_label.py` 会逐个显示 `label[0]` depth 和 `label[1]` intensity，用于检查 clean label 来源、ToF shift 同步平移和不同类别 label 是否异常。

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
| `num_aug` | `1` | 每个训练样本额外生成的增强份数 |

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
| `model_backend` | `new` | 可选 `new` / `ann_gate` / `rnn` / `lstm` / `gru` |
| `spike_backend` | `cupy` | 仅 `new` 后端使用, 可选 `auto` / `cupy` / `torch` |
| `spike_mode` | `plif` | 脉冲神经元类型, 可选 `if` / `lif` / `plif` |
| `spike_tau` | `2.0` | LIF/PLIF 膜时间常数, PLIF 用它初始化可学习 tau |
| `spike_v_threshold` | `0.8` | 脉冲发放阈值 |
| `spike_v_reset` | `0.0` | 脉冲重置电位, `None` 表示 soft reset |
| `encoding_mode` | `sinusoidal` | ToF 编码, 可选 `lut` |
| `C` | `16` | 主干通道数 |
| `chunk_size` | `64` | 时间维分块大小, 影响显存和长序列 BPTT |
| `num_blocks` | `1` | 时序主干块数量 |
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
1. 当前默认 augment_train=True, num_aug=1, keep_original_sample=False
2. 默认每个训练样本只保留 1 份增强样本, 不保留 aug_index=0 原始样本
3. 使用 --keep-original-sample 可额外保留原始样本
4. 若 num_aug=1 且保留原始样本, 训练集样本数变为原始训练集的 2 倍
5. 若 num_aug=1 且不保留原始样本, 训练集样本数变为原始训练集的 1 倍增强样本
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
  + w_depth_reg * L_depth_reg
  + w_ssim * L_SSIM
  + w_var * L_var
  + w_sparse * L_sparse
  + w_smooth * L_smooth
  + LUT 正则项
```

当前默认权重：

```text
w_gt=0.6
w_depth_reg=0.5
w_ssim=0.25
w_var=0.2
w_sparse=0.01
w_smooth=0.02
w_lut_smooth=0.01
w_lut_norm=0.005
```

`GT L1` 和 `SSIM` 默认不使用 mask，因为标签来自较干净数据的弱监督图。loss 和指标内部会把 depth 除以 `depth_range` 归一化到 `[0,1]` 计算；输出图里的 depth 仍是 ToF bin。若要把验证日志中的 depth MAE 换算成 bin 误差，可近似乘以 `depth_range`，当前默认是 `128`。

日志中 `train_items` 和 `val_items` 会记录各个 loss 分项，`val_metrics` 会记录 depth/intensity 的 MAE、RMSE、SSIM、PSNR。

## 13. 模型状态

不同后端的状态管理方式：

```text
new:
  functional.detach_net(self)  chunk 之间截断梯度, 保留状态前向延续
  functional.reset_net(self)   单次 forward 结束后重置脉冲神经元状态

rnn / lstm / gru:
  显式 state 变量在 chunk 间递推
  通过递归 detach 截断跨 chunk BPTT
  forward 结束后状态自然释放, 不跨 batch 持久化

ann_gate:
  不保存脉冲膜电位或显式循环状态
  每个 chunk 内逐 page 共享 ANN 卷积权重, 输出连续 gate
```

对 `new` 后端，不要删除 `reset_net`。它用于避免不同 batch 之间脉冲神经元膜电位串扰。

当前精修头已经拆成两个分支：

```text
depth_net      精修 depth
intensity_net  精修 intensity
```

两个分支都使用 coarse depth、coarse intensity 和 confidence 作为输入，并由 confidence 控制残差幅度。

## 14. 显存和速度建议

当前 `new` 后端仍是非流式训练图：虽然 forward 内按 `chunk_size` 分块，并在 chunk 间
`detach_net`，但最终 `depth/intensity` loss 仍通过 `weighted_sum/weight_sum` 追溯到每个
chunk；当 `return_sequence=True` 时还会保留完整 `gate/tof/valid` 用于 `var/sparse` loss。
因此当前实现的显存主要随 `batch_size * pages_per_group` 增长，`chunk_size` 只带来小幅差异，
还没有达到“显存主要由 `batch_size * chunk_size * C * 64 * 64` 决定”的流式形态。

下面表格用于 12GB 显卡上的配置选择。测量脚本：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\profile_snn_memory.py
```

测量/估算口径：

```text
模型: SPADSpikeNet, C=16, num_blocks=1, spike_mode=plif, spike_backend=cupy, return_sequence=True
步骤: 随机输入的一次 forward + SPADImagingLoss + backward + AdamW step
显存: PyTorch max_memory_reserved, 单位 GiB, 更接近训练时 CUDA allocator 占用
实测: 预测不超过 11.5 GiB 的组合实际运行; 高风险组合不实测
估算: OOM 括号内为按 P=128 基线线性外推的 reserved 显存
判定: 为避免 Windows 共享 GPU 内存拖慢, profiling 脚本用 11.5 GiB 作为严格上限;
      11.0-11.5 GiB 记为“边界”, >11.5 GiB 记为 OOM
```

### 14.1 当前非流式显存表

`chunk_size=32`：

| pages_per_group P | B=2 | B=4 | B=8 |
|---:|---:|---:|---:|
| 128 | 1.52 | 3.03 | 5.96 |
| 384 | 4.55 | 8.85 | OOM (17.89) |
| 480 | 5.68 | 11.04 边界 | OOM (22.36) |
| 640 | 7.56 | OOM (15.15) | OOM (29.81) |
| 960 | 11.31 边界 | OOM (22.72) | OOM (44.72) |
| 1000 | OOM (11.84) | OOM (23.67) | OOM (46.59) |
| 1200 | OOM (14.21) | OOM (28.40) | OOM (55.90) |
| 2400 | OOM (28.42) | OOM (56.80) | OOM (111.80) |

`chunk_size=64`：

| pages_per_group P | B=2 | B=4 | B=8 |
|---:|---:|---:|---:|
| 128 | 1.56 | 3.10 | 6.20 |
| 384 | 4.51 | 8.98 | OOM (18.60) |
| 480 | 5.93 | OOM (11.64) | OOM (23.25) |
| 640 | 7.40 | OOM (15.52) | OOM (31.01) |
| 960 | OOM (11.67) | OOM (23.28) | OOM (46.51) |
| 1000 | OOM (12.16) | OOM (24.25) | OOM (48.45) |
| 1200 | OOM (14.59) | OOM (29.10) | OOM (58.14) |
| 2400 | OOM (29.19) | OOM (58.19) | OOM (116.27) |

`chunk_size=128`：

| pages_per_group P | B=2 | B=4 | B=8 |
|---:|---:|---:|---:|
| 128 | 1.71 | 3.43 | 6.75 |
| 384 | 4.62 | 9.31 | OOM (20.25) |
| 480 | 5.96 | OOM (12.86) | OOM (25.31) |
| 640 | 7.54 | OOM (17.15) | OOM (33.75) |
| 960 | OOM (12.83) | OOM (25.72) | OOM (50.62) |
| 1000 | OOM (13.37) | OOM (26.79) | OOM (52.73) |
| 1200 | OOM (16.04) | OOM (32.15) | OOM (63.28) |
| 2400 | OOM (32.08) | OOM (64.31) | OOM (126.56) |

### 14.2 当前可用配置建议

12GB 卡上，当前非流式实现建议优先用：

```text
P=384, chunk_size=32/64, batch_size=4
P=640, chunk_size=32/64, batch_size=2
P=960, chunk_size=32, batch_size=2    # 边界配置, 训练前先关其它占显存进程
```

不建议直接训练：

```text
P>=1200 的任意 B=2/4/8
P>=640 且 B>=4
P>=384 且 B=8
```

如果需要 `P=1200/2400` 或希望 `P=960` 下使用更大的 batch，必须先把模型改成流式统计
或两遍式 streaming backward，否则仅调 `chunk_size` 不能根本解决显存随 P 增长的问题。

常规调参优先级：

```text
pages_per_group: 384 / 640 / 960
chunk_size:      32 优先, 64 次之, 128 只在吞吐明显更好时使用
batch_size:      2 / 4
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

如果 OOM，优先降低 `batch_size` 或 `pages_per_group`。当前非流式实现里，降低 `chunk_size`
只能小幅降低峰值，不能把显存复杂度从 O(P) 变成 O(chunk)。如需进一步降低显存，可尝试
`--amp`，但需要观察 loss 和 SSIM 是否稳定。

## 15. 论文章节对比实验与工具

本节用于组织博士论文章节级别的对比实验。建议围绕同一套输入输出协议展开:

```text
主实验:
  ann_gate      非脉冲 ANN gate-moment baseline
  new + IF      无泄漏脉冲基线
  new + LIF     固定膜时间常数 SNN
  new + PLIF    可学习膜时间常数 SNN
  rnn/lstm/gru  显式时序递推基线

训练策略:
  LIF 直接训练
  PLIF 直接训练
  LIF -> PLIF fine-tune
  LIF -> PLIF fine-tune + freeze_plif_epochs

Loss 消融:
  Full loss
  - w_var
  - w_sparse
  - w_smooth
  - w_ssim
  --no-return-sequence + w_var=0 + w_sparse=0

效率分析:
  pages_per_group: 64 / 128 / 256
  chunk_size:      16 / 32 / 64 / 128
  spike_backend:   cupy / torch
  amp:             on / off
```

### 15.1 ANN gate baseline

`ann_gate` 后端位于 `model/ANN_gated_moment.py`。它保留 ToF 编码、连续 gate、Gated Moment 和
SpatialRefineHead, 但不使用 IF/LIF/PLIF 脉冲神经元。该 baseline 用于回答:

```text
1. 性能提升是否仅来自 gate-moment 物理聚合结构?
2. SNN/PLIF 的脉冲时序状态是否带来额外收益?
3. 非脉冲连续 gate 在相同 loss 下的上限和代价如何?
```

训练示例:

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train.py `
  --model-backend ann_gate `
  --run-name chapter4_ann_gate `
  --pages-per-group 128 `
  --chunk-size 64 `
  --C 16 `
  --num-blocks 1
```

### 15.2 LIF -> PLIF fine-tune

`scripts/train_lif_to_plif.py` 用于从 LIF checkpoint 迁移到 PLIF。它只加载同名同形状模型权重,
PLIF 新增的 `*.w` tau 参数由 `--spike-tau` 初始化, 不恢复 LIF optimizer/scheduler。

先 dry-run 检查迁移:

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train_lif_to_plif.py `
  --lif-checkpoint D:\PYproject\SPAD\checkpoints\SNN\chapter4_snn_lif\last.pth `
  --epochs 10 `
  --lr 2e-4 `
  --spike-tau 2.0 `
  --dry-run
```

正式微调:

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\train_lif_to_plif.py `
  --lif-checkpoint D:\PYproject\SPAD\checkpoints\SNN\chapter4_snn_lif\last.pth `
  --epochs 10 `
  --lr 2e-4 `
  --spike-tau 2.0 `
  --freeze-plif-epochs 2 `
  --spike-backend cupy `
  --tf32 `
  --cudnn-benchmark `
  --cuda-prefetch
```

### 15.3 实验矩阵工具

`scripts/run_experiment_grid.py` 读取 JSON 实验表, 生成或顺序执行训练命令。示例配置:

```text
SNN_based_method/experiments/chapter_grid.json
```

只打印命令:

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\run_experiment_grid.py `
  --spec D:\PYproject\SPAD\SNN_based_method\experiments\chapter_grid.json `
  --dry-run
```

实际顺序执行:

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\run_experiment_grid.py `
  --spec D:\PYproject\SPAD\SNN_based_method\experiments\chapter_grid.json `
  --execute
```

建议先用 `--dry-run` 确认 `run-name`、数据路径、模型后端、loss 消融参数都正确, 再执行完整矩阵。

### 15.4 结果汇总工具

`scripts/collect_experiment_results.py` 扫描 checkpoint run 目录和 test summary, 输出 CSV/JSON:

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\collect_experiment_results.py `
  --checkpoint-root D:\PYproject\SPAD\checkpoints\SNN `
  --log-root D:\PYproject\SPAD\logs\SNN `
  --pattern "chapter4_*" `
  --output D:\PYproject\SPAD\SNN_based_method\artifacts\chapter4_results.csv
```

输出字段包含:

```text
config.*             训练配置
metrics.train_loss   checkpoint 中记录的训练 loss
metrics.val_loss     checkpoint 中记录的验证 loss
metrics.val_metrics.* depth/intensity 的 MAE/RMSE/SSIM/PSNR
summary.metrics.*    test.py / test1.py 生成的测试指标
```

### 15.5 可解释性分析工具

`scripts/analyze_spike_and_tau.py` 从 checkpoint 读取 PLIF tau、参数量和已记录 spike rate:

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\analyze_spike_and_tau.py `
  --checkpoint-root D:\PYproject\SPAD\checkpoints\SNN `
  --pattern best.pth `
  --output D:\PYproject\SPAD\SNN_based_method\artifacts\chapter4_tau_spike.csv
```

PLIF tau 的换算:

```text
tau = 1 / sigmoid(w)
```

建议在论文中至少报告:

```text
1. tau_mean / tau_std / tau_min / tau_max
2. parameter_count / trainable_parameter_count
3. spike_rate_all / spike_rate_stem / spike_rate_blocks / spike_rate_gate
4. depth_rmse / depth_ssim 与 tau、spike_rate 的对应关系
5. ann_gate、LIF、PLIF、LIF->PLIF 在精度与训练代价上的对比
```

`ann_gate` 没有 `*.w` 和 spike_rate, 这本身就是对照信息: 它说明非脉冲连续 gate 在相同
Gated Moment 框架下的表现, 与 SNN/PLIF 的脉冲状态机制分开比较。
