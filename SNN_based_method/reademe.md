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
    label_generate_new.py
                        默认 label_prior 生成器, 基于目标/雾/背景 bin 先验
    generate_precomputed_labels.py
                        旧版 label 目录生成器
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
D:\PYproject\SPADdata\0825\label_prior\640\A\A_0.npy ... A_4.npy
D:\PYproject\SPADdata\0826\label_prior\640\A\A_0.npy ... A_4.npy
D:\PYproject\SPADdata\0825\label_prior_debug\640\summary.csv
D:\PYproject\SPADdata\0826\label_prior_debug\640\summary.csv
```

其中 `640` 是当前默认 `pages_per_group`。如果训练改成 `--pages-per-group 128`，会检查或生成 `label_prior\128\...`，不会混用不同 P 的 label。若把 `precomputed_label_dir_name` 改成 `label`，训练会改用旧版 `label\<P>\...` 目录结构。

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

### 6.7 浓雾三死结与 flow 后端的修复 (2026-07)

浓雾下 (fog_level>=2, 目标每像素 1-2 光子 vs 雾 ~33) 旧结构存在三个互相锁死的问题，
0611 flow run 的表现为: loss 从 epoch 5 起平在 1.19、gate 激活率 0.89 (全开)、
深度锁死雾峰 (bin~40, 与目标 bin~60 差恰好等于 depth MAE 21 bin)。

```text
死结 1  逐光子 gate 在决策时刻看不到直方图形状 (唯一能区分雾/目标的统计量);
        标量 tau=1.5 的膜电位只有 1-2 页记忆, ToF-shift 增强又抹掉绝对 bin 位置
        → gate 唯一稳定解是全开。
死结 2  gated-moment 深度用 hard argmax + 局部质心, 梯度只流进当前峰 (=雾峰)
        ±half_width 邻域, 目标 bin 光子永远拿不到梯度。
死结 3  coarse intensity = peak/P 在浓雾下物理上限 ~0.02-0.2, 够不着 label 的
        0-1 置信量纲 → 识别图糊平。
```

对应修复 (模型侧 `SNN_flow.py`/`SNN_new.py`, 训练侧 `loss.py`):

```text
死结 1a  光子级 gate 直接监督 GatePhotonSupervisionLoss (w_gate_bce, P3-A):
         |tof − d_gt| ≤ radius 的前景光子为正类, 稠密 BCE 梯度直达 gate。
死结 1b  流式上下文注入 (flow_use_stream_context, P3-B): raw 直方图 running
         统计 (雾峰位置/密度/谷后占比/集中度) 4 通道逐 chunk 进 stem;
         只用当前 chunk 之前的页, 严格因果, 无梯度。
死结 1c  plif_mt 多时间尺度 PLIF (spike_mode=plif_mt, P3-C): per-channel tau
         在 [spike_tau, spike_tau_max] 对数均匀初始化, 部分通道可积累数百页
         统计。仅 torch 后端 (cupy kernel 不支持 per-channel tau)。
死结 2   _finalize_gated_peak_maps 改温度 softargmax
         (depth_softargmax_sharpness, 默认 8.0; 0 回退旧口径), 所有后端共享。
死结 3   _ValleyHumpHead v2 的 intensity 改为谷后 hump prominence 与雾峰高度
         的 log 对比度 + 可学习仿射标定, 对 P 和雾级近似不变 (P2)。
```

`_ValleyHumpHead` v2 相对 v1 的鲁棒性修正 (P1):

```text
1. 雾峰/谷位从 raw 直方图估计 (不乘 gate), 与 gate 学选择性彻底解耦;
   hump 质心/峰高仍从 gate 直方图读出 (gate 学好后信噪比更高)。
2. 谷偏移有界化: offset = min + (max-min)·sigmoid(·), 界 [3, 40] bin,
   防塌缩回雾峰/越过目标窗; 物理初始化 offset≈11。
3. 软谷门陡度 beta 可学习 (softplus 保正), 初值 2.0 (v1 的 1.0 太软)。
4. 谷后有限窗口 [valley, valley+hump_window] + 窗内均值基线扣除 →
   prominence, 平坦雾尾不再拉偏质心, 背景像素 prominence≈0。
5. 谷位诊断量 (vh_fog_peak/offset/valley/gate_beta 等) 透出到训练日志。
```

注意: 上述改动使 flow 模型的 stem 输入通道数和 valley_hump 参数结构与旧
checkpoint 不兼容 (0611 及更早的 flow ckpt 本就缺 valley_hump 权重), flow
需要重新训练; `new` 后端权重不受影响, 但 `depth_softargmax_sharpness>0` 会
改变旧 checkpoint 的评估输出, 复现旧实验时置 0。

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

默认训练启用 `use_precomputed_labels=True`、`require_precomputed_labels=True`，且 `precomputed_label_dir_name='label_prior'`。训练脚本会在构建 DataLoader 前检查当前目录名对应的 label 池：

```text
<dataset>\label_prior\<pages_per_group>\<class>\<class>_0.npy
...
<dataset>\label_prior\<pages_per_group>\<class>\<class>_4.npy
```

兼容旧版目录：

```text
<dataset>\label\<pages_per_group>\<class>\<class>_0.npy
...
<dataset>\label\<pages_per_group>\<class>\<class>_4.npy
```

训练前流程：

```text
1. 先根据 `data_paths`、`csv_paths`、`target_class` 和当前 `pages_per_group` 计算本次训练应存在的 label 路径。
2. 若 label 池已完整存在，直接进入训练。
3. 若有缺失，`train.py` 会按 `precomputed_label_dir_name` 自动调用对应生成器补齐：
   - `label_prior` -> `SNN_based_method/scripts/label_generate_new.py`
   - `label` -> `SNN_based_method/scripts/generate_precomputed_labels.py`
4. 若 `require_precomputed_labels=True`，生成后仍缺失才报错，不会直接跳过生成。
5. 若 `require_precomputed_labels=False`，训练允许对仍缺失的样本回退为在线弱标签。
```

默认 `label_prior` 生成规则：

```text
1. 只使用 CSV 中 `fog_level=0` 的 raw 作为候选来源。
2. 对每个 clean raw 的多个 group 按 0825/0826 的目标 bin 先验进行评分。
3. 每个数据集、每个 `target_class` 选取得分最高且通过 mask 面积筛选的前 5 个 group。
4. 每个 label 保存为 `float32`，shape=`(2,64,64)`。
5. 训练时同一 `target_class` 的样本随机抽取这 5 个 label 之一。
6. 若训练增强做 ToF shift，预生成 label 的 depth 通道会同步平移。
7. 生成时还会写出 `label_prior_debug\<P>\summary.csv` 和每个 label 的 `.npz` 诊断文件。
```

默认 `label_prior` 通道约定：

```text
label[0] = depth, 单位 ToF bin
label[1] = confidence, 范围 [0,1]
```

手动 dry-run 查看默认 `label_prior` 将生成哪些 label，不写文件：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\label_generate_new.py `
  --data-paths D:\PYproject\SPADdata\0825 D:\PYproject\SPADdata\0826 `
  --csv-paths D:\PYproject\SPADdata\0825\0825-group.csv D:\PYproject\SPADdata\0826\0826-group.csv `
  --pages-per-group 640 `
  --label-dir-name label_prior `
  --dry-run
```

手动生成默认 `label_prior`：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\label_generate_new.py `
  --data-paths D:\PYproject\SPADdata\0825 D:\PYproject\SPADdata\0826 `
  --csv-paths D:\PYproject\SPADdata\0825\0825-group.csv D:\PYproject\SPADdata\0826\0826-group.csv `
  --pages-per-group 640 `
  --label-dir-name label_prior
```

如需兼容旧版 `label` 目录，可改用：

```powershell
D:\Anaconda3\envs\torchnew\python.exe D:\PYproject\SPAD\SNN_based_method\scripts\generate_precomputed_labels.py `
  --data-paths D:\PYproject\SPADdata\0825 D:\PYproject\SPADdata\0826 `
  --csv-paths D:\PYproject\SPADdata\0825\0825-group.csv D:\PYproject\SPADdata\0826\0826-group.csv `
  --pages-per-group 640 `
  --label-dir-name label
```

`label_generate_new.py` 和 `generate_precomputed_labels.py` 的无参数运行都默认走 dry-run，避免误写数据。实际训练通常不需要手动先运行这些脚本；`train.py` 会按 `precomputed_label_dir_name` 自动检查并生成。

可视化已生成的 label 池：

```powershell
D:\Anaconda3\envs\torchnew\python.exe -c "from SNN_based_method.scripts.vis_label import visualize_labels; visualize_labels(r'D:\PYproject\SPADdata\0825\label_prior', '640', 'K')"
```

`vis_label.py` 会逐个显示 `label[0]` depth 和 `label[1]` confidence；若查看旧版 `label` 目录，则第二通道仍可按旧 intensity 语义理解。该脚本主要用于检查 label 来源、ToF shift 同步平移和不同类别 label 是否异常。

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
| `pages_per_group` | `640` | 每个样本使用的 page 数 `P` |
| `time_threshold` | `128` | 大于该 ToF 的值置 `0` |
| `raw_load_mode` | `group` | 直接读取当前 group, 减少整文件读取开销 |
| `split_ratios` | `0.8,0.2,0.0` | 训练/验证/测试划分 |
| `batch_size` | `2` | 单步 batch |
| `grad_accum_steps` | `8` | 梯度累积, 等效 batch 为 `batch_size * grad_accum_steps` |
| `use_precomputed_labels` | `True` | 优先读取预生成 label 池 |
| `require_precomputed_labels` | `True` | 预生成 label 缺失时先自动补齐, 仍缺失才报错 |
| `precomputed_label_dir_name` | `label_prior` | 默认 label 池父目录名 |
| `precomputed_labels_per_class` | `5` | 每个类别随机抽取的 label 数量 |
| `num_aug` | `2` | 每个训练样本额外生成的增强份数 |

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
| `model_backend` | `flow` | 可选 `flow` / `new` / `ann_gate` / `frame_photon` / `rnn` / `lstm` / `gru` |
| `spike_backend` | `cupy` | 仅脉冲后端使用, 可选 `auto` / `cupy` / `torch`; `plif_mt` 节点内部强制 torch |
| `spike_mode` | `plif` | 脉冲神经元类型, 可选 `if` / `lif` / `plif` / `plif_mt` (per-channel 多时间尺度) |
| `spike_tau` | `1.5` | LIF/PLIF 膜时间常数初值; `plif_mt` 模式下作为 tau 下界 |
| `spike_tau_max` | `128.0` | 仅 `plif_mt`: per-channel tau 对数初始化上界 |
| `spike_v_threshold` | `0.8` | 脉冲发放阈值 |
| `spike_v_reset` | `0.0` | 脉冲重置电位, `None` 表示 soft reset |
| `encoding_mode` | `sinusoidal` | ToF 编码, 可选 `lut` |
| `C` | `16` | 主干通道数 |
| `chunk_size` | `128` | 时间维分块大小, 影响显存和长序列 BPTT |
| `num_blocks` | `1` | 时序主干块数量; flow 后端默认提升为 2 |
| `refine_mid` | `8` | 深度/强度精修头中间通道 |
| `depth_softargmax_sharpness` | `8.0` | gated-moment 深度的 softargmax 锐度; 0 回退 hard argmax |
| `return_sequence` | `True` | 训练 var/sparse/gate_bce loss 时需要 |

仅 flow 后端的参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `flow_use_stream_context` | `True` | raw 直方图 running 统计 4 通道注入 stem (严格因果) |
| `flow_use_valley_hump` | `True` | 谷后 hump 物理检测头 |
| `flow_valley_spatial_pool` | `5` | 直方图空间聚合核 (奇数) |
| `flow_valley_offset_min` | `3.0` | 谷偏移下界 (bin) |
| `flow_valley_offset_max` | `40.0` | 谷偏移上界 (bin) |
| `flow_valley_offset_init` | `11.0` | 谷偏移物理初始化 (bin) |
| `flow_valley_gate_beta_init` | `2.0` | 软谷门陡度初值 (可学习) |
| `flow_valley_hump_window` | `48.0` | 谷后搜峰窗口长度 (bin) |
| `flow_state_detach_interval` | `0` | 流式累积量 TBPTT 截断间隔 (chunk 数) |

新增 loss 参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `w_gate_bce` | `0.5` | 光子级 gate BCE 直接监督权重 |
| `gate_bce_bin_radius` | `4.0` | 光子伪标签正类窗口半径 (bin) |
| `gate_bce_pos_weight` | `0.0` | 正类权重; <=0 表示 batch 内自动配平 (clamp [1,50]) |

## 11. 数据增强

训练增强只作用于训练集。

ToF shift：

```powershell
--augment-train --num-aug 2 --no-keep-original-sample --tof-shift-max 20 --tof-shift-prob 0.9
```

逻辑：

```text
1. 当前默认 augment_train=True, num_aug=2, keep_original_sample=False
2. 默认每个训练样本保留 2 份增强样本, 不保留 aug_index=0 原始样本
3. 使用 --keep-original-sample 可额外保留原始样本
4. 若 `keep_original_sample=True`, 训练集样本数变为原始训练集的 `num_aug + 1` 倍
5. 若 `keep_original_sample=False`, 训练集样本数变为原始训练集的 `num_aug` 倍
6. 增强发生在原始 raw group 上, 先于 time_threshold 裁剪
7. 对所有非零 ToF 加同一个随机整数 delta, delta 属于 `[-tof_shift_max, tof_shift_max]`
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
  + w_coarse * L_coarse
  + w_gate_bce * L_gate_bce
  + LUT 正则项
```

当前默认权重：

```text
w_gt=0.55
w_depth_reg=2.0
w_ssim=0.25
w_var=0.0        # 已停用: 浓雾下逼 gate 做单光子选择不可行, 且与 valley_hump 冲突
w_sparse=0.0     # 已停用: 与 GT/intensity 项内耗, gate 稀疏性由 gate_bce 直接监督替代
w_smooth=0.03
w_coarse=0.20
w_gate_bce=0.5   # 光子级 gate 直接监督 (死结 1/P3-A)
w_lut_smooth=0.01
w_lut_norm=0.005
```

`L_gate_bce` (`GatePhotonSupervisionLoss`) 用 label depth 构造逐光子伪标签：
前景像素内 `|tof − d_gt| ≤ gate_bce_bin_radius` 的有效光子为正类，其余有效光子
为负类，带类不平衡自动配平的 BCE。它给 gate 稠密梯度，不再依赖浓雾下 30:1
雾/目标质量比里近乎为零的矩比值梯度；伪标签随 ToF-shift 增强同步平移。
需要 `return_sequence=True`。

`GT L1` 和 `SSIM` 默认不使用 mask，因为标签来自较干净数据的弱监督图。loss 和指标内部会把 depth 除以 `depth_range` 归一化到 `[0,1]` 计算；输出图里的 depth 仍是 ToF bin。若要把验证日志中的 depth MAE 换算成 bin 误差，可近似乘以 `depth_range`，当前默认是 `128`。

日志中 `train_items` 和 `val_items` 会记录各个 loss 分项，`val_metrics` 会记录 depth/intensity 的 MAE、RMSE、SSIM、PSNR。flow + valley_hump 模式下还会记录
`vh_fog_peak_mean / vh_offset_mean / vh_valley_mean / vh_gate_beta /
vh_hump_peak_mean / vh_fog_height_mean`（谷位诊断，观察偏移是否塌缩/漂移）和
`gate_pos_rate / gate_neg_rate / gate_pos_frac`（光子级 gate 判别质量）。

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

## 16. 低空无人机探测：SPAD-可见光融合理论

本章是面向后续工作的理论说明，记录把当前 SPAD ToF 成像方法应用到低空无人机探测、
并与可见光相机融合的动机和可行性判断，不涉及具体实现。本章的结论与第 5 章一致：
SNN 在本任务中的价值定位在流式低延迟与能效，而非在 GPU 上和 ANN/递推网络比单帧精度。

### 16.1 应用动机

低空无人机探测的难点不在算力，而在目标本身：尺寸小、对比度低、常贴着复杂地物背景，
且实际场景里经常叠加雾、霾、扬尘或弱光。单一可见光相机在这些退化条件下同时受两重限制：
没有距离信息，无法靠深度把小目标从背景里分离；在雾/弱光下纹理和对比度又会快速退化。

SPAD ToF 与可见光在能力上几乎正交：

```text
SPAD ToF:
  优势  主动照明 + 光子计时, 可在雾中做距离门控, 把目标按距离从背景剥离
        直接输出深度, 对低对比度小目标的"存在性"判断更鲁棒
  限制  空间分辨率低 (当前 64x64), 单帧光子稀疏, 纹理/语义信息弱

可见光相机:
  优势  高空间分辨率, 丰富纹理与外观, 利于目标分类和精细定位
  限制  无深度; 雾/弱光下对比度退化; 小目标在复杂背景中易漏检
```

因此融合的目标是：用 SPAD 的雾穿透与测距能力解决"有没有、在多远"，用可见光的高分辨率
解决"是什么、在哪儿的精细位置"，在退化大气条件下得到比任一单模态都更稳的小目标
检测与测距。这是融合的高层动机，也是判断后续每一步是否值得做的标准。

### 16.2 为什么是 SNN 流式

第 5 章已经说明：在重复曝光的 P 轴上、用直方图化输入、在 GPU 上比单帧精度，SNN 没有
结构性优势。融合场景重新定义了比较的轴，让 SNN 的长处第一次有了落点。

```text
1. SPAD 以 25000 frame/s 输出稀疏光子帧, 这是事件驱动 SNN 的原生输入形态,
   不需要先攒满一个积分窗再处理。
2. 流式模型 (model_backend=flow, SNNFlowNet) 的估计随光子累积逐步细化,
   可在任意时刻读出当前深度/强度图, 不必等整段 P 帧到齐。
3. 对运动目标和运动平台, 低延迟与"随时可读"比离线精度更重要。
4. 价值定位是低延迟 + 恒定显存 + 神经形态硬件上的潜在能效, 而非 GPU 精度。
```

流式还有一个对融合不可或缺的正确性前提——因果性：第 t 帧的估计只能依赖第 1..t 帧。
`SNNFlowNet` 已经把这一点做实（见 `model/SNN_flow.py`）：通过把时间相关的归一化层
替换为对时间维和 batch 维都无依赖的归一化，训练和推理在任意分块粒度下都严格因果。
没有这个前提，"实时融合"只是离线对齐的假象。

### 16.3 跨模态时序对齐

两个传感器的帧率差约两个数量级，这是融合时序设计的出发点：

```text
SPAD:    25000 frame/s
可见光:  60 frame/s
比值:    约 417 : 1, 即每个可见光帧的曝光窗口内约有 417 个 SPAD 帧
```

流式 SNN 与这个比值天然契合：把落在某个可见光帧曝光窗口内的 SPAD 帧持续喂入流式状态，
在该可见光帧的时间戳处做一次 `stream_readout`，就得到一张与这一可见光帧时间对齐的
SPAD 深度/强度图。换言之，流式读出的节奏可以直接对齐到可见光快门，而不需要把 SPAD
当成离线批处理后再去和可见光硬凑时间。

时序对齐之外还需要空间配准：两个传感器视场、分辨率、光轴都不同，需要标定外参并把
一个模态投影到另一个模态的像平面。空间配准是工程前提，本章不展开。

### 16.4 融合层次与可行性

按融合发生的阶段，经典上分三层。结合当前两模态的分辨率与时序差异，对可行性做高层判断：

```text
数据级 / 早期融合:
  直接拼原始像素或光子。受 64x64 与可见光高分辨率、模态量纲差异制约, 直接拼接代价高、
  收益不确定。可行性: 低。

特征级 / 中期融合:
  各自抽特征后在对齐的空间/时间上融合。SPAD 侧用流式输出的深度/强度/置信度图,
  可见光侧用其特征, 在配准后的网格上融合。最能利用两模态互补性, 且天然容纳分辨率与
  时序差异 (用置信度和时间戳对齐加权)。可行性: 高, 推荐优先验证。

决策级 / 后期融合:
  各自独立出检测结果再合并。实现简单、鲁棒, 但丢失跨模态早期线索 (如"可见光弱响应区
  恰好有 SPAD 近距离回波")。可作为基线或退化方案。可行性: 中。
```

初步判断是以特征级融合为主线：SPAD 流式分支按可见光帧节奏输出对齐的深度/强度/置信度，
与可见光特征在配准网格上融合；置信度图为融合提供逐像素权重，在雾重区域自然提高 SPAD
深度的话语权，在清晰区域让可见光纹理主导。这与现有模型已经输出 `confidence` 是一致的，
不需要为融合另起一套不兼容的设计。以上为方向性判断，具体网络与融合算子留待后续实现。

### 16.5 待验证问题

本章是动机与可行性，不是已验证结论。落地前至少需要回答：

```text
1. 标定与配准: SPAD 与可见光的外参标定精度, 能否支撑特征级逐像素融合。
2. 分辨率贡献: 64x64 的 SPAD 深度在可见光高分辨率之上是否带来可测量的检测增益,
   还是仅在雾重区域有效。
3. 时序一致性: 运动目标下, 曝光窗口内累积的 SPAD 估计与可见光帧的时间错配是否引入
   拖影或配准漂移。
4. 流式优势的真实性: 低延迟/恒定显存的优势是否在真实采集链路与硬件上成立,
   而不只是离线仿真的结论。
5. 能效论证: 神经形态硬件上的能效优势需实测 spike 数/SOPs, 不能只靠理论外推。
6. 模态主导边界: 在不同雾浓度与光照下, 各模态贡献的切换点在哪里, 融合权重应如何随
   置信度自适应。
```

这些问题的答案决定融合方案是否值得投入，也决定 SNN 流式这条路线在无人机探测里的
真实价值边界。
