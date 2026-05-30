# SPAD 浓雾场景 SNN 成像模型

## 1. 概述

```
SPAD 时间戳 [B, 4096, P]
  → 正弦位置编码 (17 通道, 消除 ToF 幅度偏差)         [1][2]
  → 等宽 SpikeBlock 堆叠 (T / C / 64×64 全程不变)
  → PLIF 膜电位沿帧累积光子证据
  → EchoGate 逐帧目标光子选择
  → Gated Moment 深度/强度粗估计
  → 置信度精修头 (残差 CNN, 消除逐像素 gate 噪声)
  → [B, 2, 64, 64]

[1] Mildenhall B, et al. NeRF. CACM, 2022, 65(1): 99-106.
[2] Tancik M, et al. Fourier Features. NeurIPS, 2020.
```

## 2. 数据格式

```
输入: [B, 4096, P]   P 帧, 每元素为 ToF 整数 (0=无效, 1~150=有效)
编码: [P, B, C, 64, 64]   C 为编码后 timebin 维度
输出: [B, 2, 64, 64]   ch0=深度(timebin), ch1=强度(目标光子占比)
GT:   [B, 2, 64, 64]   fog_level=0 末段直方图峰值 (含噪 → 弱监督)
```

**P 维是无序集合而非时序**（独立采样，相互可交换），需避免学到"序列顺序偏置"。可选策略：

1. 训练时随机打乱 P 维，强制学习 permutation-invariant 表征。
2. 增加 histogram 分支：逐像素统计 ToF 为 `[B, 150, 64, 64]`，与 SNN 分支融合。
3. 用 DeepSets / Set Transformer / pooling aggregation 替代纯因果 PLIF。
4. 若保留 PLIF，可用 forward + backward 双向 PLIF 降低"只看过去帧"偏置。

## 3. 编码方案：位置编码

### 3.1 为什么不用标量 tof/150

标量编码下，ToF 的"位置信息"被误当作"强度信息"——PLIF 会把 `tof=93` 视为比 `tof=42` 更强的刺激，且单个线性 filter 对 ToF 只能单调响应，无法实现窗口选择。

### 3.2 正弦编码

```python
def encode_tof(tof, valid, n_freq=8, t_max=150):
    v = valid.float().unsqueeze(1)
    t = (tof.float() / t_max).unsqueeze(1) * v
    channels = [v]
    for i in range(n_freq):
        freq = (i + 1) * 3.14159
        channels.append(torch.sin(freq * t) * v)
        channels.append(torch.cos(freq * t) * v)
    return torch.cat(channels, dim=1)   # [B, 17, H, W]
```

特性：`n_freq=8` → 17 通道 (1 valid + 8 sin + 8 cos)，值域 `[-1,1]`；相邻 ToF 编码相似、远 ToF 正交、各 ToF 编码能量相同。Conv 通过 sin/cos 线性组合可选出任意 ToF 区间。编码 `[P,B,1,64,64] → [P,B,17,64,64]` 后直接在 channel 维做 Conv2d。**深度值由 Gated Moment 用原始 ToF 计算，编码只服务于分类。**

### 3.3 频率选取消融

等差频率 `[1..8]` 是 NeRF 标准做法，但 SPAD 浓雾有两个约束：
- **大尺度区分**：雾后向散射 (bin≈40) vs 目标回波 (bin≈60)，差 20 bin → gate 需准确分类。
- **精细分辨**：目标区域 bin∈[55,65] 内深度恢复需相邻 bin 可区分。

在相同通道数（8 对 = 17 通道）下对比 5 种方案：

```
评估 (t_max=150, fog_bin=40, target_bin=60):
  cos: 余弦相似度 (越接近 0 越正交 → 越好区分)
  L2:  欧氏距离 (越大越好区分)
  adj: 相邻 bin 平均 L2 (分辨率, 越大越好)

方案                            | 雾vs目标(差20) | 近雾(差4)    | 近目标(差10)  | adj分辨率
─────────────────────────────────────────────────────────────────────────────────────
A: [1,2,3,4,5,6,7,8]  (等差)    | c=-0.05 L=4.4 | c=+0.92 L=1.2 | c=+0.58 L=2.8 | 0.299
B: [1,2,4,6,8,12,16,24] (非均匀)| c=+0.12 L=4.0 | c=+0.66 L=2.5 | c=+0.25 L=3.7 | 0.689
C: [1,2,3,4,6,8,12,16] (混合A)  | c=+0.25 L=3.7 | c=+0.81 L=1.8 | c=+0.31 L=3.5 | 0.481
D: [1,2,3,4,5,6,8,12]  (混合B)  | c=+0.09 L=4.0 | c=+0.89 L=1.4 | c=+0.47 L=3.1 | 0.362
E: [1,2,3,5,8,12,16,24] (混合C) | c=+0.20 L=3.8 | c=+0.67 L=2.5 | c=+0.29 L=3.6 | 0.683
```

**分析**：A 大尺度最优 (cos=-0.05) 但相邻分辨率仅 0.30，精细恢复最弱；B 精细分辨率为 A 的 2.3 倍 (0.69)，大尺度 cos=+0.12 仍远低于 0.5 混淆门限；E 接近 B 且保留 freq=3 使低频过渡更平滑；C/D 折中无突出优势。

**结论**：选用方案 B `freqs=[1,2,4,6,8,12,16,24]`。

```
1. 瓶颈不在大尺度: 雾vs目标差 20 bin, 所有方案 L2∈[3.7,4.4], gate 均能区分
2. 瓶颈在精细分辨: 目标区 bin∈[55,65] 深度恢复需高分辨率, B(0.689) ≫ A(0.299)
3. 通道数不变: 仍 8 频率对 → 17 通道, 不增参数/显存
4. 最高频 24π 的 Nyquist 分辨率 ≈ 150/(2×24) ≈ 3.1 bin, 覆盖精细距离需求
```

可视化：`visualize_encoding.py --target_bin 60`（频率响应、区分度、雾/目标编码对比）。

### 3.4 可学习 LUT 编码消融

**动机**：正弦编码的频率结构人工预设，无法适应数据分布。若让编码完全数据驱动，能否更优？

**方案**：用可学习查表 `nn.Embedding(151, embed_dim, padding_idx=0)` 替代正弦公式，将整数 ToF bin 直接映射为特征向量。LUT 与正弦编码解耦，**embed_dim 可自由调整 (8/16/32)**，stem 层自动适配输入通道。

```python
# index 0 (invalid) → 全零, 不参与梯度; index 1~150 → 可学习向量
model = SPADSpikeNet(encoding_mode="lut", embed_dim=16, lut_init="rbf")
```

训练时由成像损失反向传播，**仅当前 batch 访问到的 bin 被更新**（稀疏梯度）。

**初始化策略**：

```
sinusoidal  用 sin/cos 值填充 (截断/补零适配 embed_dim); 与固定正弦同起点, 最稳定
rbf         embed_dim 个高斯中心均匀分布在 [1,t_max], σ=0.8×间距 (~60%重叠); 独立对照
random      N(0,0.5), index 0 保持全零; 基线, 验证正则能否稳定训练
```

**稳定性约束**：裸 LUT 易过拟合/不稳定（低 bin 更新过快、高 bin 样本稀少易发散、相邻 bin 无约束可能剧烈跳变），故配合正则：

```
约束              实现                                         权重
─────────────────────────────────────────────────────────────────
invalid 映射      padding_idx=0 → 全零, 不参与梯度             内建
valid mask        编码后 × valid                              内建
相邻平滑 L_adj    mean(||emb[i+1]-emb[i]||²), i∈[1,149]       w=0.01
范数一致 L_norm   var(||emb[i]||)                             w=0.005

L = L_imaging + 0.01·L_adj + 0.005·L_norm
其中 L_imaging = L_GT + L_SSIM + L_var + L_sparse + L_smooth
```

**参数开销**：

```
配置                        参数量    LUT参数   输出形状
──────────────────────────────────────────────────────────
sinusoidal (n_freq=8)       26,026    0         [T,B,17,H,W]
lut D=16 (任意 init)        28,410    2,416     [T,B,16,H,W]
lut D=32 init=rbf           31,338    4,832     [T,B,32,H,W]

LUT 参数 = 151 × embed_dim; stem Conv(embed_dim→C,1) 自动适配
```

**与正弦编码本质区别**：

```
维度          正弦编码                  LUT 编码
─────────────────────────────────────────────────────
频率结构      人工预设                  数据驱动学习
输出维度      固定 2·n_freq+1 (=17)     embed_dim 可调
相邻 bin 关系 sin/cos 连续性天然保证    需 L_adj 显式约束
参数量        0                         151 × embed_dim
泛化性        对未见 ToF 天然泛化       依赖初始化+正则
训练稳定性    稳定                      需正则防发散
表达能力      受限于预设频率            理论上可学任意映射
```

**PLIF 时间维**：LUT 输出 `[T,B,embed_dim,H,W]`，膜电位沿 T 累积不受影响。`padding_idx=0` 保证无效帧输入全零（等价于正弦编码的 valid mask 清零），chunk 间 detach 机制不变。

**预期消融结果**：

| 场景 | 正弦 | LUT | 说明 |
|------|------|-----|------|
| 小数据 (<1k) | 更稳定 | 可能过拟合 | 自由度高, 正则不足 |
| 大数据 (>10k) | 受限预设频率 | 有望更优 | 数据足以驱动最优映射 |
| 多场景泛化 | 天然泛化 | 需充分覆盖 | 连续性是优势 |
| embed_dim 扫描 | 固定 17 | 可调 8/16/32 | 维度 vs 参数平衡 |
| 初始化对比 | — | sin>rbf>random | 验证先验对收敛影响 |

## 4. 网络结构

### 4.1 结构图

```
[B, 4096, P]  (训练 P=120, 测试 P=60/240/500)
        │ reshape → [P, B, 64, 64], split by chunk_size
        ▼
per chunk [T,B,64,64]: valid mask (tof∈[1,t_max])
  → encode: sinusoidal [T,B,17,H,W] 或 LUT [T,B,D,H,W]
        │
        ▼ Stem:   Conv1×1 → BN → PLIF → Conv3×3 → BN          → [T,B,C,64,64]
        ▼ SpikeBlock × num_blocks (residual):
              PLIF → MSDSConv → PLIF → Conv1×1 → BN → +id     → [T,B,C,64,64]
        ▼ EchoGate: PLIF → Conv(C/2) → BN → PLIF → Conv1 → σ  → gate [T,B,1,64,64]∈[0,1]
        │ (所有 chunk 共享权重并累加)
        ▼ Gated Moment on original ToF:
              depth_coarse     = Σ(gate·tof·valid) / Σ(gate·valid)
              intensity_coarse = Σ(gate·valid) / P
              confidence       = weight_sum / (weight_sum + 1)
        ▼ Confidence-gated refine head:
              normalize[depth/t_max, intensity] + confidence
              → Conv3×3 → BN → ReLU → Conv3×3 → residual·confidence
              → clamp(depth∈[0,t_max], intensity∈[0,1])
        ▼ output [B, 2, 64, 64]
```

**精修头作用**：
- Gated Moment 逐像素独立计算，gate 随机误差引入空间噪声。
- 精修头用 3×3 邻域平滑 gate 误差，同时保留目标边缘；残差连接使深度仍以 Gated Moment 为基础，只做微调。
- 极小（两层 Conv, <1K 参数），不增显存。
- `confidence = weight_sum/(weight_sum+1)`：置信度越低残差越小，避免有效光子稀少处过度平滑或凭空补结构。
- 内部 depth 归一化到 `[0,1]` 与 intensity/confidence 同量纲建模，输出再还原 depth 到 `[0,t_max]`。
- 可扩展置信度特征：`selected_count`、`selected_var`、`valid_count`、`gate_entropy`。

### 4.2 存在问题

需监控 firing rate，否则可能退化为"带 PLIF 的 ANN"（因 Fourier 特征含负值）。

### 4.3 关键组件

**MSDSConv**（多尺度膨胀深度可分离卷积）：
```
三路并行 DW-Conv 3×3 (dilation=1,2,4) → cat(3C) → PW-Conv(3C→C) → BN
等效感受野 3×3 / 5×5 / 9×9, 空间全程 64×64
```
注意：dilation 是**空间维**空洞率（配 padding 后输出仍 64×64），不抽样时间维 P，不跳过 SPAD pages，不改变 chunk 内 PLIF 时间步。

**Chunked 处理**（P > chunk_size）：
```
帧 1~128 → 等宽SNN ─膜电位(detach)→ 帧 129~256 → ... → 帧 ...~500
            (有梯度)                  (有梯度)          (有梯度)
              └────────── 累加 Gated Moment ──────────┘ → depth, intensity

权重共享; 膜电位跨 chunk 延续但梯度截断; 显存 = 1 个 chunk
```

### 4.4 数据形状 (chunk=128, B=8, C=32)

```
层                  形状                空间
──────────────────────────────────────────
编码输入            [1024, 17, 64, 64]  64×64
Stem                [1024, 32, 64, 64]  64×64
SpikeBlock 1~3      [1024, 32, 64, 64]  64×64
EchoGate            [1024,  1, 64, 64]  64×64
全程 T / C / 空间不变
```

## 5. 损失函数与评估指标

### 5.1 训练损失

各项 loss 先统一量纲再加权：

```
depth_norm     = clamp(depth/depth_range, 0, 1), depth_range = t_max = 150
intensity_norm = clamp(intensity, 0, 1)
tof_norm       = tof / depth_range
```

注意：此处仅指 loss/refine 内部量纲统一，**模型编码仍接收原始 ToF bin**（valid mask、LUT index、Gated Moment 都依赖整数 bin 语义）。

```
L = w_gt·L_GT + w_ssim·L_SSIM + w_var·L_var + w_sparse·L_sparse
  + w_smooth·L_smooth + w_lut_smooth·L_lut_smooth + w_lut_norm·L_lut_norm
```

| loss | 量纲 | 目的 |
|------|------|------|
| `L_GT` | `[0,1]` 上 depth/intensity L1 | 弱标签锚点 |
| `L_SSIM` | `[0,1]`, data_range=1 | 局部结构相似性 |
| `L_var` | 归一化 ToF 上方差超额 `relu(var-(σ/depth_range)²)` | gate 选中光子应集中 |
| `L_sparse` | `mean(gate·valid)` | 目标光子在浓雾中应稀疏 |
| `L_smooth` | 归一化 depth 上 `\|grad d\|·exp(-β\|grad I\|)` | 边缘保持平滑 |
| `L_lut_*` | embedding 正则 | LUT 编码稳定性 |

**loss 系数之和无需为 1**：它们是拉格朗日乘子/优化偏好，不是概率权重。关键是每项数值尺度稳定可解释，再据验证集调系数；强行归一可能削弱关键物理约束（如 `L_var`）。

```
默认: w_gt=0.3, w_ssim=0.1, w_var=1.0, w_sparse=0.05, w_smooth=0.1
      w_lut_smooth=0.01, w_lut_norm=0.005
```

`L_var` 需重点监控：权重过大时模型可能只选最尖锐的一小段回波而非完整回波。建议记录 `selected_count`、`weighted_var` 分布、`mean(gate·valid)`。

### 5.2 评估指标

`ImageMetrics`（不参与梯度），对 depth 和 intensity 各计算，均归一化到 `[0,1]`（depth/150）后：

```
MAE   = mean(|pred-gt|)            像素级绝对误差
RMSE  = sqrt(mean((pred-gt)²))     对大误差敏感
SSIM  = 结构相似性 (7×7 高斯窗)    局部结构质量
PSNR  = 10·log10(1/MSE) dB         data_range=1.0
```

### 5.3 存在问题：奖励作弊风险

```
L_var:    选中 photon 的 ToF 方差要小
L_sparse: gate × valid 要稀疏
```
若 gate 全接近 0，或每像素只选 1 个 photon → 方差≈0、sparse≈小 → 两项均"满意"但结果退化。需监控：`mean(gate)`、`mean(gate·valid)`、每像素选中 photon 数、gate 直方图、denominator=`Σ(gate·valid)`。

### 5.4 P 维 shuffle 对比实验

P 维来自独立采样帧，更接近无序集合。PLIF 沿 P 累积膜电位，若 raw page 顺序带采集系统偏置，模型可能学到不该学的顺序模式。

| 组 | 训练 | 测试 | 目的 |
|----|------|------|------|
| A | `shuffle_pages=False` | 原始顺序 | 基线 |
| B | `shuffle_pages=True` | 原始顺序 | 强制顺序不敏感 |
| C | `shuffle_pages=True` | 多次 shuffle 取均值/方差 | 测排列敏感性 |

```powershell
python .\SNN\train.py --data-paths data\raw --pages-per-group 120 --run-name p120_no_shuffle
python .\SNN\train.py --data-paths data\raw --pages-per-group 120 --shuffle-pages --run-name p120_shuffle
python .\SNN\test.py  --data-paths data\raw --pages-per-group 120 --checkpoint SNN\artifacts\p120_shuffle\best.pth
```

**判断标准**：
1. B 在正常测试集上不应明显劣于 A。
2. 同一样本不同 P 排列下，输出 MAE/RMSE 或像素方差应更低。
3. 若 B 明显更稳 → 任务更接近 independent frame set，应默认开启 `--shuffle-pages`。
4. 若 A 明显更好 → 顺序中可能含物理/系统信息，需谨慎解释并考虑双向 PLIF / Set 分支。

注意：`shuffle_pages=True` 只打乱输入 frames，弱标签仍由未打乱 group 统计生成，标签不受影响。

## 6. 参数与显存

| 参数 | 默认 | 说明 |
|------|------|------|
| C | 32 | 通道数 (全程不变) |
| chunk_size | 128 | 每 chunk 帧数 |
| pages_per_group | 500 | 每样本 SPAD pages 数 P; 可改 120 训练 |
| n_freq | 8 | 正弦频率数 → 17 通道 |
| spike_mode | "plif" | 神经元类型 |
| t_max | 150 | 最大有效 ToF bin |

### 6.1 训练 P=120，测试可变 P

模型不把 P 写死：`forward` 读取实际 P，按 chunk_size 切块并累加所有 chunk 的 Gated Moment。只要网络结构参数一致，测试可换 `pages_per_group`。

```
训练: pages_per_group=120, chunk_size=60/120
测试: pages_per_group=60/120/240/500 均可
```

```powershell
python .\SNN\train.py --data-paths data\raw --pages-per-group 120 --chunk-size 120 --run-name train_p120
python .\SNN\test.py  --data-paths data\raw --pages-per-group 500 --chunk-size 120 --checkpoint SNN\artifacts\train_p120\best.pth --run-name test_p500
python .\SNN\test1.py --raw-path data\raw\sample.raw --pages-per-group 240 --checkpoint SNN\artifacts\train_p120\best.pth
```

注意：
1. P 增大 → 每样本聚合更多独立帧，噪声更低，但样本数减少。
2. chunk_size 只决定单次进入 SNN 的时间步与显存，不要求等于 P。
3. 膜电位跨 chunk 延续、梯度截断；测试无梯度时主要影响状态累计路径。
4. 训练 P=120、测试 P=500 属分布变化，建议报告 P=60/120/240/500 曲线而非单点。

### 6.2 显存估计

```
经验公式 (fp16 + gradient checkpoint, C=32): 显存 ≈ chunk_size × B × 6.5 MB

24GB GPU 推荐: chunk=128, B=8 (≈6.5 GB)
目标光子/chunk: chunk=128 约 12 个, chunk=500 约 47 个
```

## 7. 配置管理 (SNNConfig)

所有可调参数由 `SNN_config.py` 的 `SNNConfig` dataclass 管理，模型/损失/指标均经它构建：

```python
from SNN_config import SNNConfig

cfg = SNNConfig()                                            # 默认正弦
cfg = SNNConfig(encoding_mode="lut", embed_dim=16, lut_init="rbf")  # LUT (相关参数自动激活)
cfg = SNNConfig.load("experiment_01.json")                  # 从 JSON 加载

model     = cfg.build_model()      # SPADSpikeNet
criterion = cfg.build_loss()       # SPADImagingLoss
metrics   = cfg.build_metrics()    # ImageMetrics
print(cfg.summary())
cfg.save("experiment_01.json")
```

**参数分组与自动激活**：

```
分组          参数                                  sinusoidal  lut
─────────────────────────────────────────────────────────────────
数据          pages_per_group, shuffle_pages        ✓          ✓
数据          t_max / time_threshold                ✓          ✓
编码(公共)    encoding_mode, n_freq                 ✓          ✓
编码(LUT)     embed_dim, lut_init, lut_max_norm     ✗          ✓
网络          C, chunk_size, spike_mode, num_blocks ✓          ✓
损失(公共)    w_gt, w_ssim, w_var, w_sparse,w_smooth ✓         ✓
损失(LUT)     w_lut_smooth, w_lut_norm              ✗          ✓
损失超参      sigma_target, rho_target, beta_smooth ✓          ✓
```

预置：`SINUSOIDAL_DEFAULT, LUT_RBF_16, LUT_SIN_16, LUT_RBF_32`。

标准入口：
```powershell
python .\SNN\train.py --config experiment.json
python .\SNN\test.py  --checkpoint SNN\artifacts\train_xxx\best.pth --data-paths data\raw
python .\SNN\test1.py --checkpoint SNN\artifacts\train_xxx\best.pth --raw-path data\raw\one.raw --group-index 0
```

## 8. SNN_new.py — activation_based API 版本

`SNN_new.py` 将后端从旧版 `clock_driven` 迁移到 `activation_based`，对外接口（输入/输出形状、参数名）与 `SNN.py` 完全一致，可直接替换。

### 8.1 API 差异

| 维度 | SNN.py (clock_driven) | SNN_new.py (activation_based) |
|------|----------------------|-------------------------------|
| 导入 | `spikingjelly1.clock_driven.neuron` | `spikingjelly.activation_based.neuron/functional` |
| 神经元 | `MultiStepParametricLIFNode(timestep=T)` | `neuron.ParametricLIFNode(step_mode='m')` |
| 时间步 | 构造传 `timestep`, 内部展开 | `step_mode='m'` 多步模式 |
| 输入形状 | `[T*B, C, H, W]` | `[T, B, C, H, W]` |
| ANN 子模块 | 直接调用 (已展平) | `functional.seq_to_ann_forward(x, module)` |
| chunk 截断 | 手动 `m.v = m.v.detach()` | `functional.detach_net(self)` |
| 网络重置 | `functional.reset_net(self)` | 同 (接口不变) |

### 8.2 结构变化

Stem/GateHead 提取为独立模块 `_Stem`/`_GateHead`，内部经 `seq_to_ann_forward` 处理时间维：

```
_Stem.forward(x: [T,B,C_enc,H,W]):
    x = seq_to_ann_forward(x, Conv+BN)   # [T,B,C,H,W]
    x = spike(x)                          # PLIF step_mode='m'
    x = seq_to_ann_forward(x, Conv+BN)
```
`SpikeBlock` 同理：ANN 子模块经 `seq_to_ann_forward` 包装，脉冲神经元直接接收 `[T,B,C,H,W]`。

### 8.3 使用方式

```python
from SNN_based_method.SNN_new import SPADSpikeNet
model = SPADSpikeNet(C=32, chunk_size=128, spike_mode="plif")
model = SPADSpikeNet(encoding_mode="lut", embed_dim=16, lut_init="rbf")
out = model(raw_data)         # raw_data [B,4096,P] → out["output"] [B,2,64,64]
```

### 8.4 环境要求

```
SNN.py     → spikingjelly1 (本地 clock_driven 副本, env: pytorch)
SNN_new.py → spikingjelly  (activation_based, env: torchnew)
两文件各自环境独立运行, 互不依赖
```

### 8.5 cupy backend 自动检测

模块加载时自动探测 cupy，结果缓存到 `_CUPY_AVAILABLE`：可用 → 神经元用 `backend='cupy'`，否则回退 `backend='torch'`。探测逻辑：实际构造 `IFNode(backend='cupy')` 跑一次前向确认端到端可用（仅 `import cupy` 不够，缺 CUDA headers 时 import 成功但运算会失败）。

环境配置 (torchnew)：

| 依赖 | 版本 | 说明 |
|------|------|------|
| cupy-cuda12x | 14.1.0 | `pip install cupy-cuda12x` |
| pytest | ≥9.0 | cupy.testing 依赖 |
| CUDA Toolkit | 12.8 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` |

`CUDA_PATH`：`conda run` 子进程可能未继承新增环境变量，`_probe_cupy_backend()` 在其缺失时自动兜底默认路径。

spikingjelly 开发版兼容修复：
```
activation_based              → 缩进对齐
activation_based/neuron/base_node.py → 文件头加 from abc import abstractmethod
```

---

## 9. 文件说明

```
SNN_based_method/
  reademe.md            本文档
  SNN_config.py         统一配置 (SNNConfig + 预置 + JSON)
  SNN.py                模型 — clock_driven API (env: pytorch)
  SNN_new.py            模型 — activation_based API (env: torchnew)
  loss.py               训练损失 + 评估指标
  visualize_encoding.py 编码可视化工具
```

## 10. 显存占用测试

测试方法（`run_5d_memory_benchmark`，SNN_new v1）：

```
设备显存上限: 12.0 GB
cupy backend: available
输入: [T, B, C, 64, 64]
网络路径: stem → SpikeBlocks → gate_head → temporal aggregation → refine
模式: forward + backward
扫描: 对 (C, B) 网格逐步增大 T
停止规则: 当 peak_allocated > 12 GB 时, 跳过当前 (B,C) 更大的 T
记录指标: input 显存 / peak_allocated / peak_reserved, 标注 PASS / OVER_12GB / OOM
```

> 逐行测试结果略（运行脚本即可复现）。

## 11. baseline 对比

1. 传统 histogram peak / matched filter
2. 2D CNN on histogram `[150,64,64]`
3. 3D CNN / 1D ToF Conv + 2D spatial Conv
4. 同结构 PLIF → ReLU/GELU 的 ANN 版本
5. scalar `tof/150` 编码
6. one-hot / histogram 编码
7. sinusoidal vs LUT
8. 有无 confidence-gated refine head；refine 前后是否 clamp
9. 有无 `L_var` / `L_sparse` / `L_smooth`
10. P shuffle 训练 vs 不 shuffle
11. train P=120, test P=60/120/240/500 可变 P 泛化曲线

