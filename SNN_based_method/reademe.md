# SPAD 浓雾场景 SNN 成像模型

## 1. 概述

```
SPAD 时间戳 [B, 4096, P=500]
  → 正弦位置编码 (17 通道, 消除 tof 幅度偏差) [1][2]
  → 等宽 SpikeBlock 堆叠 (全程 T、C、64×64 不变)
  → PLIF 膜电位自然积累帧间光子证据
  → EchoGate 逐帧目标光子选择
  → Gated Moment 深度/强度粗估计
  → 空间精修头 (残差 CNN, 消除逐像素 gate 噪声)
  → [B, 2, 64, 64]

[1] Mildenhall B, et al. NeRF. Communications of the ACM, 2022, 65(1): 99-106.
[2] Tancik M, et al. Fourier Features. NeurIPS, 2020.
```

## 2. 数据格式

```
输入:  [B, 4096, P]     P=500 帧, 每元素为 ToF 整数 (0=无效, 1~150=有效)
输出:  [B, 2, 64, 64]   ch0=深度(timebin), ch1=强度(目标光子占比)
GT:    [B, 2, 64, 64]   fog_level=0 末段直方图峰值, 含噪 → 弱监督
```

## 3. 编码方案：正弦位置编码

### 3.1 为什么不用标量 tof/150

```
tof 是位置信息, 不是强度信息
标量编码下 PLIF 会把 tof=93 当作比 tof=42 "更强的刺激"
单个线性 filter 对 tof 只能单调响应, 无法实现窗口选择
```

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

```
n_freq=8 → 17 通道 (1 valid + 8 sin + 8 cos), 值域 [-1, 1]
相邻 tof 编码相似, 远 tof 编码不同, 所有 tof 编码能量相同
Conv 通过 sin/cos 线性组合可选出任意 tof 区间
深度值由 Gated Moment 直接用原始 tof, 编码只服务于分类
```
也就是说，经过编码，[P, B, 1, 64, 64]->[P, B, 17, 64, 64]
后续直接conv2d在channel上操作

### 3.3 频率选取消融实验

编码效果取决于频率分量的选取方式。等差频率 `[1..8]` 是 NeRF 的标准做法，但 SPAD 浓雾场景有特殊约束：
- **大尺度区分**：雾后向散射 (bin≈40) vs 目标回波 (bin≈60)，差 20 bin → gate 需准确分类
- **精细分辨率**：目标区域 bin∈[55,65] 内的深度恢复需要相邻 bin 可区分

对 5 种频率方案在相同通道数 (8对=17通道) 下进行对比：

#### 实验设置

```
t_max = 150, 固定 fog_bin=40, target_bin=60
评估指标:
  - cos: 余弦相似度 (越接近0 → 编码方向越正交 → 越好区分)
  - L2:  欧氏距离 (越大 → 编码空间中距离越远 → 越好区分)
  - adj: 相邻 bin 平均 L2 距离 (分辨率, 越大 → 精细区分能力越强)
```

#### 结果

```
方案                            | 雾vs目标(差20) | 近雾(差4)    | 近目标(差10)  | 雾边缘(差5)  | 目标边缘(差4) | adj分辨率
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
A: [1,2,3,4,5,6,7,8]  (等差)    | c=-0.05 L=4.4 | c=+0.92 L=1.2 | c=+0.58 L=2.8 | c=+0.88 L=1.5 | c=+0.92 L=1.2 | 0.299
B: [1,2,4,6,8,12,16,24] (非均匀) | c=+0.12 L=4.0 | c=+0.66 L=2.5 | c=+0.25 L=3.7 | c=+0.53 L=2.9 | c=+0.66 L=2.5 | 0.689
C: [1,2,3,4,6,8,12,16] (混合A)  | c=+0.25 L=3.7 | c=+0.81 L=1.8 | c=+0.31 L=3.5 | c=+0.72 L=2.2 | c=+0.81 L=1.8 | 0.481
D: [1,2,3,4,5,6,8,12]  (混合B)  | c=+0.09 L=4.0 | c=+0.89 L=1.4 | c=+0.47 L=3.1 | c=+0.83 L=1.7 | c=+0.89 L=1.4 | 0.362
E: [1,2,3,5,8,12,16,24] (混合C) | c=+0.20 L=3.8 | c=+0.67 L=2.5 | c=+0.29 L=3.6 | c=+0.54 L=2.9 | c=+0.67 L=2.5 | 0.683
```

#### 分析

**方案 A (等差 [1..8])** — 大尺度区分最优 (cos=-0.05，几乎正交)，但相邻 bin 分辨率仅 0.30，差 4 bin 的两点余弦相似度高达 0.92，近乎不可分。精细深度恢复能力最弱。

**方案 B (非均匀 [1,2,4,6,8,12,16,24])** — 精细分辨率是 A 的 2.3 倍 (0.69 vs 0.30)，差 4 bin 的 L2 从 1.2 提升到 2.5。大尺度区分从 cos=-0.05 变为 +0.12，仍远低于 0.5 的混淆门限，对 gate 学习无实质影响。

**方案 E ([1,2,3,5,8,12,16,24])** — 性能接近 B (adj=0.68)，但保留了 freq=3 使低频过渡更平滑。

**方案 C/D** — 折中策略，指标介于 A 和 B 之间，无突出优势。

#### 结论

```
选用方案 B: freqs = [1, 2, 4, 6, 8, 12, 16, 24]

理由:
1. 瓶颈不在大尺度: 雾(bin≈40) vs 目标(bin≈60) 差 20 bin,
   所有方案的 L2 都在 3.7~4.4, gate 均能学会区分
2. 瓶颈在精细分辨: 目标区域 bin∈[55,65] 的深度恢复需要高分辨率,
   方案 B 的 adj=0.689 远优于 A 的 0.299
3. 通道数不变: 仍为 8 频率对 → 17 通道, 不增加参数/显存
4. 高频延伸: 最高频 24π 的 Nyquist 分辨率约 150/(2×24)≈3.1 bin,
   覆盖了目标区域内精细距离估计的需求
```

可视化脚本: `visualize_encoding.py --target_bin 60` 可生成频率响应、编码区分度、雾/目标编码对比等图。

### 3.4 可学习 LUT 编码消融实验

#### 动机

正弦编码的频率结构是人工预设的，无法适应数据分布。一个自然的问题是：如果让编码完全由数据驱动学习，能否获得更优的 tof 表征？

#### 方案: Learnable Embedding LUT

用一个独立的可学习查表 (Look-Up Table) 替代正弦公式，将整数 tof bin 直接映射为低维特征向量。LUT 与正弦编码完全解耦：**embed_dim 可自由调整**（8/16/32 等），不必等于正弦编码的 17 通道。

```python
# 核心: nn.Embedding(151, embed_dim, padding_idx=0)
#   index 0 (tof=0, invalid) → 全零向量, 不参与梯度
#   index 1~150             → 可学习的 embed_dim 维特征向量
#   embed_dim 独立可调, stem 层自动适配输入通道数

model = SPADSpikeNet(encoding_mode="lut", embed_dim=16, lut_init="rbf")
```

训练时通过成像损失 (L_GT + L_SSIM + ...) 反向传播，**只有当前 batch 实际访问到的 bin 才被更新**（embedding 的稀疏梯度特性）。

#### 初始化策略

LUT 无内建结构，初始化方式直接影响训练起点和收敛性。支持三种策略：

```
策略          方法                                              特点
──────────────────────────────────────────────────────────────────────────
sinusoidal    用 sin/cos 编码值填充                             与固定正弦编码同起点, 最稳定
              (截断或补零适配 embed_dim)                        适合验证 "学习能否超越固定编码"

rbf           高斯径向基函数                                    embed_dim 个中心均匀分布在 [1, t_max]
              exp(-(b-c_j)^2 / 2σ^2)                          σ 按相邻中心间距自动设定 (0.8×间距)
              σ 保证 ~60% 重叠                                  结构与正弦无关, 独立对照组

random        标准正态 N(0, 0.5)                               无先验, 完全靠训练自组织
              index 0 保持全零                                  基线: 验证正则是否足以稳定训练
```

#### 稳定性约束

裸 LUT 训练容易出现过拟合和训练不稳定：
- 低频 bin（如 bin=40 雾区）被高频访问，更新过快
- 高 bin（如 bin>100）样本稀少，embedding 可能发散
- 相邻 bin 无结构约束，可能学出剧烈跳变

因此配合以下正则化手段：

```
约束                    实现方式                                    权重
────────────────────────────────────────────────────────────────────────
invalid 映射            padding_idx=0 → 全零, 不参与梯度            内建
valid mask              编码后 × valid, 无效像素不贡献信号           内建
相邻平滑 L_adj          mean(||emb[i+1]-emb[i]||^2), i∈[1,149]    w=0.01
范数一致性 L_norm       var(||emb[i]||), 各bin编码能量应均匀         w=0.005
```

完整训练 loss:
```
L = L_imaging + 0.01 × L_adj + 0.005 × L_norm

其中 L_imaging = 原有的 (L_GT + L_SSIM + L_var + L_sparse + L_smooth)
```

#### 数据形状与参数开销

```
配置                        参数量    LUT参数   输出形状
──────────────────────────────────────────────────────────
sinusoidal (n_freq=8)       26,026    0         [T, B, 17, H, W]
lut D=16 init=sinusoidal    28,410    2,416     [T, B, 16, H, W]
lut D=16 init=rbf           28,410    2,416     [T, B, 16, H, W]
lut D=16 init=random        28,410    2,416     [T, B, 16, H, W]
lut D=32 init=rbf           31,338    4,832     [T, B, 32, H, W]

LUT 参数 = 151 × embed_dim (index 0~150)
stem 层 Conv(embed_dim→C, 1) 自动适配输入通道, 后续网络无需修改
```

#### 与正弦编码的本质区别

```
维度          正弦编码                    LUT 编码
──────────────────────────────────────────────────────────
频率结构      人工预设 (等差/非均匀)       无预设, 数据驱动学习
输出维度      固定 2*n_freq+1 (=17)       embed_dim 可自由调整
相邻 bin 关系  由 sin/cos 连续性天然保证    需显式 L_adj 正则约束
参数量        0                           151 × embed_dim
泛化性        对未见 tof 天然泛化           依赖初始化 + 正则
训练稳定性    稳定 (无可学习参数)           需正则防止发散
表达能力      受限于预设频率组合            理论上可学出任意映射
```

#### PLIF 时间维处理

LUT 编码输出形状 `[T, B, embed_dim, H, W]`，PLIF 的膜电位沿 T 轴累积过程不受影响。关键注意：
- LUT 的 `padding_idx=0` 保证无效帧输入全零 → PLIF 膜电位不被无效帧干扰（与正弦编码中 valid mask 清零效果等价）
- chunk 间膜电位 detach 机制不变

#### 预期消融结果

| 场景 | 正弦编码 | LUT 编码 | 说明 |
|------|---------|---------|------|
| 小数据 (<1000 样本) | 更稳定 | 可能过拟合 | LUT 自由度高, 小样本下正则不足 |
| 大数据 (>10000 样本) | 受限于预设频率 | 有望更优 | 数据足以驱动 LUT 学出最优映射 |
| 多场景泛化 (不同雾浓度) | 天然泛化 | 需充分训练覆盖 | 正弦编码的连续性是优势 |
| embed_dim 扫描 | 固定 17 通道 | 可调 8/16/32 | 寻找编码维度 vs 参数量最优平衡 |
| 初始化对比 | — | sin > rbf > random (预期) | 验证先验结构对收敛速度的影响 |

```python
# 使用方式
model_sin = SPADSpikeNet(encoding_mode="sinusoidal")                          # 默认正弦编码
model_lut = SPADSpikeNet(encoding_mode="lut", embed_dim=16, lut_init="rbf")   # LUT, RBF 初始化
model_lut = SPADSpikeNet(encoding_mode="lut", embed_dim=32, lut_init="sinusoidal")  # LUT, 32维
model_lut = SPADSpikeNet(encoding_mode="lut", embed_dim=8,  lut_init="random")      # LUT, 随机
```

## 4. 网络结构

### 4.1 结构图

```
输入 [B, 4096, P=500]
     │
     ▼ reshape + 正弦编码
[T*B, 17, 64, 64]          T=chunk_size (128~500), 超出则分 chunk
     │
     ▼ Stem: Conv(17→C,1)+BN+PLIF + Conv(C→C,3)+BN
[T*B, C, 64, 64]
     │
     ▼ SpikeBlock ×3 (残差 + 多尺度膨胀 DSConv)
     │
     │   每个 block:
     │     identity = x
     │     → PLIF → [DW 3×3 d=1,2,4 并行] → cat → PW → BN
     │     → PLIF → PW → BN
     │     → + identity
     │
[T*B, C, 64, 64]           T, C, 64×64 全程不变
     │
     ▼ EchoGate: PLIF → Conv(C→1) → sigmoid
[T, B, 1, 64, 64]          gate ∈ [0,1], 逐帧逐像素
     │
     ▼ Gated Moment (用原始 tof, 不用编码值)
     │   depth     = Σ(gate × tof × valid) / Σ(gate × valid)
     │   intensity = Σ(gate × valid) / T
     │
[B, 2, 64, 64]  粗估计（逐像素独立, 可能有椒盐噪声）
     │
     ▼ 空间精修头 (普通 CNN, 不是 SNN)
     │   Conv(2→8, 3×3, pad=1) + BN + ReLU
     │   Conv(8→2, 3×3, pad=1)
     │   + 残差 (粗估计直接加回来, 精修头只学修正量)
     │
[B, 2, 64, 64]  精修输出
```

精修头的作用:
- Gated Moment 逐像素独立计算, gate 的随机误差导致空间噪声
- 精修头利用 3×3 邻域平滑 gate 误差, 同时保留目标边缘
- 残差连接: 深度值仍以 Gated Moment 为基础, 精修头只做微调
- 极小: 两层 Conv, 参数 < 1K, 不增加显存负担

### 4.2 为什么不用时间 U-Net

```
任务是 64×64 → 64×64, 输入输出同分辨率, P 帧之间是独立采样
PLIF 膜电位天然沿帧轴积累光子证据, 不需要人为压缩/恢复时间维
fp16 + checkpoint 可将 chunk 开到 128~500, 无需靠 U-Net 扩展时间窗口
等宽结构: gate 在完整 T 分辨率下输出, 无上采样信息损失
```

### 4.3 关键组件

**MSDSConv** (多尺度膨胀深度可分离卷积):
```
三路并行 DW-Conv 3×3 (dilation=1,2,4) → cat(3C) → PW-Conv(3C→C) → BN
感受野: 3×3 / 5×5 / 9×9, 全程 64×64 不变
```

**Chunked 处理** (P > chunk_size 时):
```
chunk 1          chunk 2               chunk n
帧 1~128         帧 129~256    ...     帧 ...~500
   │                │                     │
   ▼                ▼                     ▼
┌────────┐     ┌────────┐          ┌────────┐
│ 等宽SNN │─膜电位→│ 等宽SNN │─膜电位→...→│ 等宽SNN │
│(有梯度) │(detach)│(有梯度) │           │(有梯度) │
└───┬────┘     └───┬────┘          └───┬────┘
    └────── 累加 gated moment ────────┘ → depth, intensity

权重共享, 膜电位跨 chunk 延续但梯度截断, 显存 = 1 个 chunk
```

### 4.4 数据形状 (chunk=128, B=8, C=32)

```
层                        形状                    空间
──────────────────────────────────────────────────────
编码输入                  [1024, 17, 64, 64]      64×64
Stem                      [1024, 32, 64, 64]      64×64
SpikeBlock 1~3            [1024, 32, 64, 64]      64×64
EchoGate                  [1024,  1, 64, 64]      64×64

全程: T 不变, C 不变, 空间不变
```

## 5. 损失函数与评估指标

### 5.1 训练损失

```
L = 0.3  × L_GT       |depth-GT| + |intensity-GT|       弱标签锚点 (L1/MAE)
  + 0.1  × L_SSIM     1 - SSIM(pred, GT)                结构相似性 (局部亮度/对比度/结构)
  + 1.0  × L_var      gate 选中光子的 tof 方差 (须聚集)   核心物理约束
  + 0.05 × L_sparse   mean(gate×valid), 浓雾下应稀疏      先验
  + 0.1  × L_smooth   |∇d|·exp(-β|∇I|)                  边缘保持平滑
```

L_SSIM 使用 7×7 高斯窗口 (适配 64×64 分辨率), 对 depth 和 intensity 分别计算后加权合并。
depth 的 data_range=150 (tof bin 范围), intensity 的 data_range=1.0。
SSIM 捕获局部结构信息, 弥补纯像素级 L1 对边缘和纹理的盲区。

### 5.2 评估指标

验证/测试时使用 `ImageMetrics` 类 (不参与梯度), 对 depth 和 intensity 各计算:

```
所有指标在归一化到 [0,1] 后计算 (depth 除以 depth_range=150), 与论文标准一致

MAE   = mean(|pred - gt|)               像素级绝对误差
RMSE  = sqrt(mean((pred - gt)^2))       对大误差更敏感
SSIM  = 结构相似性 (7×7 高斯窗口)       局部结构质量
PSNR  = 10 × log10(1 / MSE)  (dB)       信噪比, 归一化后 data_range=1.0
```

## 6. 参数与显存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| C | 32 | 通道数 (全程不变) |
| chunk_size | 128 | 每 chunk 帧数 |
| n_freq | 8 | 正弦编码频率数 → 17 输入通道 |
| spike_mode | "plif" | 神经元类型 |
| t_max | 150 | 最大有效 ToF bin |

**显存** (fp16 + gradient checkpoint):

```
公式: chunk_size × B × 6.5 MB  (C=32)

chunk  B=4     B=8     B=16    B=32
──────────────────────────────────────
128    3.3GB   6.5GB   13GB    26GB!
256    6.5GB   13GB    26GB!   --
500    13GB    26GB!   --      --

24GB GPU 推荐: chunk=128, B=8 (6.5 GB)
目标光子/chunk: chunk=128 约 12 个, chunk=500 约 47 个
```

## 7. 配置管理 (SNNConfig)

所有可调参数统一由 `SNN_config.py` 中的 `SNNConfig` dataclass 管理，模型、损失函数、评估指标均通过 config 构建:

```python
from SNN_config import SNNConfig

# 方式 1: 默认正弦编码
cfg = SNNConfig()

# 方式 2: 自定义 LUT 编码 (LUT 相关参数自动激活)
cfg = SNNConfig(
    encoding_mode="lut",   # 切换到 LUT → 自动启用 embed_dim / lut_init / w_lut_*
    embed_dim=16,
    lut_init="rbf",
)

# 方式 3: 从 JSON 加载
cfg = SNNConfig.load("experiment_01.json")

# 构建全部组件
model     = cfg.build_model()       # SPADSpikeNet, 自动配置编码方式
criterion = cfg.build_loss()        # SPADImagingLoss, LUT 正则权重自动传入
metrics   = cfg.build_metrics()     # ImageMetrics

# 查看配置摘要
print(cfg.summary())

# 保存配置 (实验复现)
cfg.save("experiment_01.json")
```

参数分组与自动激活逻辑:
```
分组            参数                                 sinusoidal  lut
───────────────────────────────────────────────────────────────────────
数据            t_max                                生效        生效
编码 (公共)     encoding_mode, n_freq                生效        生效
编码 (LUT)      embed_dim, lut_init, lut_max_norm    忽略        生效
网络            C, chunk_size, spike_mode, num_blocks 生效        生效
损失 (公共)     w_gt, w_ssim, w_var, w_sparse, w_smooth 生效     生效
损失 (LUT)      w_lut_smooth, w_lut_norm             忽略        生效
损失超参        sigma_target, rho_target, beta_smooth 生效        生效
```

预置配置:
```python
from SNN_config import SINUSOIDAL_DEFAULT, LUT_RBF_16, LUT_SIN_16, LUT_RBF_32
```

## 8. SNN_new.py — activation_based API 版本

`SNN_new.py` 是 `SNN.py` 的新版实现，将 SpikingJelly 后端从旧版 `clock_driven` 迁移到 `activation_based`（新版官方 API）。对外接口（输入/输出形状、参数名）与 `SNN.py` 完全一致，可直接替换。

### 8.1 API 差异对照

| 维度 | SNN.py (clock_driven) | SNN_new.py (activation_based) |
|------|----------------------|-------------------------------|
| 导入 | `from spikingjelly1.clock_driven.neuron import MultiStepParametricLIFNode` | `from spikingjelly.activation_based import neuron, functional` |
| 神经元构造 | `MultiStepParametricLIFNode(timestep=T, ...)` | `neuron.ParametricLIFNode(step_mode='m', ...)` |
| 时间步参数 | 构造时传入 `timestep`，神经元内部展开 | 无 `timestep`，`step_mode='m'` 表示多步模式 |
| 输入形状 | `[T*B, C, H, W]`（时间批次展平） | `[T, B, C, H, W]`（时间维独立） |
| ANN 子模块 | 直接调用（已展平） | `functional.seq_to_ann_forward(x, module)` 展开时间维 |
| chunk 间截断 | 手动遍历 `m.v = m.v.detach()` | `functional.detach_net(self)` |
| 网络重置 | `functional.reset_net(self)` | `functional.reset_net(self)`（接口不变） |

### 8.2 结构变化

新版将 Stem 和 GateHead 提取为独立模块 `_Stem` / `_GateHead`，内部通过 `seq_to_ann_forward` 正确处理时间维：

```
SNN.py:
  stem = nn.Sequential(Conv, BN, MultiStepPLIF, Conv, BN)
  # 输入 [T*B, C, H, W]，PLIF 内部按 timestep 展开

SNN_new.py:
  _Stem.forward(x: [T, B, C_enc, H, W]):
    x = seq_to_ann_forward(x, Conv+BN)   # [T, B, C, H, W]
    x = spike(x)                          # PLIF step_mode='m'
    x = seq_to_ann_forward(x, Conv+BN)   # [T, B, C, H, W]
```

`SpikeBlock` 同理：ANN 子模块（DSConv、BN、PW）均通过 `seq_to_ann_forward` 包装，脉冲神经元直接接收 `[T, B, C, H, W]`。

### 8.3 使用方式

```python
# 与 SNN.py 接口完全一致，直接替换导入即可
from SNN_based_method.SNN_new import SPADSpikeNet

model = SPADSpikeNet(C=32, chunk_size=128, spike_mode="plif")
model = SPADSpikeNet(encoding_mode="lut", embed_dim=16, lut_init="rbf")

out = model(raw_data)   # raw_data: [B, 4096, P]
# out["output"]: [B, 2, 64, 64]
```

### 8.4 环境要求

```
SNN.py      → spikingjelly1 (项目本地 clock_driven 副本, conda env: pytorch)
SNN_new.py  → spikingjelly  (activation_based, conda env: torchnew)
```

两个文件可在各自环境下独立运行，不互相依赖。

### 8.5 cupy backend 自动检测

`SNN_new.py` 在模块加载时自动探测 cupy backend 是否可用，无需手动配置：

```python
# 模块加载时自动运行, 结果缓存到 _CUPY_AVAILABLE
# True  → 所有神经元使用 backend='cupy' (GPU 加速脉冲计算)
# False → 回退到 backend='torch'
```

探测逻辑：构造一个 `IFNode(backend='cupy')` 并实际跑一次前向，确认端到端可用后才返回 `True`。仅 `import cupy` 不够，因为 cupy 在缺少 CUDA headers 时 import 可能成功但运算会失败。

#### 环境配置 (torchnew)

| 依赖 | 版本 | 说明 |
|------|------|------|
| cupy-cuda12x | 14.1.0 | `pip install cupy-cuda12x` |
| pytest | ≥9.0 | cupy.testing 依赖, `pip install pytest` |
| CUDA Toolkit | 12.8 | 安装路径: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` |

**CUDA_PATH 说明**: `conda run` 子进程可能未继承系统新增的环境变量。`_probe_cupy_backend()` 内部会在 `CUDA_PATH` 缺失时自动兜底设置默认路径，无需手动传入。

#### spikingjelly 开发版兼容性修复

若使用 spikingjelly 开发版（非 pip 稳定版），需手动修复以下两处：

```
文件: spikingjelly/activation_based
问题: 缩进没对齐
修复: 从错误跳转后对齐即可

文件: spikingjelly/activation_based/neuron/base_node.py
问题: abstractmethod 未导入
修复: 文件头加 from abc import abstractmethod
```

---

## 9. 文件说明

```
SNN_based_method/
  reademe.md              本文档
  SNN_config.py           统一配置管理 (SNNConfig dataclass + 预置配置 + JSON 序列化)
  SNN.py                  模型实现 — clock_driven API (旧版 spikingjelly, env: pytorch)
  SNN_new.py              模型实现 — activation_based API (新版 spikingjelly, env: torchnew)
  loss.py                 训练损失 (L_GT + L_SSIM + L_var + L_sparse + L_smooth + LUT正则) + 评估指标
  visualize_encoding.py   编码可视化工具 (频率响应、多帧聚合、雾/目标区分度分析)
```

## 10. 显存占用测试
SNN_new v1 使用run_5d_memory_benchmark进行：
```
Total memory: 12.0 GB
cupy backend: available (backend=cupy)
Benchmark input: [T, B, C, 64, 64]
Network path: stem -> SpikeBlocks -> gate_head -> temporal aggregation -> refine
Mode: forward + backward
Stop rule: peak_allocated > 12 GB, skip larger T for current B/C

=== C= 4, B= 4, H=W=64 ===
T= 50  input=    12.5 MB  peak_alloc=    578.1 MB  peak_reserved=    688.0 MB  PASS
T=100  input=    25.0 MB  peak_alloc=   1182.4 MB  peak_reserved=   1344.0 MB  PASS
T=150  input=    37.5 MB  peak_alloc=   1734.8 MB  peak_reserved=   1970.0 MB  PASS
T=200  input=    50.0 MB  peak_alloc=   2299.1 MB  peak_reserved=   2604.0 MB  PASS
T=250  input=    62.5 MB  peak_alloc=   2870.7 MB  peak_reserved=   3286.0 MB  PASS
T=300  input=    75.0 MB  peak_alloc=   3473.3 MB  peak_reserved=   3912.0 MB  PASS
T=350  input=    87.5 MB  peak_alloc=   4023.9 MB  peak_reserved=   4538.0 MB  PASS
T=400  input=   100.0 MB  peak_alloc=   4579.7 MB  peak_reserved=   5174.0 MB  PASS
T=450  input=   112.5 MB  peak_alloc=   5153.1 MB  peak_reserved=   5868.0 MB  PASS
T=500  input=   125.0 MB  peak_alloc=   5757.4 MB  peak_reserved=   6494.0 MB  PASS

=== C= 4, B= 8, H=W=64 ===
T= 50  input=    25.0 MB  peak_alloc=   1185.1 MB  peak_reserved=   1348.0 MB  PASS
T=100  input=    50.0 MB  peak_alloc=   2303.6 MB  peak_reserved=   2608.0 MB  PASS
T=150  input=    75.0 MB  peak_alloc=   3475.8 MB  peak_reserved=   3916.0 MB  PASS
T=200  input=   100.0 MB  peak_alloc=   4584.3 MB  peak_reserved=   5178.0 MB  PASS
T=250  input=   125.0 MB  peak_alloc=   5760.1 MB  peak_reserved=   6498.0 MB  PASS
T=300  input=   150.0 MB  peak_alloc=   6878.6 MB  peak_reserved=   7758.0 MB  PASS
T=350  input=   175.0 MB  peak_alloc=   8050.8 MB  peak_reserved=   9066.0 MB  PASS
T=400  input=   200.0 MB  peak_alloc=   9159.3 MB  peak_reserved=  10328.0 MB  PASS
T=450  input=   225.0 MB  peak_alloc=  10335.1 MB  peak_reserved=  11648.0 MB  PASS
T=500  input=   250.0 MB  peak_alloc=  11453.6 MB  peak_reserved=  12908.0 MB  PASS

=== C= 4, B=16, H=W=64 ===
T= 50  input=    50.0 MB  peak_alloc=   2320.6 MB  peak_reserved=   2610.0 MB  PASS
T=100  input=   100.0 MB  peak_alloc=   4601.6 MB  peak_reserved=   5180.0 MB  PASS
T=150  input=   150.0 MB  peak_alloc=   6895.6 MB  peak_reserved=   7760.0 MB  PASS
T=200  input=   200.0 MB  peak_alloc=   9176.6 MB  peak_reserved=  10330.0 MB  PASS
T=250  input=   250.0 MB  peak_alloc=  11470.6 MB  peak_reserved=  12910.0 MB  PASS
T=300  input=   300.0 MB  peak_alloc=  13751.6 MB  peak_reserved=  15480.0 MB  OVER_12GB
T=300  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C= 4, B=32, H=W=64 ===
T= 50  input=   100.0 MB  peak_alloc=   4613.1 MB  peak_reserved=   5196.0 MB  PASS
T=100  input=   200.0 MB  peak_alloc=   9188.1 MB  peak_reserved=  10346.0 MB  PASS
T=150  input=   300.0 MB  peak_alloc=  13763.1 MB  peak_reserved=  15496.0 MB  OVER_12GB
T=150  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C= 8, B= 4, H=W=64 ===
T= 50  input=    25.0 MB  peak_alloc=   1182.7 MB  peak_reserved=   1332.0 MB  PASS
T=100  input=    50.0 MB  peak_alloc=   2295.1 MB  peak_reserved=   2580.0 MB  PASS
T=150  input=    75.0 MB  peak_alloc=   3466.2 MB  peak_reserved=   3876.0 MB  PASS
T=200  input=   100.0 MB  peak_alloc=   4571.6 MB  peak_reserved=   5126.0 MB  PASS
T=250  input=   125.0 MB  peak_alloc=   5745.2 MB  peak_reserved=   6432.0 MB  PASS
T=300  input=   150.0 MB  peak_alloc=   6857.6 MB  peak_reserved=   7680.0 MB  PASS
T=350  input=   175.0 MB  peak_alloc=   8028.7 MB  peak_reserved=   8976.0 MB  PASS
T=400  input=   200.0 MB  peak_alloc=   9134.1 MB  peak_reserved=  10226.0 MB  PASS
T=450  input=   225.0 MB  peak_alloc=  10307.7 MB  peak_reserved=  11532.0 MB  PASS
T=500  input=   250.0 MB  peak_alloc=  11420.1 MB  peak_reserved=  12780.0 MB  PASS

=== C= 8, B= 8, H=W=64 ===
T= 50  input=    50.0 MB  peak_alloc=   2311.8 MB  peak_reserved=   2586.0 MB  PASS
T=100  input=   100.0 MB  peak_alloc=   4588.6 MB  peak_reserved=   5132.0 MB  PASS
T=150  input=   150.0 MB  peak_alloc=   6874.3 MB  peak_reserved=   7686.0 MB  PASS
T=200  input=   200.0 MB  peak_alloc=   9151.1 MB  peak_reserved=  10232.0 MB  PASS
T=250  input=   250.0 MB  peak_alloc=  11436.8 MB  peak_reserved=  12786.0 MB  PASS
T=300  input=   300.0 MB  peak_alloc=  13713.6 MB  peak_reserved=  15332.0 MB  OVER_12GB
T=300  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C= 8, B=16, H=W=64 ===
T= 50  input=   100.0 MB  peak_alloc=   4599.6 MB  peak_reserved=   5144.0 MB  PASS
T=100  input=   200.0 MB  peak_alloc=   9162.1 MB  peak_reserved=  10244.0 MB  PASS
T=150  input=   300.0 MB  peak_alloc=  13724.6 MB  peak_reserved=  15344.0 MB  OVER_12GB
T=150  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C= 8, B=32, H=W=64 ===
T= 50  input=   200.0 MB  peak_alloc=   9197.1 MB  peak_reserved=  10282.0 MB  PASS
T=100  input=   400.0 MB  peak_alloc=  18322.1 MB  peak_reserved=  20482.0 MB  OVER_12GB
T=100  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=16, B= 4, H=W=64 ===
T= 50  input=    50.0 MB  peak_alloc=   2308.5 MB  peak_reserved=   2584.0 MB  PASS
T=100  input=   100.0 MB  peak_alloc=   4582.1 MB  peak_reserved=   5130.0 MB  PASS
T=150  input=   150.0 MB  peak_alloc=   6864.7 MB  peak_reserved=   7684.0 MB  PASS
T=200  input=   200.0 MB  peak_alloc=   9138.4 MB  peak_reserved=  10230.0 MB  PASS
T=250  input=   250.0 MB  peak_alloc=  11421.0 MB  peak_reserved=  12784.0 MB  PASS
T=300  input=   300.0 MB  peak_alloc=  13694.6 MB  peak_reserved=  15330.0 MB  OVER_12GB
T=300  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=16, B= 8, H=W=64 ===
T= 50  input=   100.0 MB  peak_alloc=   4592.9 MB  peak_reserved=   5146.0 MB  PASS
T=100  input=   200.0 MB  peak_alloc=   9149.1 MB  peak_reserved=  10246.0 MB  PASS
T=150  input=   300.0 MB  peak_alloc=  13705.4 MB  peak_reserved=  15346.0 MB  OVER_12GB
T=150  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=16, B=16, H=W=64 ===
T= 50  input=   200.0 MB  peak_alloc=   9183.6 MB  peak_reserved=  10278.0 MB  PASS
T=100  input=   400.0 MB  peak_alloc=  18296.1 MB  peak_reserved=  20478.0 MB  OVER_12GB
T=100  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=16, B=32, H=W=64 ===
T= 50  input=   400.0 MB  peak_alloc=  18367.1 MB  peak_reserved=  20576.0 MB  OVER_12GB
T= 50  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=24, B= 4, H=W=64 ===
T= 50  input=    75.0 MB  peak_alloc=   3473.3 MB  peak_reserved=   3874.0 MB  PASS
T=100  input=   150.0 MB  peak_alloc=   6866.2 MB  peak_reserved=   7698.0 MB  PASS
T=150  input=   225.0 MB  peak_alloc=  10303.3 MB  peak_reserved=  11452.0 MB  PASS
T=200  input=   300.0 MB  peak_alloc=  13693.2 MB  peak_reserved=  15344.0 MB  OVER_12GB
T=200  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=24, B= 8, H=W=64 ===
T= 50  input=   150.0 MB  peak_alloc=   6896.9 MB  peak_reserved=   7736.0 MB  PASS
T=100  input=   300.0 MB  peak_alloc=  13724.2 MB  peak_reserved=  15380.0 MB  OVER_12GB
T=100  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=24, B=16, H=W=64 ===
T= 50  input=   300.0 MB  peak_alloc=  13769.7 MB  peak_reserved=  15416.0 MB  OVER_12GB
T= 50  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=24, B=32, H=W=64 ===
T= 50  OOM, skip larger T for this B/C

=== C=32, B= 4, H=W=64 ===
T= 50  input=   100.0 MB  peak_alloc=   4589.6 MB  peak_reserved=   5144.0 MB  PASS
T=100  input=   200.0 MB  peak_alloc=   9142.7 MB  peak_reserved=  10244.0 MB  PASS
T=150  input=   300.0 MB  peak_alloc=  13695.8 MB  peak_reserved=  15344.0 MB  OVER_12GB
T=150  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=32, B= 8, H=W=64 ===
T= 50  input=   200.0 MB  peak_alloc=   9177.0 MB  peak_reserved=  10280.0 MB  PASS
T=100  input=   400.0 MB  peak_alloc=  18283.2 MB  peak_reserved=  20480.0 MB  OVER_12GB
T=100  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=32, B=16, H=W=64 ===
T= 50  input=   400.0 MB  peak_alloc=  18353.7 MB  peak_reserved=  20572.0 MB  OVER_12GB
T= 50  peak_allocated exceeds 12 GB, skip larger T for this B/C

=== C=32, B=32, H=W=64 ===
T= 50  OOM, skip larger T for this B/C
```
