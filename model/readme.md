# 单光子激光雷达 (SPAD) 目标检测与分布建模研究框架

## 1. 研究背景与问题动机
单光子雪崩二极管 (SPAD) 激光雷达因其极高的灵敏度和皮秒级时间分辨率，在自动驾驶、无人机避障等领域展现出巨大潜力。然而，在复杂场景（如浓雾、强背景光）下，SPAD 采集的原始数据由海量的光子到达时间直方图 (Histogram) 组成，不仅存在极强的环境噪声干扰，且数据维度庞大。

单帧观测通常可表示为时间-空间网格 $\mathbf I = \{I_{p,t}\}_{p=1,t=1}^{P,T}\in\mathbb R_+^{P\times T}$，其中空间像素 $P=64\times 64$，时间 bin 维度 $T=128$。$I_{p,t}$ 表示像素 $p$ 在时间 bin $t$ 上的累积光子强度。
*   **目标信号特征**：回波强度服从高斯分布，峰值较低（如 $<100$）。
*   **环境噪声干扰（如浓雾）**：回波强度服从伽马分布，峰值极高（如 $>700$），且在时间 bin 上可能与目标回波存在交叠。

**现有方法的局限性与本文研究逻辑：**
传统方法通常基于峰值提取 (Peak Extraction) 或简单的阈值滤波，将庞大的网格数据降采样为稀疏的 3D 点云（如 $1024 \times 4$ 维度的坐标及强度），再输入常规三维视觉网络。这种范式存在两个缺陷：
1. **现有网络难以适配单光子噪声特性**：通用点云网络在降采样后的单光子高噪点云上表现不佳，缺乏对局部几何中心位置的强约束。
2. **直方图信息丢失严重**：峰值提取不可避免地丢弃了时间序列上的统计分布特征，使得算法在浓雾等极端工况下无法分离目标与噪声。

针对上述瓶颈，本课题沿着**“改进稀疏点云表征”**到**“突破全物理直方图建模”**的递进逻辑，设计了三个核心研究任务，以期形成一套完整的单光子数据解算与感知体系。

---

## 2. 核心研究任务

### 任务 1：基于图残差网络的稀疏单光子点云目标检测（监督学习）
**目标与动机**：在传统的降采样点云处理范式下，提出一种定制化的 3D 目标分类与边界框 (3D Box) 回归网络。针对单光子点云中噪点易导致目标几何结构变形的问题，设计带有”坐标残差”的局部图特征提取模块。

**方法与总体流程**：
- **输入**：降采样点云矩阵 $(B, N, 4)$，$N{=}1024$，包含 $x, y, z$ 与反射强度 $i$。全程采用 channel-first $(B, C, N)$ 布局。

#### 整体网络结构

| 阶段 | 模块 | 输入 → 输出 | 说明 |
|:----:|:----:|:----------:|:----:|
| Stem | Conv1d×2 + BN1d + LeakyReLU | $(B,4,N) \to (B,32,N)$ | 升维至初始特征 |
| Block 1 | GraphResidualBlock | $(B,32,N) \to (B,64,N)$ | 双流 EdgeConv + Q/K/V 注意力 + 坐标门控 |
| Block 2 | GraphResidualBlock | $(B,64,N) \to (B,64,N)$ | 同上 |
| Block 3 | GraphResidualBlock | $(B,64,N) \to (B,128,N)$ | 同上 |
| Block 4 | GraphResidualBlock | $(B,128,N) \to (B,256,N)$ | 同上 |
| 多尺度聚合 | Conv1d + BN1d + LeakyReLU | $\text{cat}(f_1,f_2,f_3,f_4){=}(B,512,N) \to (B,512,N)$ | 各层特征拼接后跨层融合 |
| 全局池化 | max-pool + avg-pool | $(B,512,N) \to (B,1024)$ | 全局描述子 |
| 分类头 | MLP (1024→512→256→$C$) | $(B,1024) \to (B,C)$ | BN + LeakyReLU + Dropout |
| Box 头 | MLP (1032→256→128→3) + centroid-offset | $(B,1032) \to (B,3)$ | 预测 = 点云质心 + 偏移 |

> **无下采样设计**: 全部 1024 点贯穿 4 层 Block，保留完整的 intensity 空间结构；多尺度拼接融合各层局部/全局特征。每个 Block 输出 `(p, f_out)`，其中 **p 始终为原始输入坐标 $(B,4,N)$ 不变**，仅 f 逐层升维。

#### 双路动态 KNN (GPU) + 坐标图缓存

网络维护两套独立的 KNN 图（$k{=}20$），服务于两路 EdgeConv：
- **特征空间 KNN** (`knn_f`): 从当前学到的特征 $\mathbf f$ 构建（语义驱动，DGCNN "Dynamic Graph" 范式），**每层重新计算**
- **坐标+intensity 空间 KNN** (`knn_p`): 从原始 4D 坐标 $\mathbf P{=}(x,y,z,i)$ 构建（保留 SPAD 强度空间结构）

KNN 通过负平方距离 + topk 实现，全程 GPU matmul。

> **v7 优化**: 由于 $\mathbf P$ 全程不变，`knn_p` 和 `get_graph_feature(p)` 在网络入口**一次性预计算**，生成 $\text{p\_graph} \in \mathbb R^{B \times 8 \times N \times k}$，4 个 Block 共享复用，省掉 3 次 $O(N^2)$ KNN + 3 次 gather/permute，约**提速 30%**。

#### 单 Block 数据流

每个 GraphResidualBlock 接收 `(p, f, p_graph)` 三个输入，内部执行以下步骤：

**Step 1: 双流 EdgeConv (Conv2d + BN2d + LeakyReLU)**
分别在**特征 KNN 图**和**坐标图缓存**上做 DGCNN 风格 EdgeConv：
- **GCN_f (特征 EdgeConv)**：`get_graph_feature(f, knn_f)` → $[\mathbf f_j{-}\mathbf f_i, \mathbf f_i] \in \mathbb R^{2C_{in}}$, 经 `Conv2d + BN2d + LeakyReLU` → $\mathbf F^k \in \mathbb R^{B \times C_{out} \times N \times k}$，编码邻域**语义关系** → 作为 **Value** 来源
- **GCN_p (位置 EdgeConv)**：复用预计算的 `p_graph` $\in \mathbb R^{B \times 8 \times N \times k}$, 经 `Conv2d + BN2d + LeakyReLU` → $\mathbf P^k \in \mathbb R^{B \times C_{out} \times N \times k}$，编码邻域**几何关系** → 作为 **Key** 来源

**Step 2: 标准 Q/K/V 图注意力 (Scaled Dot-Product)**

$$\mathbf Q_i = \text{Conv1d}_{q}([\mathbf f_i \| \mathbf P_i]) \quad \text{(中心点联合查询)}$$
$$\mathbf K_{ij} = \text{Conv2d}_{k}(\mathbf P^k_{ij}) \quad \text{(位置 EdgeConv → Key: 几何驱动权重)}$$
$$\mathbf V_{ij} = \text{Conv2d}_{v}(\mathbf F^k_{ij}) \quad \text{(特征 EdgeConv → Value: 语义被聚合)}$$
$$\alpha_{ij} = \text{softmax}_j\left(\frac{\mathbf Q_i \cdot \mathbf K_{ij}}{\sqrt{C_{out}}}\right),\quad \mathbf{attn}_i = \sum_j \alpha_{ij} \cdot \mathbf V_{ij}$$

> **解耦设计**: "关注谁"(K, 几何驱动) 与 "聚合什么"(V, 语义驱动) 由两路**独立 KNN 图**分别产生。

**Step 3: 坐标门控跳跃连接 (Coordinate-Gated Residual)**

$$\mathbf g = \sigma(\text{Conv1d}_{gate}(\mathbf P)),\quad \mathbf c = \text{Conv1d}_{res}(\mathbf P)$$
$$\mathbf{out} = \mathbf g \odot \text{Conv1d}_{out}(\mathbf{attn}) + (1 - \mathbf g) \odot \mathbf c$$

门控含义：$\mathbf g \to 1$ 信任注意力聚合的语义特征；$\mathbf g \to 0$ 信任原始坐标+强度信息。

```text
[单 Block 数据流向]
 f(B,C,N) + p(B,4,N) + p_graph(B,8,N,k) [预计算缓存]
  │
  ├── KNN from f (特征空间, 每层重算)   p_graph (坐标图, 入口预计算复用)
  │        ↓                                    │
  ├── [GCN_f]                              [GCN_p]
  │   get_graph_feature(f,knn_f)           Conv2d+BN2d+LeakyReLU
  │   Conv2d+BN2d+LeakyReLU                    │
  │       ↓                                     ↓
  │   Fk (B,C_out,N,k)                    Pk (B,C_out,N,k)
  │       |                                     |
  │   Conv2d [W_v]                         Conv2d [W_k]
  │       ↓                                     ↓
  │       V                                     K
  │        \       Q = Conv1d+BN(f‖p)          /
  │         \             |                   /
  │     ┌──────────────────────────────────┐
  │     │       Attention Module            │
  │     │                                   │
  │     │  score = Q · K / √C_out           │
  │     │  weights = softmax(score, dim=k)  │
  │     │  attn = Σ weights · V             │
  │     └──────────────────────────────────┘
  │                 │
  │          Conv1d+BN1d [out_conv]
  │                 │
  └──> gate · mapped + (1-gate) · coord_res ──> LeakyReLU ──> f_out
        gate = σ(Conv1d+BN(P[4D]))                            p 不变
```

#### Box 头: Centroid-Offset 预测

Box 头接收全局特征 + 点云坐标统计量 $(B, 1024{+}8)$，预测从**点云质心到目标中心的偏移量**（而非绝对坐标）：
$$\hat{\mathbf c} = \bar{\mathbf P}_{xyz} + \text{MLP}([\mathbf f_{pool} \| \bar{\mathbf P} \| \sigma_{\mathbf P}])$$
其中 $\bar{\mathbf P}$ 为 1024 点的 4D 坐标均值（质心锚点），$\sigma_{\mathbf P}$ 为标准差（尺度先验）。网络只需学习接近零的残差偏移，收敛显著加快。推理时由固定归一化半宽 `FIXED_BBOX_HALF_SIZE_NORMALIZED` 重建完整 3D 边界框。

> **GCN 变体使用直接回归**：为确保与 baseline (DGCNN/PointNet++ 等) 的对比公平性，GCN 版改为 `box_pred = self.box_head(f_pooled)` 直接从全局池化特征预测中心点坐标，不依赖质心先验。


#### 损失函数

仅 **2 项损失**，采用 **Kendall et al. (CVPR 2018) 同方差不确定性自适应权重** 自动平衡：

$$\mathcal L = e^{-s_{cls}} \mathcal L_{cls} + s_{cls} + e^{-s_{box}} \mathcal L_{box}^{gauss} + s_{box}$$

其中 $s = \log \sigma^2$ 为可学习参数（与模型参数一同优化），效果：
- $\sigma$ 大 → 该任务权重低（不确定性高，暂缓学习）
- $\sigma$ 小 → 该任务权重高（不确定性低，加速精化）
- $+s$ 正则项防止模型将 $\sigma \to \infty$ 来逃避困难任务

**SPAD Log-Cauchy Box Loss** $\mathcal L_{box}$ 受启发于 Deng et al. (Optics Letters, 2026) 提出的 Soft-histogram depth loss — 将 GT 深度建模为具有测量不确定度的分布而非硬点目标。本文将此思想扩展至 3D 中心点回归，采用 Log-Cauchy 形式：

$$\mathcal L_{box} = \sum_{d \in \{x,y,z\}} \log\left(1 + \frac{(\hat c_d - c_d^{gt})^2}{h_d^2}\right)$$

其中 $h_d$ 为各维归一化半宽（$h_x{=}h_y{\approx}0.119,\; h_z{\approx}0.023$），由目标物理尺寸（$x/y$ 宽 15 bin, $z$ 宽 5 bin）和归一化分母（63, 63, 109）导出，作为该维度的测量不确定度尺度。

| 误差范围 | 行为 | 梯度 |
|:-------:|:----:|:----:|
| $\|\Delta\| \ll h_d$ | $\approx \Delta^2/h^2$（MSE 级，柔和精调） | $\approx 2\Delta/h^2$（维度加权） |
| $\|\Delta\| \approx h_d$ | $\log(2) \approx 0.69$（转折点） | 峰值梯度 |
| $\|\Delta\| \gg h_d$ | $\approx 2\log(\|\Delta\|/h)$（对数增长） | $\approx 2/\Delta$（缓慢衰减但**永不为零**） |

| 特性 | 说明 |
|:----:|:----:|
| **物理含义** | 将 GT 中心建模为具有 per-dim 不确定度 $h_d$ 的分布; 误差在半宽内→低 loss, 超出→对数惩罚 |
| **维度自适应** | 内建: $z$ 轴 $h{=}0.018$ 使 $z$ 误差被自动放大 $\sim$43 倍 |
| **无梯度消失** | 对数增长保证任意大的误差仍有非零梯度 (避免了纯 Gaussian/IoU 的饱和问题) |
| **鲁棒性** | 大误差时对数增长而非线性/平方增长, 不会被离群样本主导 |

> **GCN 变体使用 Soft-histogram depth loss**：直接建模 SPAD 物理过程 (时间 bin 量化 + 高斯脉冲展宽)，见下文 GCN 版优化路线。

此设计**统一替代了之前的 SmoothL1 + DIoU 两项 loss**，将 3 项损失简化为 2 项。

两个损失分项：

| 损失项 | 公式 | 说明 |
|:-----:|:----:|:----:|
| $\mathcal L_{cls}$ | CrossEntropy | 26 类分类 |
| $\mathcal L_{box}$ | $\sum_d \log(1 + \Delta_d^2 / h_d^2)$ | SPAD Log-Cauchy 中心回归 |

#### 模型规模与速度优化 (v7)

- **参数量**：约 1.65 M（+ 2 个可学习损失权重参数）
- **显存**：RTX 4070 SUPER 上 $B{=}32$ 训练峰值约 7.4 GB（含梯度检查点）
- **v7 速度优化**：坐标 KNN + `get_graph_feature(p)` 在网络入口一次性预计算，4 个 Block 共享缓存张量 `p_graph`$(B,8,N,k)$，省掉 3 次 $O(N^2)$ KNN 和 3 次 gather/permute，约提速 30%

#### PyG 图卷积变体 (GCN Version)

为探索真图卷积 (Graph Convolutional Network) 在单光子点云上的潜力，我们在 `model/graph_res_GCN.py` 中实现了 **PyTorch Geometric SAGEConv 变体**，替换原版的 DGCNN 风格 Conv2d EdgeConv。

**核心区别**：

| 维度 | 原版 (DGCNN EdgeConv) | GCN 变体 (PyG SAGEConv) |
|:----:|:--------------------:|:----------------------:|
| **卷积核** | `Conv2d` on `[x_j - x_i, x_i]` | `SAGEConv` (真消息传递) |
| **聚合方式** | Max pool over edge features | Mean aggregation: $\mathbf h_i' = W_1 \mathbf h_i + W_2 \cdot \text{mean}_{j \in \mathcal N(i)} \mathbf h_j$ |
| **权重结构** | 单一 MLP (边特征 → 输出) | 中心-邻居分离权重 ($W_1$ vs $W_2$) |
| **归纳性** | Transductive (依赖边特征拼接) | Inductive (可处理未见图拓扑) |
| **缓存格式** | `p_graph` $(B,8,N,k)$ 边特征 | `p_edge_index` $(2, B{\cdot}N{\cdot}k)$ PyG 边索引 |
| **双流融合** | Q/K/V 注意力 (k 邻居 softmax) | SE channel gate + Conv1d 融合 |
| **Box 头** | Centroid-offset ($\bar{\mathbf P} + \text{MLP}$) | 直接回归 (与 baseline 一致) |
| **深度 Loss** | Log-Cauchy | Soft-histogram (物理驱动) |

**架构变化**：

1. **GCN_f (特征流)**：SAGEConv on 特征空间动态 KNN 图 (DGCNN "动态图" 范式保留，卷积核换为真 GNN)
2. **GCN_p (位置流)**：SAGEConv on 坐标空间静态 KNN 图 (预计算复用，同原版)
3. **SE channel gate**：因 SAGEConv 已将邻域聚合为单向量，全局 $(B,N,N)$ softmax 注意力既嘈杂又过拟合；改为 SE-style channel gate 对 $f_{gcn}$ 做通道加权，再与 $p_{gcn}$ 通过 Conv1d 融合
4. **Box 头直接回归**：与 DGCNN / PointNet++ 等 baseline 保持一致，从全局池化特征直接预测中心点坐标，不依赖质心先验
5. **Soft-histogram depth loss**：直接建模 SPAD 物理过程 (时间 bin 量化 + 高斯脉冲展宽)，替代 Log-Cauchy 数学近似

**Block 数据流 (GCN 版)**：

```text
[单 Block 数据流向 — PyG SAGEConv 版 (SE gate)]
 f(B,C,N) + p(B,4,N) + p_edge_index(2,E) [预计算缓存]
  │
  ├── KNN from f (特征空间, 每层重算)   p_edge_index (坐标图边, 入口预计算复用)
  │        ↓                                    │
  ├── [GCN_f]                              [GCN_p]
  │   batched_knn_edge_index(f_knn)        SAGEConv(p) + BN1d + LReLU
  │   SAGEConv(f) + BN1d + LReLU                │
  │       ↓                                     ↓
  │   f_gcn (B,C_out,N)                   p_gcn (B,C_out,N)
  │       |                                     |
  │   GAP → Linear → ReLU → Linear → Sigmoid    |
  │       ↓ (SE weight)                         |
  │   f_gcn_gated (B,C_out,N)                   |
  │       |                                     |
  │     ┌──────────────────────────────────┐
  │     │    SE-style Channel Gate         │
  │     │                                   │
  │     │  cat([f_gcn_gated, p_gcn])       │
  │     │  → Conv1d+BN1d → fused           │
  │     └──────────────────────────────────┘
  │                 │
  │                 ↓
  └──> gate · fused + (1-gate) · coord_res ──> LeakyReLU ──> f_out
        gate = σ(Conv1d+BN(P[4D]))                            p 不变
```

**模型规模与显存**：

- **参数量**：约 1.56 M (SE gate + Conv1d 融合替代 Q/K/V 注意力，参数略减)
- **显存**：RTX 4070 SUPER 上 $B{=}32$ 训练峰值约 **0.5 GB** (原版 v7 约 7.4 GB) — SAGEConv 消息传递 + SE gate 比 Conv2d EdgeConv + 全局注意力更轻量
- **训练命令**：默认使用固定权重 $\lambda_{cls} \cdot \mathcal L_{cls} + \lambda_{depth} \cdot \mathcal L_{depth}$

```powershell
$env:PYTHONPATH = "D:\PYproject\SPAD"
& "D:\anaconda3\envs\pytorch\python.exe" "D:\PYproject\SPAD\scripts\train.py" --model graph_residual_gcn --batch-size 32 --epochs 100
```

**设计动机**：SAGEConv 的 mean 聚合天然保留邻域分布统计信息，适合 SPAD 点云的噪声建模；中心-邻居分离权重 ($W_1, W_2$) 表达力更强；归纳式学习可处理未见过的图拓扑 (如不同噪声强度下的 KNN 图变化)。SE channel gate 替代全局注意力，消除 1024×1024 注意力矩阵的过拟合风险与显存开销。

#### GCN 版优化路线 (v2)

基于首轮 100 epoch 训练结果诊断 (train_top1=0.993 vs val_top1=0.872，过拟合 gap 达 12.9%)，已完成以下优化：

**P0 — 移除全局注意力 (已完成)**：原 SAGEConv 已将邻域聚合为单向量 (无邻居维度 $k$)，被迫做 $(B,N,N)$ 全局 softmax — 对 1024 点全连接图注意力既嘈杂又过拟合。改为 SE-style channel gate + Conv1d 直接融合 $f_{gcn}$ 与 $p_{gcn}$，消除 1024×1024 注意力矩阵 (~512 MB 中间显存)。

**统一 Box 头 (已完成)**：原 centroid-offset 预测依赖质心先验，与 baseline (DGCNN/PointNet++ 等) 的直接回归方式不一致，导致对比实验不公平。改为 `box_pred = self.box_head(f_pooled)` 直接回归，确保 backbone 成为唯一变量。

**Soft-histogram depth loss (已完成)**：原 Log-Cauchy 为数学近似，改为直接建模 SPAD 物理过程：
$$\mathcal L_{depth} = \sum_{d \in \{x,y,z\}} \sum_{k=-K}^{K} w_k \cdot (\hat c_d - (c_d^{gt} + k \cdot \delta_d))^2$$
其中 $\delta_d$ 为各维度 bin 宽度 (归一化空间: $x/y$ 为 $1/63$, $z$ 为 $1/108$)，$w_k \propto e^{-k^2/2\sigma^2}$ 为高斯权重 ($K{=}2$, $\sigma{=}1.5$)。

**固定权重 (已完成)**：原 Kendall 自适应权重导致训练 loss 变负 (log-variance 项无约束增长)，改为 $\lambda_{cls} \cdot \mathcal L_{cls} + \lambda_{depth} \cdot \mathcal L_{depth}$ (默认 $\lambda_{cls} = \lambda_{depth} = 1.0$)。

**待优化**：

**P1 — 缩减 head**：cls_head 占参数 40% (664K)，box_head 占 18% (298K)，head 总参数 1.23M → 目标降至 ~0.2M。

**P2 — 通道缩减**：stem 4→16, block 通道减半，总参数目标 ~0.4–0.5M (当前 1.56M 的 ~30%)。

### 任务 2：面向特定领域（输电杆塔）的数据模拟与算法验证
**目标与动机**：为了验证所提算法（任务1）在真实工程场景中的可用性与泛化能力，将研究目标投射到电网巡检领域的具体应用上。

**研究内容**：
- **物理与噪声模拟**：将现有的高精度输电杆塔点云转换为带有时间 bin 属性的单光子网格模拟数据。引入符合伽马分布的雾回波模型，并混入高斯目标信号。
- **验证与评估**：通过任务 1 的模型进行训练与推理，采用分类 Top-1/Top-3 准确率、以及 3D Box IoU / AP（平均精度）作为量化指标，评估算法的鲁棒性。

### 任务 3：基于深度隐变量模型的全网格直方图建模（无监督/半监督）
**目标与动机**：突破“先降采样后检测”所造成的信息瓶颈，摒弃传统峰值提取。直接对 $64\times64\times128$ 的全网格直方图数据进行处理，实现对每个像素物理分布特征的深入挖掘。

**方法论（Histogram-based Deep Latent Model）**：
- **统计建模假设**：对于任意像素 $p$，其时间序列 $\mathbf I_p \in \mathbb R_+^{T}$，引入低维连续隐变量 $\mathbf z_p$ 来捕获潜在的信号混合模式（目标 vs 雾）。其概率生成模型为：
  $$p(\mathbf I_p) = \int p(\mathbf I_p\mid\mathbf z_p)\,p(\mathbf z_p)\,d\mathbf z_p$$
- **观测分布设计**：针对光子探测物理过程，可选用泊松 (Poisson)、负二项分布 (Negative Binomial) 或 Gamma-Poisson 混合模型作为似然函数 $p(\mathbf I_p\mid\mathbf z_p)$。
- **网络结构设计**：
  1. **编码器 (Encoder)**：$\mathbf I_p \rightarrow {\mu_p, \sigma_p} \rightarrow \mathbf z_p$。提取序列的统计特征。
  2. **解码器 (Decoder)**：$\mathbf z_p \rightarrow \hat{\lambda}_{p,t}$。重构时间 bin 上的光子到达率分布。
  3. **任务预测头 (Task Head)**：$\mathbf z_p \rightarrow \pi_{p,c}$，直接输出像素级的类别概率 $\Pi\in\mathbb R^{P\times C}$。
- **损失函数设计**：
  - 重构损失（基于负对数似然）：$\mathcal L_{rec}(\mathbf I_p, \hat{\mathbf I}_p)$
  - 隐空间正则化（KL散度）：$\mathcal L_{kl}(q(\mathbf z_p)\|p(\mathbf z_p))$
  - 分类一致性损失（如提供弱监督或监督标签）：$\mathcal L_{cls}$

---

## 3. 对比实验与评估方案

- **针对降采样图谱架构（任务1与2）**：
  - 将开发的”图残差网络”与 13 个基线模型在同等降采样的单光子数据集上开展消融实验与横向对比。基线覆盖经典与前沿方法：PointNet, PointNet++, DGCNN, CurveNet, PointMLP, PointNeXt, GDANet, PointTransformerV2, PointTransformerV3, SimplePointTransformer, PointMAE, PointRWKV, 3DETR, UPP 等。
  - **内部消融**：对比原版 Graph Residual (DGCNN EdgeConv) 与 PyG GCN 变体 (SAGEConv)，验证真图卷积对单光子点云噪声建模的有效性。
  - 主要评估指标：分类 Top-1/Top-3 准确率、3D Box IoU、3D Box Center L1。

- **针对全数据网格隐变量架构（任务3）**：
  - 可视化模型输出的逐像素分类概率图 (Probability Map)。
  - 展示模型在浓雾（Gamma强噪声）与信噪比极低情形下的去噪/分类性能，证明直方图层面隐变量建模相较于硬阈值降采样的先进性与理论优势。

## 4. 文档约定
- 本文档路径为 [model/readme.md](model/readme.md)，作为该硕士课题建模与实验设计的核心指引文件。
- 架构的搭建、参数调优及对比实验的开展，均须遵循本文件确立的逻辑链路，以保证研究工作的严谨性、递进性和自洽性。