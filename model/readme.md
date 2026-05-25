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
- **输入**：降采样点云矩阵 $(B, N, 4)$，$N{=}1024$，包含 $x, y, z$ 与反射强度 $i$。

- **Stem 层**：两层 `Conv1d + BN1d + LeakyReLU` 将原始 4 维输入升维至 32 维初始特征：$(B, 4, N) \to (B, 32, N)$。全程采用 channel-first $(B, C, N)$ 布局以适配卷积与批归一化。

- **动态特征空间 KNN (GPU)**：
  每个 Block 内部从**当前学到的特征** $\mathbf f \in \mathbb R^{B \times C \times N}$ 重新构建 $k{=}20$ 近邻图（DGCNN "Dynamic Graph" 范式），使图结构随网络学习动态演化。KNN 通过负平方距离 + topk 实现，全程 GPU matmul，无需 CPU 回传。

- **主干网络：图残差模块 (Graph Residual Block) × 4**
  核心思想是通过**双流图卷积 + 标准 Q/K/V 注意力**提取语义，再用**坐标门控跳跃连接**显式保留几何约束。单个 Block 的数据流如下：

  **Step 1: 动态构图**
  从当前层特征 $\mathbf f$ 用 GPU KNN 构建 $k$ 近邻索引 $\text{idx} \in \mathbb Z^{B \times N \times k}$。

  **Step 2: 双流 EdgeConv (Conv2d + BN2d + LeakyReLU)**
  在同一 KNN 图上，分别对特征和坐标做 DGCNN 风格 EdgeConv：
  - **GCN_f (特征 EdgeConv)**：`get_graph_feature(f)` → $[\mathbf f_j{-}\mathbf f_i, \mathbf f_i] \in \mathbb R^{2C_{in}}$, 经 `Conv2d(2C_in, C_out, 1) + BN2d + LeakyReLU` → $\mathbf F^k \in \mathbb R^{B \times C_{out} \times N \times k}$，编码邻域**语义关系**。
  - **GCN_p (位置 EdgeConv)**：`get_graph_feature(p)` → $[\mathbf P_j{-}\mathbf P_i, \mathbf P_i] \in \mathbb R^{8}$ (完整 4D×2), 经 `Conv2d(8, C_out, 1) + BN2d + LeakyReLU` → $\mathbf P^k \in \mathbb R^{B \times C_{out} \times N \times k}$，编码邻域**几何关系**。

  **Step 3: 标准 Q/K/V 注意力 (Scaled Dot-Product Graph Attention)**
  三路分别投影后做标准注意力运算：
  $$\mathbf Q_i = \text{Conv1d}_{q}([\mathbf f_i \| \mathbf P_i]) \quad \text{(中心点联合查询, Conv1d+BN1d)}$$
  $$\mathbf K_{ij} = \text{Conv2d}_{k}(\mathbf P^k_{ij}) \quad \text{(位置 EdgeConv → Key)}$$
  $$\mathbf V_{ij} = \text{Conv2d}_{v}(\mathbf F^k_{ij}) \quad \text{(特征 EdgeConv → Value)}$$
  $$\alpha_{ij} = \text{softmax}_j\left(\frac{\mathbf Q_i \cdot \mathbf K_{ij}}{\sqrt{C_{out}}}\right),\quad \mathbf{attn}_i = \sum_j \alpha_{ij} \cdot \mathbf V_{ij}$$

  > **注意力机制的物理含义**:
  > - $\mathbf K$ 由位置 EdgeConv 产生 → 注意力分数**由几何关系主导**，空间相关邻居获高权重；
  > - $\mathbf V$ 由特征 EdgeConv 产生 → 聚合内容为**邻域语义信息**；
  > - $\mathbf Q$ 由中心点 $(\mathbf f \| \mathbf P)$ 经 `Conv1d + BN1d` 产生 → **联合查询身份**。
  > 解耦设计: "关注谁"(K, 几何驱动) 与 "聚合什么"(V, 语义驱动) 分离。

  **Step 4: 坐标门控跳跃连接 (Coordinate-Gated Residual)**
  原始 4D 坐标 $\mathbf P = (x, y, z, i)$ 同时产生门控信号和坐标信息：
  $$\mathbf g = \sigma(W_{gate}(\mathbf P)),\quad \mathbf c = W_{res}(\mathbf P)$$
  $$\mathbf{out} = \mathbf g \odot W_{out}(\mathbf{attn}) + (1 - \mathbf g) \odot \mathbf c$$
  经 LeakyReLU 激活。门控含义：$\mathbf g \to 1$ 信任语义；$\mathbf g \to 0$ 信任坐标。

  **Step 5: 层间下采样**
  基于特征 L2 范数的加权无放回随机采样 (`torch.gather`) 将点数减半。
  
  ```text
  [单 Block 数据流向]
   f(B,C,N) + p(B,4,N)
    │
    ├── Dynamic KNN from f (GPU) → idx(B,N,k)
    │
    ├── [GCN_f]                         [GCN_p]
    │   get_graph_feature(f,idx)        get_graph_feature(p,idx)
    │   → (B, 2C, N, k)                → (B, 8, N, k)
    │   Conv2d+BN2d+LeakyReLU          Conv2d+BN2d+LeakyReLU
    │       ↓                               ↓
    │   Fk (B,C_out,N,k)               Pk (B,C_out,N,k)
    │       |                               |
    │   Conv2d [W_v]                    Conv2d [W_k]
    │       ↓                               ↓
    │       V                               K
    │        \       Q = Conv1d+BN(f‖p)    /
    │         \             |             /
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
    └──> gate · mapped + (1-gate) · coord_res ──> LeakyReLU ──> downsample
          gate = σ(Conv1d+BN(P[4D]))                             N → N/2
  ```
  
  通道与点数阶梯（4 层 Block，每层通道翻倍、点数减半）：
  | Block | $C_{in} \to C_{out}$ | $N_{in} \to N_{out}$ |
  |:-----:|:--------------------:|:--------------------:|
  | 1     | 32 → 64              | 1024 → 512           |
  | 2     | 64 → 128             | 512 → 256            |
  | 3     | 128 → 256            | 256 → 128            |
  | 4     | 256 → 512            | 128 → 64             |

- **全局池化**：对最后一层 64 个点的 512 维特征同时做 max-pool 与 avg-pool，拼接得到 1024 维全局描述子。

- **双任务预测头**：
  - **分类头**：$\text{Linear}(1024,256) \to \text{BN} \to \text{ReLU} \to \text{Dropout} \to \text{Linear}(256, C)$，输出分类 logits $\mathbb R^{B\times C}$。
  - **Box 头**：结构同分类头，输出 3 维中心坐标 $\mathbb R^{B\times 3}$（归一化空间下的 $\hat x, \hat y, \hat z$）。推理时由固定归一化半宽 `FIXED_BBOX_HALF_SIZE_NORMALIZED` 重建完整 3D 边界框。

- **模型规模**：约 2.53 M 参数；RTX 4070 SUPER 上 $B{=}32$ 训练峰值显存约 2173 MB。

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
  - 主要评估指标：分类 Top-1/Top-3 准确率、3D Box IoU、3D Box Center L1。

- **针对全数据网格隐变量架构（任务3）**：
  - 可视化模型输出的逐像素分类概率图 (Probability Map)。
  - 展示模型在浓雾（Gamma强噪声）与信噪比极低情形下的去噪/分类性能，证明直方图层面隐变量建模相较于硬阈值降采样的先进性与理论优势。

## 4. 文档约定
- 本文档路径为 [model/readme.md](model/readme.md)，作为该硕士课题建模与实验设计的核心指引文件。
- 架构的搭建、参数调优及对比实验的开展，均须遵循本文件确立的逻辑链路，以保证研究工作的严谨性、递进性和自洽性。