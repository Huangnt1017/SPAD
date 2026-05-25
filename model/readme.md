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

- **Stem 层**：两层线性映射将原始 4 维输入升维至 32 维初始特征：$(B, N, 4) \to (B, N, 32)$。

- **Intensity 加权 4D KNN 构图 (双向门控)**：
  近邻搜索综合几何距离与 intensity 差异：
  $$d_{ij} = \|\mathbf p_i - \mathbf p_j\|^2 \cdot (1 - \alpha + 2\alpha \cdot \hat d^{(i)}_{ij})$$
  其中 $\hat d^{(i)}_{ij} \in [0,1]$ 为归一化的 intensity 距离，$\alpha{=}0.3$ 控制门控强度。效果：
  - 强度相似的点对：几何距离被缩小 30%，更容易成为邻居（聚合同目标信号）；
  - 强度差异最大的点对：几何距离被放大 30%，抑制噪点混入真实目标邻域。

- **主干网络：图残差模块 (Graph Residual Block) × 4**
  核心思想是通过**标准 Q/K/V 图注意力**在提取深层语义的同时，用**坐标门控跳跃连接**显式保留几何约束。单个 Block 数据流如下：
  1. **Flow_A (图分支)**：特征经层归一化 (`LN`) 后，通过 KNN 索引收集邻居，构造边特征 $[\mathbf f_i,\; \mathbf f_j{-}\mathbf f_i,\; \mathbf p_j{-}\mathbf p_i]$（坐标差为完整 4D 含 intensity），经共享 MLP 得到 **per-neighbor** 图特征 $\mathbf F^k \in \mathbb R^{B \times N \times k \times C}$。
  2. **Flow_B (点分支)**：基于同一归一化特征，通过 `Linear` 投影得到中心点特征 $\mathbf F \in \mathbb R^{B \times N \times C}$。
  3. **标准 Q/K/V 图注意力 (Scaled Dot-Product Graph Attention)**：
     $$\mathbf Q_i = W_q(\mathbf F_i),\quad \mathbf K_{ij} = W_k(\mathbf F^k_{ij}),\quad \mathbf V_{ij} = W_v(\mathbf F^k_{ij})$$
     $$\alpha_{ij} = \text{softmax}_j\left(\frac{\mathbf Q_i \cdot \mathbf K_{ij}}{\sqrt{C_{out}}}\right),\quad \mathbf{attn}_i = \sum_j \alpha_{ij} \cdot \mathbf V_{ij}$$
     Q 来自中心点语义 (Flow_B)，K/V 来自图构建的邻域上下文 (Flow_A)，注意力在 k 个邻居上做 softmax，学习自适应聚合权重。
  4. **坐标门控跳跃连接 (Coordinate-Gated Residual)**：
     原始 4D 坐标 $\mathbf P = (x, y, z, i)$ 同时产生门控信号和坐标信息：
     $$\mathbf g = \sigma(W_{gate}(\mathbf P)),\quad \mathbf c = W_{res}(\mathbf P)$$
     $$\mathbf{out} = \mathbf g \odot W_{out}(\mathbf{attn}) + (1 - \mathbf g) \odot \mathbf c$$
     门控含义：$\mathbf g \to 1$ 信任注意力聚合的语义特征；$\mathbf g \to 0$ 信任原始坐标+强度信息。该设计使模型在噪声区域可动态偏向语义特征，而在几何清晰区域保留位置约束。
  5. **层间下采样**：基于特征 L2 范数的加权无放回随机采样将点数减半，新点集上重建 intensity 加权 4D KNN 图。
  
  ```text
  [单 Block 数据流向]
   Input (P[4D], Feature)
    │           \      /
    │            [ LN ]
    │            /    \
    │        [NGF]   [Linear]    ← Flow_A: 边特征 MLP (per-neighbor, 含 4D 坐标差)
    │          |        |           Flow_B: 中心特征投影
    │        [GCN]      |
    │          ↓        ↓
    │       Fk(B,N,k,C) F(B,N,C)
    │          |        |
    │        [W_k]    [W_q]      ← Q/K/V 投影
    │        [W_v]      |
    │          ↓        ↓
    │         K,V       Q
    │           \      /
    │    softmax(Q·K/√C) @ V     ← Scaled Dot-Product Attention over k neighbors
    │              │
    │           [Linear]         ← 特征映射
    │              │
    └───> [Coord Gate] ⊗ / ⊕ ──> ReLU ──> [downsample] ──> Output
           P(4D) → σ(W_gate)     gate·mapped + (1-gate)·W_res(P)
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

- **模型规模**：约 2.82 M 参数；RTX 4070 SUPER 上 $B{=}32$ 训练峰值显存约 859 MB。

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