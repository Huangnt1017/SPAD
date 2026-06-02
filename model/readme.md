# 单光子激光雷达 (SPAD) 目标检测与分布建模研究框架

## 分类用模型参数量一览 (截至 2026-06-02)

| 模型 | 类型 | 参数量 (M) |
|:-----|:----:|:----------:|
| PointMLP-Elite | baseline | 0.644 |
| Graph Residual GCN (PyG SAGEConv) | 自研 | 1.290 |
| Graph Residual (DGCNN EdgeConv) | 自研 | 1.288 |
| PointNet++ SSG | baseline | 1.403 |
| PointNet++ MSG | baseline | 1.675 |
| DGCNN | baseline | 1.738 |
| PointTransformer | baseline | 2.317 |
| PointNet | baseline | 3.399 |
| 3DETR | baseline | 4.011 |
| SPT (脉冲神经网络) | baseline | 9.826 |
| PointTransV2 | baseline | 9.943 |
| PointBERT | baseline | 11.683 |
| PointMLP | baseline | 13.166 |
| PointTransV3 | baseline | 15.930 |
| PointMAE | baseline | 22.290 |
| UPP | baseline | 30.811 |
| PointRWKV | baseline | 64.230 |

> 自研模型 Graph Residual 系列参数量最小 (~1.2-1.3M)，在所有对比模型中属于轻量级。

---

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
| 分类头 | `build_standard_cls_head(1024, C)` | $(B,1024) \to (B,C)$ | 3 层 MLP (1024→256→128→C), BN + LeakyReLU(0.2) + Dropout(0.3) |
| Box 头 | `build_standard_box_head(1024, 3)` | $(B,1024) \to (B,3)$ | 3 层 MLP (1024→256→128→3), 直接回归中心点 |

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

#### Box 头: 直接回归 (Center-Only)

Box 头从全局池化特征直接预测中心点坐标 $(B, 1024) \to (B, 3)$：
$$\hat{\mathbf c} = \text{MLP}(\mathbf f_{pool})$$

统一头结构 (由 `utils/heads.py` 中 `build_standard_cls_head` / `build_standard_box_head` 构建):

$$\text{cls\_head}: \text{Linear}(1024 \to 256) \to \text{BN} \to \text{LeakyReLU}(0.2) \to \text{Drop}(0.3) \to \text{Linear}(256 \to 128) \to \text{BN} \to \text{LeakyReLU}(0.2) \to \text{Drop}(0.3) \to \text{Linear}(128 \to C)$$

$$\text{box\_head}: \text{Linear}(1024 \to 256) \to \text{BN} \to \text{LeakyReLU}(0.2) \to \text{Drop}(0.3) \to \text{Linear}(256 \to 128) \to \text{BN} \to \text{LeakyReLU}(0.2) \to \text{Linear}(128 \to 3)$$

> **设计原因**：为确保与 baseline (DGCNN / PointNet++ 等) 的对比公平性，**所有 14 个模型**统一使用相同的头结构 (中间维度 256→128、LeakyReLU 激活、Dropout=0.3)，不依赖质心先验。推理时由固定归一化半宽 `FIXED_BBOX_HALF_SIZE_NORMALIZED` 重建完整 3D 边界框。

> **例外**: SPT (脉冲神经网络) 使用 Conv1d + SpikeNode 但中间维度对齐；3DETR (query-based) 使用 per-query GenericMLP。


#### 损失函数

仅 **2 项损失**，采用 **固定权重** $\lambda_{cls} \cdot \mathcal L_{cls} + \lambda_{depth} \cdot \mathcal L_{depth}$ (默认 $\lambda_{cls} = \lambda_{depth} = 1.0$):

$$\mathcal L = 1.0 \cdot \mathcal L_{cls} + 1.0 \cdot \mathcal L_{depth}$$

> **设计原因**：Kendall 自适应权重 ($e^{-s}\mathcal L + s$) 在训练中 log-variance 项会无约束增长，导致总 loss 变负且训练不稳定。固定权重更稳定、论文对比更公平。

**SPAD Soft-histogram depth loss** $\mathcal L_{depth}$ 直接建模单光子激光雷达物理过程 — 时间 bin 量化 + 高斯脉冲展宽：

$$\mathcal L_{depth} = \sum_{d \in \{x,y,z\}} \sum_{k=-K}^{K} w_k \cdot \left(\hat c_d - (c_d^{gt} + k \cdot \delta_d)\right)^2$$

其中：
- $\delta_d$ 为各维度归一化 bin 宽度：$\delta_x = \delta_y = 1/63$，$\delta_z = 1/108$
- $w_k \propto \exp(-k^2 / 2\sigma^2)$ 为高斯权重 ($K=2$, $\sigma=1.5$)，共 5 个 bin 加权求和
- 将 GT 深度建模为具有物理不确定度的分布，而非硬点目标

| 特性 | 说明 |
|:----:|:----:|
| **物理含义** | 直接对应 SPAD 时间 bin 量化 (δ) + 高斯脉冲响应 (w) |
| **维度自适应** | z 轴 bin 宽度 $1/108$ 使深度方向分辨力天然高于 x/y ($1/63$) |
| **平滑梯度** | 高斯加权的 MSE 求和，任意位置均有非零梯度 |
| **鲁棒性** | 多 bin 加权相当于 soft-histogram 拟合，对噪声 bin 不敏感 |

两个损失分项：

| 损失项 | 公式 | 说明 |
|:-----:|:----:|:----:|
| $\mathcal L_{cls}$ | CrossEntropy | 26 类分类 |
| $\mathcal L_{depth}$ | $\sum_k w_k (\hat c_d - c_d^{gt} - k\delta_d)^2$ | Soft-histogram 中心回归 (SPAD 物理驱动) |

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
| **分类 Pooling** | max+avg 全局池化 | max+avg 全局池化 (类别与位置无关) |
| **Box Pooling** | max+avg 全局池化 | max+avg 全局池化 (backbone 内 coord_encoder 注入位置信息) |
| **Block 改进** | Q/K/V 注意力 + EdgeConv | SE gate + SAGEConv + **坐标编码残差注入** |
| **深度 Loss** | Soft-histogram (物理驱动) | Soft-histogram (物理驱动) |
| **头架构** | `build_standard_cls_head/box_head` | `build_standard_cls_head/box_head` |

**架构变化**：

1. **GCN_f (特征流)**：SAGEConv on 特征空间动态 KNN 图 (DGCNN "动态图" 范式保留，卷积核换为真 GNN)
2. **GCN_p (位置流)**：SAGEConv on 坐标空间静态 KNN 图 (预计算复用，同原版)
3. **SE channel gate**：因 SAGEConv 已将邻域聚合为单向量，全局 $(B,N,N)$ softmax 注意力既嘈杂又过拟合；改为 SE-style channel gate 对 $f_{gcn}$ 做通道加权，再与 $p_{gcn}$ 通过 Conv1d 融合
4. **坐标编码残差注入**：每个 Block 新增 `coord_encoder`（2 层 Conv1d MLP），对 4D 原始坐标提取位置特征，通过残差加法注入到最终特征。使每个点特征显式包含其空间位置编码，全局池化后位置统计量仍被保留，改善深度回归
5. **统一头架构**：两个变体均使用 `utils/heads.py` 构建标准 cls_head (1024→256→128→C) 和 box_head (1024→256→128→3)，确保 backbone 为唯一变量
6. **Soft-histogram depth loss**：直接建模 SPAD 物理过程 (时间 bin 量化 + 高斯脉冲展宽)，固定权重 $\lambda_{cls}=\lambda_{depth}=1.0$

**坐标编码残差注入 — 改善深度回归的 Block 内改进**：

全局池化 (max+avg) 将 $(B, 512, N)$ 压成 $(B, 512)$，所有空间位置信息丢失，导致深度估计只能从统计量中"盲猜"中心点。改进方案在 Block 内部将位置信息注入到每个点特征中：

$$\mathbf{pos} = \text{coord\_encoder}(\mathbf P) \in \mathbb R^{B \times C_{out} \times N}$$

$$\text{coord\_encoder}: \text{Conv1d}(4 \to C_{out}) \to \text{BN} \to \text{LeakyReLU}(0.2) \to \text{Conv1d}(C_{out} \to C_{out}) \to \text{BN}$$

$$\mathbf f_{out} = \text{act}(\mathbf{out} + \mathbf{pos})$$

设计动机：
- **不改变 head 架构**：backbone 内部改进，head 结构完全一致，对比公平性保持
- **参数量极小**：4 个 Block 共新增 ~0.09M (原版 1.194M → 改进 1.29M)
- **残差加法**：每个点 $\mathbf f_{out}[:, :, i]$ 显式包含 $\mathbf P[:, :, i]$ 的 2 层 MLP 编码
- **池化后保留**：即使全局 max+avg 池化，池化结果也自然聚合了各点的位置统计信息
- **与 coord_gate / coord_res 的区别**：后两者是门控混合（权重由 sigmoid 决定），coord_encoder 是残差加法（位置信息强制注入）

**Block 数据流 (GCN 版)**：

```text
[单 Block 数据流向 — PyG SAGEConv 版 (SE gate + 坐标编码注入)]
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
  │     │    双流融合                       │
  │     │  cat([f_gcn_gated, p_gcn])       │
  │     │  → Conv1d+BN1d → fused           │
  │     └──────────────────────────────────┘
  │                 │
  │                 ↓
  │     ┌──────────────────────────────────┐
  │     │    坐标门控跳跃                    │
  │     │  gate = σ(coord_gate(p))         │
  │     │  coord_info = coord_res(p)       │
  │     │  out = gate·fused + (1-gate)·ci  │
  │     └──────────────────────────────────┘
  │                 │
  │                 ↓
  │     ┌──────────────────────────────────┐
  │     │    坐标编码残差注入 ★ 新增         │
  │     │  pos_feat = coord_encoder(p)     │
  │     │    (Conv1d+BN+LReLU+Conv1d+BN)  │
  │     │  f_out = act(out + pos_feat)     │
  │     └──────────────────────────────────┘
  │                 │
  │                 ↓
  └─────────────────> f_out (B,C_out,N)    p 不变
```

**模型规模与显存**：

- **参数量**：约 1.29 M (coord_encoder 新增 ~0.09M，相比原版 1.194M 增幅仅 ~8%)
- **显存**：RTX 4070 SUPER 上 $B{=}32$ 训练峰值约 **908 MB** (原版 v7 约 7.4 GB) — SAGEConv 消息传递 + SE gate 比 Conv2d EdgeConv + 全局注意力更轻量
- **对比公平性**：head 架构与所有 baseline 完全一致 (统一 max+avg pooling + 标准 cls/box head)，backbone 内部改进不影响对比公平
- **训练命令**：默认使用固定权重 $\lambda_{cls} \cdot \mathcal L_{cls} + \lambda_{depth} \cdot \mathcal L_{depth}$

```powershell
$env:PYTHONPATH = "D:\PYproject\SPAD"
& "D:\anaconda3\envs\pytorch\python.exe" "D:\PYproject\SPAD\scripts\train.py" --model graph_residual_gcn --batch-size 32 --epochs 100
```

**设计动机**：SAGEConv 的 mean 聚合天然保留邻域分布统计信息，适合 SPAD 点云的噪声建模；中心-邻居分离权重 ($W_1, W_2$) 表达力更强；归纳式学习可处理未见过的图拓扑 (如不同噪声强度下的 KNN 图变化)。SE channel gate 替代全局注意力，消除 1024×1024 注意力矩阵的过拟合风险与显存开销。Block 内坐标编码残差注入 (coord_encoder) 改善深度回归，使每个点特征显式包含其空间位置编码，全局池化后位置统计量仍被保留。

#### GCN 版优化路线 (v2)

基于首轮 100 epoch 训练结果诊断 (train_top1=0.993 vs val_top1=0.872，过拟合 gap 达 12.9%)，已完成以下优化：

**P0 — 移除全局注意力 (已完成)**：原 SAGEConv 已将邻域聚合为单向量 (无邻居维度 $k$)，被迫做 $(B,N,N)$ 全局 softmax — 对 1024 点全连接图注意力既嘈杂又过拟合。改为 SE-style channel gate + Conv1d 直接融合 $f_{gcn}$ 与 $p_{gcn}$，消除 1024×1024 注意力矩阵 (~512 MB 中间显存)。

**统一 Box 头 + 统一头架构 (已完成)**：原 centroid-offset 预测依赖质心先验，与 baseline (DGCNN/PointNet++ 等) 的直接回归方式不一致，导致对比实验不公平。`graph_residual.py` 和 `graph_res_GCN.py` 均已改为 `box_pred = self.box_head(f_pooled)` 直接回归。进一步地，所有模型 (14 个 baseline + 自研) 的 cls_head 和 box_head 现统一使用 `utils/heads.py` 中的标准构建函数，中间维度 256→128、LeakyReLU(0.2)、Dropout(0.3)，确保 **backbone 是唯一变量**。

**Soft-histogram depth loss (已完成)**：原 Log-Cauchy 为数学近似，改为直接建模 SPAD 物理过程：
$$\mathcal L_{depth} = \sum_{d \in \{x,y,z\}} \sum_{k=-K}^{K} w_k \cdot (\hat c_d - (c_d^{gt} + k \cdot \delta_d))^2$$
其中 $\delta_d$ 为各维度 bin 宽度 (归一化空间: $x/y$ 为 $1/63$, $z$ 为 $1/108$)，$w_k \propto e^{-k^2/2\sigma^2}$ 为高斯权重 ($K{=}2$, $\sigma{=}1.5$)。

**固定权重 (已完成)**：原 Kendall 自适应权重导致训练 loss 变负 (log-variance 项无约束增长)，改为 $\lambda_{cls} \cdot \mathcal L_{cls} + \lambda_{depth} \cdot \mathcal L_{depth}$ (默认 $\lambda_{cls} = \lambda_{depth} = 1.0$)。

**Block 内坐标编码残差注入 (已完成)**：原 GCN 变体使用 max+avg 全局池化后接 box_head，空间信息完全丢失导致深度估计效果差。在每个 Block 内新增 `coord_encoder`（2 层 Conv1d MLP），对 4D 原始坐标提取位置特征，通过残差加法注入到最终特征：
$$\mathbf f_{out} = \text{act}(\mathbf{out} + \text{coord\_encoder}(\mathbf P))$$
使每个点特征显式包含其空间位置编码，全局池化后位置统计量仍被保留。相比 head 端改进 (如 BoxQueryPool)，block 内改进参数量极小 (4 Block 共 ~0.09M)，head 架构与所有 baseline 完全一致，对比公平性保持。总参数量从 1.194M 增至 1.29M。

**待优化**：

**P1 — 通道缩减**：stem 4→16, block 通道减半，总参数目标 ~0.4–0.5M (当前 1.29M 的 ~30%)。

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

---

## 5. 完整模型代码 (graph_res_GCN.py)

以下为 PyG SAGEConv 变体的完整实现代码 (截至 2026-06-02)，包含坐标编码残差注入：

```python
"""
单光子点云图残差多任务网络 — PyG 图卷积版 (Graph Residual Multi-Task Network, GCN variant)

与 graph_residual.py (v7) 的核心区别:
    原版使用 DGCNN 风格 Conv2d + BN2d 做 EdgeConv ([x_j - x_i, x_i] → MLP);
    本版使用 PyTorch Geometric 的 SAGEConv 做 **真正的消息传递图卷积**:

        h_i' = W_1 · h_i + W_2 · mean_{j ∈ N(i)} h_j     (SAGEConv, mean 聚合)

    两路均替换:
    - GCN_f (特征流): SAGEConv on 特征空间动态 KNN 图 (DGCNN "动态图" 范式保留, 卷积核换成真 GNN)
    - GCN_p (位置流): SAGEConv on 坐标空间静态 KNN 图 (预计算复用, 同原版)

    Block 内改进 (相比原版 GraphResidualBlock):
    - SE-style channel gate 替代全局 Q/K/V 注意力
    - **坐标编码器残差注入**: 2 层 MLP 对 4D 坐标提取位置特征, 通过残差加法注入到最终特征,
      使每个点特征显式包含其空间位置编码, 改善深度回归 (避免全局池化后位置信息丢失)

    其余 Q/K/V 注意力 + 坐标门控残差逻辑不变 (详见 readme.md 任务 1)。

Block 数据流 (全程 (B, C, N) 布局, 无下采样, N=1024):
    f(B,C,N) + p(B,4,N) + p_edge_index(E_total,2) [预计算缓存]
        ↓
    Dynamic KNN from f → idx(B,N,k)   ← 仅特征 KNN 每层重算
    → f_edge_index(E_total,2)
        ↓
    ┌ GCN_f: SAGEConv(f) + BN1d + LReLU → f_gcn(B,C_out,N)   ← V source
    └ GCN_p: SAGEConv(p) + BN1d + LReLU → p_gcn(B,C_out,N)   ← K source (复用预计算边)
        ↓
    SE gate on f_gcn, 然后 cat(f_gcn_gated, p_gcn) → Conv1d → fused
        ↓
    coord_gate(p), coord_res(p): gate·fused + (1-gate)·coord_res → out
        ↓
    coord_encoder(p): pos_feat  ← 新增: 2 层 MLP 提取位置特征
        ↓
    f_out = act(out + pos_feat)  ← 坐标编码残差注入

References:
    - model/readme.md 任务 1
    - model/graph_residual.py (v7 原版)
    - Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017 (GraphSAGE)
    - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

from utils.heads import build_standard_cls_head, build_standard_box_head

try:
    from torch_geometric.nn import SAGEConv
except ImportError as exc:
    raise ImportError(
        "graph_res_GCN 依赖 torch_geometric (PyG)。\n"
        "安装指南: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html\n"
        "典型命令 (conda): conda install pyg -c pyg\n"
        "或 (pip):       pip install torch_geometric"
    ) from exc

try:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _HAS_CKPT = True
except (ImportError, AttributeError):
    _HAS_CKPT = False

    def _ckpt(fn, *args, **kwargs):
        return fn(*args)


# ══════════════════════════════════════════════════
# GPU KNN (DGCNN 风格: 特征空间, 负距离 topk)
# ══════════════════════════════════════════════════

def knn_gpu(x: torch.Tensor, k: int) -> torch.Tensor:
    """在特征空间做 KNN (与 DGCNN 一致)。

    用负平方距离 + topk 实现, 全程 GPU matmul, 无需排序。

    Args:
        x: (B, C, N) — 任意维特征 (坐标 / 学到的特征均可)。
        k: 近邻数 (含自身, 因距离为 0 时自身总是最近邻)。

    Returns:
        idx: (B, N, k), int64。
    """
    # (B, N, N) 负平方距离: 越大越近
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    neg_dist = -xx - inner - xx.transpose(2, 1)
    _, idx = neg_dist.topk(k=k, dim=-1)
    return idx


# ══════════════════════════════════════════════════
# 批量 edge_index 构建 (KNN 索引 → PyG 格式)
# ══════════════════════════════════════════════════

def batched_knn_edge_index(
    knn_idx: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    """将 KNN 索引转换为 PyG 格式的 batched edge_index。

    对 batch 内每个样本, 为每点及其 k 个邻居创建有向边 (i→j 表示 j 是 i 的邻居),
    并通过 batch 偏移拼接为全局 edge_index。由于 KNN 结果中每点的最近邻通常包含
    自身 (距离=0), 所得 edge_index 天然含有自环。

    Args:
        knn_idx: (B, N, k) — KNN 索引, 每个元素是 [0, N) 范围内的邻居编号。
        batch_size: 批次大小 B。
        num_nodes: 每样本点数 N。

    Returns:
        edge_index: (2, B*N*k), int64 — PyG 格式的全局边索引。
            row = 中心节点 (i), col = 邻居节点 (j), 均已加上 batch 偏移。

    Shape 推导:
        输入 (B, N, k) → 每样本 N*k 条边 → 全局 B*N*k 条边 → (2, B*N*k)
    """
    device = knn_idx.device
    # batch 偏移: 第 b 个样本的全局节点编号从 b*N 开始
    # (B, 1, 1) 广播到 (B, N, k)
    offset = torch.arange(batch_size, device=device).view(-1, 1, 1) * num_nodes

    # 中心节点 (每点重复 k 次) → 全局编号
    # (B, N, 1) → (B, N, k) → (B*N*k,)
    row = (torch.arange(num_nodes, device=device).view(1, -1, 1) + offset)
    row = row.expand_as(knn_idx).reshape(-1)

    # 邻居节点 → 全局编号
    col = (knn_idx + offset).reshape(-1)

    # PyG 格式: row = source (中心), col = target (邻居)
    return torch.stack([row, col], dim=0)                   # (2, B*N*k)


# ══════════════════════════════════════════════════
# 加权下采样 (B, C, N) 布局 — 保留自原版
# ══════════════════════════════════════════════════

def weighted_downsample(
    p: torch.Tensor,
    f: torch.Tensor,
    target_n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按特征 L2 范数做无放回加权采样。

    Args:
        p: (B, 4, N)。
        f: (B, C, N)。
        target_n: 目标点数。

    Returns:
        (p_down, f_down): (B, 4, target_n), (B, C, target_n)。
    """
    B, C, N = f.shape
    if target_n >= N:
        return p, f

    scores = f.norm(p=2, dim=1).clamp(min=1e-8)              # (B, N)
    probs = scores / scores.sum(dim=1, keepdim=True)
    idx = torch.multinomial(probs, target_n, replacement=False)  # (B, target_n)

    # gather 沿 N 维采样
    idx_f = idx.unsqueeze(1).expand(-1, C, -1)                # (B, C, target_n)
    idx_p = idx.unsqueeze(1).expand(-1, 4, -1)                # (B, 4, target_n)
    return torch.gather(p, 2, idx_p), torch.gather(f, 2, idx_f)


# ══════════════════════════════════════════════════
# Batched SAGEConv 包装层
# ══════════════════════════════════════════════════

class BatchedSAGEConv(nn.Module):
    """基于 PyG SAGEConv 的批量图卷积包装。

    将 (B, C, N) 格式的点云特征展平为 (B*N, C), 在预计算的 batched edge_index 上
    执行一次 GraphSAGE 消息传递, 再恢复为 (B, C_out, N)。

    SAGEConv 核心公式 (Hamilton et al., NeurIPS 2017, mean 聚合):
        h_i' = W_1 · h_i + W_2 · mean_{j ∈ N(i)} h_j

    相比 DGCNN 的 Conv2d EdgeConv:
        h_i' = MLP( max_{j ∈ N(i)} [h_j - h_i, h_i] )

    关键差异:
        - SAGEConv 是归纳式 (inductive), 可处理未见过的图拓扑;
          DGCNN EdgeConv 依赖 [h_j - h_i, h_i] 拼接, 本质是 edge-level MLP
        - SAGEConv 用 mean 聚合替代 max pool, 保留更多邻域统计信息
        - SAGEConv 有中心-邻居分离权重 (W_1 vs W_2), 表达能力更强

    Args:
        in_channels: 输入特征维度。
        out_channels: 输出特征维度。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """批量 SAGEConv 前向。

        Args:
            x: (B, C_in, N) — channel-first 格式。
            edge_index: (2, E_total) — batched PyG 边索引, E_total = B*N*k。

        Returns:
            (B, C_out, N) — 图卷积输出。
        """
        B, C_in, N = x.shape

        # (B, C_in, N) → (B, N, C_in) → (B*N, C_in): PyG 要求 (num_nodes, features)
        x_flat = x.permute(0, 2, 1).reshape(B * N, C_in)

        # SAGEConv 消息传递: 一次调用处理整个 batched graph
        out = self.conv(x_flat, edge_index)                    # (B*N, C_out)

        # (B*N, C_out) → (B, N, C_out) → (B, C_out, N): 恢复 channel-first
        return out.view(B, N, -1).permute(0, 2, 1).contiguous()


# ══════════════════════════════════════════════════
# Graph Residual Block — PyG 图卷积版
# ══════════════════════════════════════════════════

class GraphResidualBlockGCN(nn.Module):
    """图残差模块 — PyG SAGEConv 版 (真消息传递 + Q/K/V 注意力 + 坐标门控)。

    与原版 GraphResidualBlock 的区别:
        原版 GCN_f / GCN_p 使用 Conv2d 对 [x_j - x_i, x_i] 做 edge MLP;
        本版使用 SAGEConv 做真正的邻居聚合消息传递:
            GCN_f: SAGEConv on 特征空间动态 KNN 图 → f_gcn
            GCN_p: SAGEConv on 坐标空间静态 KNN 图 → p_gcn
        之后, f_gcn / p_gcn 分别作为 V / K 的来源, 进入 Q/K/V 注意力。

    数据流 (全程 (B, C, N) 布局, 无下采样, N 不变):
        f(B,C_in,N) + p(B,4,N) + p_edge_index(2,E) [外部预计算缓存]
            ↓
        Dynamic KNN from f → idx(B,N,k)    ← 仅特征 KNN 每层重算
        → f_edge_index(2,E)
            ↓
        ┌ GCN_f: BatchedSAGEConv(f) + BN1d + LReLU → f_gcn(B,C_out,N)
        └ GCN_p: BatchedSAGEConv(p) + BN1d + LReLU → p_gcn(B,C_out,N)
            ↓
        SE gate: f_gap = GAP(f_gcn) → MLP → sigmoid → se_weight
        f_gcn_gated = f_gcn * se_weight
        fused = Conv1d(cat(f_gcn_gated, p_gcn))
            ↓
        gate = sigmoid(coord_gate(p)), coord_info = coord_res(p)
        out = gate * fused + (1-gate) * coord_info
            ↓
        pos_feat = coord_encoder(p)  ← 2 层 MLP, 位置编码
        f_out = act(out + pos_feat)   ← 坐标编码残差注入
            ↓
        f_out(B,C_out,N), p 原样传出

    设计说明:
        - p 全程不变 → p_edge_index 在 Net 层预计算一次, 4 个 Block 共享
        - SAGEConv 的 mean 聚合天然保留邻域分布信息, 适合 SPAD 点云的噪声建模
        - SE channel gate 替代全局注意力, 消除 1024×1024 过拟合风险
        - **coord_encoder 残差注入**: 2 层 MLP 对 4D 坐标提取位置特征,
          通过残差加法注入到最终特征, 使每个点特征显式包含其空间位置编码,
          改善深度回归 (避免全局池化后位置信息丢失)

    Args:
        in_channels: C_in。
        out_channels: C_out。
        k: 近邻数。
        downsample: 是否启用 N→N/2 下采样 (当前配置为 False)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 20,
        downsample: bool = True,
    ):
        super().__init__()
        self.k = k
        self.downsample = downsample

        # ── GCN_f: 特征流 SAGEConv ──
        # 对特征空间动态 KNN 图做消息传递 (DGCNN "动态图" 范式, 卷积核换为真 GNN)
        self.gcn_f = BatchedSAGEConv(in_channels, out_channels)
        self.bn_f = nn.BatchNorm1d(out_channels)

        # ── GCN_p: 位置流 SAGEConv ──
        # 对坐标空间静态 KNN 图做消息传递 (输入为 4D 原始坐标 x,y,z,i)
        # SAGEConv 从固定的坐标拓扑中学习几何特征, 替代原版 Conv2d EdgeConv
        self.gcn_p = BatchedSAGEConv(4, out_channels)
        self.bn_p = nn.BatchNorm1d(out_channels)

        # ── SE-style channel gate (替代原版全局注意力) ──
        # SAGEConv 已将邻域聚合为单向量, 全局 (B,N,N) 注意力既嘈杂又过拟合;
        # SE 模块通过通道注意力自适应加权 f_gcn, 计算量可忽略。
        # 结构: GAP → Linear(C→C/4) → ReLU → Linear(C/4→C) → Sigmoid
        se_ratio = 4
        self.se_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels // se_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // se_ratio, out_channels, bias=False),
            nn.Sigmoid(),
        )

        # ── 双流融合 Conv1d ──
        # 将 SE 加权后的 f_gcn 与 p_gcn 拼接后, 用 Conv1d 融合回 out_channels。
        # 替代原版的 Q/K/V 投影 + 注意力 + out_conv, 参数量大幅缩减。
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # 坐标门控 + 坐标残差 (4D: x,y,z,i)
        self.coord_gate = nn.Sequential(
            nn.Conv1d(4, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.coord_res = nn.Sequential(
            nn.Conv1d(4, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # 坐标编码器 (2 层 MLP): 对原始 4D 坐标提取更丰富的位置特征
        # 通过残差加法注入到最终特征中, 使每个点特征显式包含其空间位置编码
        # 全局池化后, 位置统计量仍被保留, 改善深度回归 (避免 "盲猜" 中心点)
        self.coord_encoder = nn.Sequential(
            nn.Conv1d(4, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        p: torch.Tensor,
        f: torch.Tensor,
        p_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block 前向。

        Args:
            p: (B, 4, N) 原始坐标+intensity。
            f: (B, C_in, N) 当前层特征。
            p_edge_index: (2, B*N*k) 坐标图的 batched PyG 边索引
                (Net 层预计算, 4 个 Block 共享)。

        Returns:
            (p, f_out): p 原样传出 (当前配置不下采样),
                f_out 为升维后特征 (B, C_out, N)。
        """
        B, C_in, N = f.shape
        k = min(self.k, N - 1)

        # ── 特征空间 KNN (每层重算, 语义驱动动态图) ──
        knn_f = knn_gpu(f, k)                                       # (B, N, k)
        f_edge_index = batched_knn_edge_index(knn_f, B, N)          # (2, B*N*k)

        # ── GCN_f: 特征流 SAGEConv → f_gcn ──
        # SAGEConv: h_i' = W_1·f_i + W_2·mean_{j∈N(i)} f_j
        f_gcn = self.gcn_f(f, f_edge_index)                         # (B, C_out, N)
        f_gcn = self.act(self.bn_f(f_gcn))                          # (B, C_out, N)

        # ── GCN_p: 位置流 SAGEConv → p_gcn (复用预计算的坐标边) ──
        # 输入为 4D 原始坐标, SAGEConv 从固定拓扑中学习几何结构特征
        p_gcn = self.gcn_p(p, p_edge_index)                         # (B, C_out, N)
        p_gcn = self.act(self.bn_p(p_gcn))                          # (B, C_out, N)

        # ── SE-style channel gate (替代全局注意力) ──
        # 对 f_gcn 做全局平均池化 → MLP → sigmoid → 通道加权
        # (B, C_out, N) → (B, C_out) → (B, C_out) → (B, C_out, 1) → broadcast
        f_gap = f_gcn.mean(dim=-1)                                  # (B, C_out)
        se_weight = self.se_gate(f_gap).unsqueeze(-1)               # (B, C_out, 1)
        f_gcn_gated = f_gcn * se_weight                             # (B, C_out, N)

        # ── 双流融合 ──
        # cat([f_gcn_gated, p_gcn]) → Conv1d → fused
        fused = self.fuse_conv(torch.cat([f_gcn_gated, p_gcn], dim=1))  # (B, C_out, N)

        # ── 坐标门控跳跃连接 ──
        gate = torch.sigmoid(self.coord_gate(p))                    # (B, C_out, N)
        coord_info = self.coord_res(p)                              # (B, C_out, N)
        out = gate * fused + (1.0 - gate) * coord_info

        # ── 坐标编码残差注入 (改善深度回归) ──
        # 2 层 MLP 对 4D 坐标提取丰富位置特征, 通过残差加法注入到最终特征
        # 使每个点的 f_out[:, :, i] 显式包含 p[:, :, i] 的空间编码
        # 全局池化后位置统计量仍被保留, 避免深度回归 "盲猜" 中心点
        pos_feat = self.coord_encoder(p)                            # (B, C_out, N)
        f_out = self.act(out + pos_feat)

        # ── 层间下采样 ──
        if self.downsample:
            p, f_out = weighted_downsample(p, f_out, N // 2)
        return p, f_out


# ══════════════════════════════════════════════════
# 外层多任务网络 — PyG 图卷积版
# ══════════════════════════════════════════════════

class GraphResidualMultiTaskNetGCN(nn.Module):
    """图残差多任务网络 — PyG SAGEConv 版。

    与原版 GraphResidualMultiTaskNet (v7) 的区别:
        1. Block 内 GCN_f / GCN_p 使用 PyG SAGEConv (真消息传递)
           替代 DGCNN Conv2d EdgeConv
        2. p_graph 缓存从 (B, 8, N, k) 边特征变为 (2, B*N*k) PyG 边索引
        3. 注意力从 k 邻居 softmax 变为全局 N 点 softmax (GCN 已聚合邻域)
        4. Block 内坐标编码残差注入改善深度回归

    全程 (B, C, N) 布局, Conv+BN+LeakyReLU。
    通道: 4→32→64→64→128→256→512, 点数: 全程 1024 不下采样。

    Args:
        num_classes: 分类数。
        k: 近邻数 (默认 20, 对齐 DGCNN)。
        use_checkpoint: 梯度检查点 (默认 True)。
        dropout: 头部 Dropout。
        box_dim: bbox 维度。
    """

    def __init__(
        self,
        num_classes: int = 26,
        k: int = 20,
        use_checkpoint: bool = True,
        dropout: float = 0.3,
        box_dim: int = 3,
    ):
        super().__init__()
        self.k = k
        self.box_dim = box_dim
        self.use_checkpoint = use_checkpoint

        # Stem: (B, 4, N) → (B, 32, N)
        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )

        # 无下采样: 全部 1024 点贯穿 4 层, p 不变只升维 f
        block_cfg = dict(k=k, downsample=False)
        self.block1 = GraphResidualBlockGCN(32, 64, **block_cfg)
        self.block2 = GraphResidualBlockGCN(64, 64, **block_cfg)
        self.block3 = GraphResidualBlockGCN(64, 128, **block_cfg)
        self.block4 = GraphResidualBlockGCN(128, 256, **block_cfg)

        # 多尺度拼接: cat(b1, b2, b3, b4) → Conv1d 聚合
        cat_dim = 64 + 64 + 128 + 256  # 512
        self.agg_conv = nn.Sequential(
            nn.Conv1d(cat_dim, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        pooled_dim = 1024  # 512 * 2 (max + avg)

        # 统一分类头: 3 层 MLP (1024 → 256 → 128 → num_classes)
        self.cls_head = build_standard_cls_head(pooled_dim, num_classes, dropout=dropout)

        # 统一中心点回归头: 3 层 MLP (1024 → 256 → 128 → box_dim)
        # 直接回归, 与 baseline 一致, 确保 backbone 为唯一变量。
        self.box_head = build_standard_box_head(pooled_dim, box_dim=box_dim, dropout=dropout)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (B, N, 4) — x, y, z, intensity。

        Returns:
            dict: 'logits' (B, C), 'box_pred' (B, box_dim)。
        """
        # (B, N, 4) → (B, 4, N) 全程 channel-first
        p = points.transpose(1, 2).contiguous()                # (B, 4, N)
        B, _, N = p.shape
        f = self.stem(p)                                        # (B, 32, N)

        # ── 坐标 KNN + edge_index 预计算 (p 全程不变, 4 个 Block 共享) ──
        k = min(self.k, N - 1)
        knn_p = knn_gpu(p, k)                                   # (B, N, k)
        p_edge_index = batched_knn_edge_index(knn_p, B, N)     # (2, B*N*k)

        # 4 层无下采样, 全程 N=1024; 仅特征 KNN 每层重算
        use_ckpt = self.use_checkpoint and self.training and _HAS_CKPT

        def _run_block(block, _p, _f, _pe):
            return block(_p, _f, _pe)

        if use_ckpt:
            p, f1 = _ckpt(_run_block, self.block1, p, f, p_edge_index, use_reentrant=False)
            p, f2 = _ckpt(_run_block, self.block2, p, f1, p_edge_index, use_reentrant=False)
            p, f3 = _ckpt(_run_block, self.block3, p, f2, p_edge_index, use_reentrant=False)
            p, f4 = _ckpt(_run_block, self.block4, p, f3, p_edge_index, use_reentrant=False)
        else:
            p, f1 = self.block1(p, f, p_edge_index)             # 32→64
            p, f2 = self.block2(p, f1, p_edge_index)            # 64→64
            p, f3 = self.block3(p, f2, p_edge_index)            # 64→128
            p, f4 = self.block4(p, f3, p_edge_index)            # 128→256

        # 多尺度拼接 + 聚合
        f = self.agg_conv(torch.cat([f1, f2, f3, f4], dim=1))  # (B, 512, N)

        # 全局池化: max + avg → (B, 1024)
        f_max = f.max(dim=-1)[0]
        f_avg = f.mean(dim=-1)
        f_pooled = torch.cat([f_max, f_avg], dim=1)              # (B, 1024)

        logits = self.cls_head(f_pooled)

        # Box head: 直接回归 (与 baseline 一致)
        # backbone 内 Block 的 coord_encoder 已将位置信息注入到每个点特征中,
        # 全局池化后位置统计量仍被保留, 深度回归不再 "盲猜" 中心点
        box_preds = self.box_head(f_pooled)                       # (B, 3)

        return {"logits": logits, "box_pred": box_preds}
```

> **代码位置**: [model/graph_res_GCN.py](graph_res_GCN.py)