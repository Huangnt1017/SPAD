# 降采样方法官方源码复现审计

审计日期：2026-07-15

## 总结

`downsampling` 当前三个方法都**不能整体标注为“完全按照官方 GitHub 源码复现”**：

- `IntensityWeightedRandomSampler`：论文核心键排序等价，但论文没有作者官方
  GitHub；强度权重和批量实现是本地设计。
- `SampleNetXYZI`：保留 SampleNet 的方法骨架，但网络结构、温度、投影、
  硬匹配和 XYZI 数据契约均与官方实现不同。
- `APESLocalXYZI`：保留 APES-Local 的注意力标准差打分和 Top-K 核心，但
  Embedding、邻域图、N2PAttention 和输出路径不是官方完整网络。

因此，准确表述应为“论文方法的 SPAD-XYZI 本地适配/重实现”，而不是
“官方源码等价复现”或“官方 checkpoint 兼容复现”。

## 审计基线

| 方法 | 论文 | 官方源码基线 |
|---|---|---|
| I-WRS | *Weighted random sampling with a reservoir* | 未发现作者官方 GitHub 仓库 |
| SampleNet | *SampleNet: Differentiable Point Cloud Sampling* | `itailang/SampleNet`，commit `3d20c7a62f6788cc56b68d5367ff25a8a2c13fad` |
| APES | *Attention-Based Point Cloud Edge Sampling* | `JunweiZheng93/APES`，commit `988aa892980261d8685dc6734422ed4c0da25a52` |

本次 SampleNet 重点对照：

- `registration/src/samplenet.py`
- `registration/src/soft_projection.py`
- `registration/src/sputils.py`

本次 APES 重点对照：

- `apes/models/utils/layers.py` 中的 `Embedding`、`N2PAttention`、
  `LocalDownSample`
- `apes/models/utils/ops.py`
- `apes/models/backbones/apes_cls_backbone.py`

## I-WRS

### 一致部分

- 论文 Algorithm A-Res 使用键 `U ** (1 / w)`；当前实现使用
  `log(U) / w`。由于对数严格单调，两种键的排序等价。
- 从全部键中选择最大 K 个，得到加权无放回样本；`topk` 同时保证索引互异。

### 本地差异

- 论文算法要求正权重，但不规定 SPAD 强度如何变为权重。当前实现使用
  `log1p` 强度归一化，再使用 `(intensity + eps) ** gamma`。
- 论文描述的是 reservoir/streaming 形式；当前实现一次生成整批随机键并
  `topk`，抽样分布等价，但内存与执行方式不等价。
- 未找到作者官方 GitHub，因此不能进行“GitHub 逐行一致性”验证。

**结论：核心抽样公式等价，工程实现不是官方 GitHub 复现。**

## SampleNet-XYZI

| 项目 | 官方 SampleNet | 当前 `SampleNetXYZI` | 结论 |
|---|---|---|---|
| 输入 | `(B, 3, N)` XYZ | `(B, N, 4)` XYZI | 本地适配 |
| 编码器 | `3→64→64→64→128→bottleneck`，逐点 BN/ReLU | `4→64→128→feature_dim` | 不一致 |
| 全局池化 | max pooling | max 与 mean 拼接 | 不一致 |
| 解码器 | 三层 `256 + BN + ReLU` 后输出 `3K` | 两层可配置隐层、LayerNorm 后输出 `3K` | 不一致 |
| 查询点范围 | 线性输出，不强制 `[0, 1]` | `sigmoid` 约束到 `[0, 1]` | 不一致 |
| 软投影温度 | `sigma=max(T**2,min_sigma)`，损失为 `sigma` | 正值 softplus 温度作分母，损失为温度平方 | 参数化不一致 |
| 投影通道 | 在 XYZ 坐标中投影 | 以 XYZ 距离聚合原始 XYZI 四通道 | 本地适配 |
| 简化损失距离 | 官方 Chamfer 算子 | `torch.cdist` 欧氏距离 | 距离尺度不一致 |
| 推理硬匹配 | 最近邻去重后 continued FPS 补齐 | 首次命中去重后按查询覆盖优先级补齐 | 不一致 |
| 训练目标 | 任务损失 + SampleNet 正则 | 另有任务无关强度覆盖损失 | 本地扩展 |

**结论：方法骨架相似，但不是逐层/逐行复现，官方权重不能直接加载。**

## APES-Local-XYZI

| 项目 | 官方 APES-Local | 当前 `APESLocalXYZI` | 结论 |
|---|---|---|---|
| 输入 | XYZ，3 通道 | XYZI，4 通道 | 本地适配 |
| Embedding | 两级 `center_diff` 邻域分组、Conv2d、邻域 max，拼成 128 通道 | 两层逐点 Conv1d | 不一致 |
| 下采样前上下文 | `Embedding → N2PAttention → LocalDownSample` | 直接在本地 embedding 后下采样 | 缺少官方模块 |
| Local KNN | 在当前 128 维特征上分组，官方 `ops.knn` 会包含自身 | 在归一化 XYZ 上分组并显式排除自身 | 不一致 |
| Q/K/V 与打分 | 局部 attention，按邻域维标准差评分，Top-K | 保留同一核心打分 | 核心一致 |
| 特征输出 | 对选中中心的 attention/value 加权结果 | 同类加权结果后额外 BN/LeakyReLU | 部分一致 |
| 点坐标输出 | 官方下采样层返回特征并保存 `idx` | 直接返回原始 XYZI 子集与 `idx` | 本地接口 |

**结论：复现了 APES-Local 的核心选点准则，不是官方完整网络复现。**

## 代码处理原则

本次只补充可追溯引用、固定源码基线和真实复现状态，没有把现有实现直接
改写成官方网络。原因是当前训练脚本与已有 checkpoint 依赖现有参数名和
张量结构；原地重写会破坏 checkpoint 兼容性。若后续需要严格对照实验，
应新增独立的 `SampleNetOfficialXYZIAdapter` / `APESOfficialXYZIAdapter`
实现和独立 checkpoint，而不是覆盖当前类。
