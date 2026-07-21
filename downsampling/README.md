# SPAD XYZI 点云降采样模块

本目录提供三个彼此独立的降采样器，只依赖 PyTorch，不修改项目现有训练、测试或模型入口。

## 统一输入输出

所有采样器接收：

```text
points: (B, N, 4)
通道:   x, y, z, intensity
```

并返回 `DownsampleOutput`：

```text
output.points:  (B, K, 4)  从原始输入提取的硬点子集
output.indices: (B, K)     每个样本内互异的原始点索引
output.scores:  (B, N)     可选逐点分数
```

学习型方法还会返回：

```text
SampleNet:
    output.projected_points: (B, K, 4)  可微软投影，训练任务网络时使用
    output.generated_points: (B, K, 3)  归一化三维查询点
    output.aux_losses:                   简化损失和温度损失

APES-Local:
    output.features: (B, C, K)          可微注意力聚合特征
```

采样器内部对 `xyz` 做逐样本极差归一化，对非负光子计数执行 `log1p` 后再归一化。最终硬输出仍取自未修改的原始输入。

## 方法一：I-WRS

```python
import torch

from downsampling import IntensityWeightedRandomSampler

points = torch.rand(2, 4096, 4, device="cuda")
sampler = IntensityWeightedRandomSampler(
    num_samples=1024,
    gamma=0.5,
).to(points.device)

output = sampler(points)
sampled_points = output.points
sampled_indices = output.indices
```

如需完全复现随机结果，传入与点云同设备的生成器：

```python
generator = torch.Generator(device=points.device).manual_seed(42)
output = sampler(points, generator=generator)
```

## 方法二：SampleNet-XYZI

```python
from downsampling import SampleNetXYZI

sampler = SampleNetXYZI(
    num_samples=1024,
    projection_neighbors=8,
    distance_chunk_size=256,
).to(points.device)

sampler.train()
output = sampler(points)

# 训练时，固定任务网络使用可微软投影点。
predictions = task_model(output.projected_points)
task_loss = task_criterion(predictions, targets)
sampler_loss = sampler.sampler_loss(
    output,
    simplification_weight=1.0,
    projection_weight=1.0,
)
loss = task_loss + sampler_loss
loss.backward()
```

评价和导出时使用硬子集：

```python
sampler.eval()
with torch.no_grad():
    output = sampler(points)
    sampled_points = output.points
    sampled_indices = output.indices
```

本地适配的关键约定：编码器读取四维 `xyzi`；解码器只生成三维查询点；邻域距离只在归一化 `xyz` 中计算；同一软投影权重聚合完整的原始 `xyzi`。

## 强度感知任务无关 SampleNet-XYZI 训练

`scripts/train_task_agnostic_samplenet_xyzi.py` 提供独立于分类、检测和
A--Z 标签的训练流程。脚本复用本目录已有的 `SampleNetXYZI`，不连接任何
下游任务网络。正式数据目录默认为：

```text
D:\PYproject\SPADdata\20250430\2025-04-30-pc
```

扫描器只读取递归目录中严格匹配以下规则的 TXT：

```text
yyyy-mm-dd_hh-mm-ss_Delay-0_Width-200-i-(i+2).txt
```

日期、时间必须有效，`i >= 1`，末尾窗口号必须等于 `i + 2`。父目录名称只
用于保持导出相对路径，不作为标签。`A.txt`、JSON、NPY、带 `_hmc` 后缀或
窗口宽度不为 3 的文件均不会进入该流程。

### 输入、模型输出和损失

默认数据契约：

```text
正式 TXT:             (N, 4), N >= 8192, x/y/z/intensity
模型候选输入:         (B, 8192, 4)
generated_points:     (B, 1024, 3), 可微归一化 XYZ 查询点
projected_points:     (B, 1024, 4), 可微软投影 XYZI
output.points:        (B, 1024, 4), 原始候选输入中的真实 XYZI 行
output.indices:       (B, 1024), 每个样本内唯一
```

每个正式文件按“相对路径 + seed”稳定地无放回选择 8192 个候选行。文件不足
8192 行时直接报错，不使用重复补点。训练损失为：

```text
total = geometry_weight * 现有 simplification
      + intensity_weight * log1p 强度加权覆盖
      + projection_weight * 现有软投影温度平方
```

其中现有几何简化/覆盖损失由 `generated_points` 产生；强度项计算输入点到
`projected_points` 的归一化 XYZ 最近距离，并以 `log1p(max(intensity, 0))`
加权。全零强度样本退化为均匀覆盖。训练不以硬 `output.points` 计算损失，
因此生成点、软投影点和温度参数均保持梯度。

### 先运行短检查

以下命令只读取 1 个正式文件，完成前向、反向、有限非零梯度、1024 个唯一
索引、checkpoint 保存/恢复和单文件 TXT 导出：

```powershell
python scripts/train_task_agnostic_samplenet_xyzi.py `
  --mode sanity `
  --device cpu `
  --max-files 1 `
  --run-dir outputs/samplenet_xyzi_task_agnostic/sanity_manual
```

直接在 IDE 中无参数运行也只执行相同的单文件短检查：

```powershell
python scripts/train_task_agnostic_samplenet_xyzi.py
```

无参数入口不会启动全量长期训练。sanity 输出包括：

```text
<run-dir>/sanity_config.json
<run-dir>/checkpoints/sanity_last.pth
<run-dir>/export/<原相对目录>/<正式文件名>.txt
```

### 启动训练

全量训练必须显式指定 `--mode train`：

```powershell
python scripts/train_task_agnostic_samplenet_xyzi.py `
  --mode train `
  --data-root "D:\PYproject\SPADdata\20250430\2025-04-30-pc" `
  --candidate-points 8192 `
  --num-samples 1024 `
  --projection-neighbors 8 `
  --batch-size 8 `
  --epochs 100 `
  --device cuda
```

默认训练集/验证集比例为 0.9/0.1，按路径稳定哈希划分，不读取类别。用于短
实验时可加 `--max-files 2` 或更大的有限值。每个训练 run 默认保存到：

```text
outputs/samplenet_xyzi_task_agnostic/<yyyyMMdd_HHmmss>/
├── config.json
├── train.log
└── checkpoints/
    ├── best.pth
    └── last.pth
```

`best.pth` 按验证集总损失最小保存；`last.pth` 每个完整 epoch 覆盖。两者均
包含模型、优化器、scheduler、epoch、最佳验证损失、模型构造参数、候选点
数、损失配置和完整运行配置。

常用训练参数：

```text
--candidate-points 8192       每文件模型候选点数
--num-samples 1024            输出点数
--projection-neighbors 8      每个生成点的软投影邻居数
--batch-size 8                初始 batch size
--feature-dim 256             PointNet 编码器最终通道数
--hidden-dim 512              查询点解码器隐层宽度
--distance-chunk-size 256     模型内部生成点距离分块
--intensity-chunk-size 512    强度覆盖损失输入点距离分块
--geometry-weight 1.0         几何损失权重
--intensity-weight 1.0        log1p 强度覆盖权重
--projection-weight 1.0       温度损失权重
--learning-rate 1e-3          AdamW 初始学习率
--min-learning-rate 1e-5      余弦退火最低学习率
--weight-decay 1e-4           AdamW 权重衰减
--val-ratio 0.1               无标签验证集比例
--num-workers 0               DataLoader worker 数
--device auto                 auto/cpu/cuda
--max-files 0                 0=扫描全部；正数=只取排序后前若干文件
```

### 恢复训练

恢复时使用 checkpoint 元数据，不从日志推断 epoch。模型构造参数和
`candidate_points` 必须与 checkpoint 一致；优化器和 scheduler 状态会一并
恢复。`--epochs` 表示恢复后的目标总 epoch 数：

```powershell
python scripts/train_task_agnostic_samplenet_xyzi.py `
  --mode train `
  --resume outputs/samplenet_xyzi_task_agnostic/<run>/checkpoints/last.pth `
  --epochs 150 `
  --batch-size 8 `
  --device cuda
```

未显式设置 `--run-dir` 时，恢复结果继续写入 checkpoint 所属 run，配置另存
为 `resume_config_<timestamp>.json`。

### 冻结模型并批量导出

导出模式从 checkpoint 恢复模型构造参数和候选点数，冻结所有参数，递归
扫描同一正式文件规则，并保持输入根目录下的相对目录结构和正式文件名：

```powershell
python scripts/train_task_agnostic_samplenet_xyzi.py `
  --mode export `
  --checkpoint outputs/samplenet_xyzi_task_agnostic/<run>/checkpoints/best.pth `
  --data-root "D:\PYproject\SPADdata\20250430\2025-04-30-pc" `
  --export-dir "D:\PYproject\SPADdata\20250430\samplenet_xyzi_task_agnostic" `
  --batch-size 8 `
  --device cuda
```

导出前逐 batch 校验：`output.indices` 唯一、映射回原 TXT 的行索引唯一、
`output.points == input[output.indices]`。落盘只使用 `output.points`，只生成
逗号分隔 TXT，不导入绘图库、不保存图片。默认拒绝覆盖；可选
`--overwrite` 或 `--skip-existing`，两者不能同时启用。

## 方法三：APES-Local-XYZI

```python
from downsampling import APESLocalXYZI

sampler = APESLocalXYZI(
    num_samples=1024,
    num_neighbors=32,
    embedding_dim=128,
    knn_chunk_size=1024,
).to(points.device)

sampler.train()
output = sampler(points)

# Top-K 索引不可微；训练任务头必须使用可微聚合特征。
predictions = task_head(output.features)
loss = task_criterion(predictions, targets)
loss.backward()

# 点云导出始终使用原始点子集。
sampled_points = output.points
sampled_indices = output.indices
```

APES-Local 使用归一化 `xyz` 构造 KNN，以 `xyzi` 编码特征。KNN 查询采用分块距离计算，降低一次保存完整距离矩阵的峰值内存，但总体计算量仍随输入点数近似二次增长。

## 工厂接口

```python
from downsampling import available_downsamplers, build_downsampler

print(available_downsamplers())

sampler = build_downsampler(
    "i_wrs",               # 也可用 samplenet_xyzi / apes_local_xyzi
    num_samples=1024,
    gamma=0.5,
)
```

## 保存与恢复

I-WRS 没有可学习参数。两个学习型采样器使用标准 PyTorch 状态字典：

```python
torch.save(sampler.state_dict(), "sampler.pth")
sampler.load_state_dict(torch.load("sampler.pth", map_location="cpu"))
```

保存 checkpoint 时还应记录构造参数，例如 `num_samples`、邻居数、特征维度和 SampleNet 温度设置。

## 验证

在项目根目录运行：

```powershell
python -m compileall -q downsampling
python -m unittest discover -s downsampling\tests -v
```

测试覆盖：输入校验、输出形状、索引唯一性、原始点一致性、随机种子复现、学习型反向传播、工厂构建和状态字典恢复。

## A--Z I-WRS 批处理

`scripts/run_iwrs_az_downsampling.py` 默认读取：

```text
D:\PYproject\SPADdata\20250430\A.txt
...
D:\PYproject\SPADdata\20250430\Z.txt
```

先执行不写文件的参数检查：

```powershell
python scripts/run_iwrs_az_downsampling.py --dry-run
```

再生成 26 份 1024 点结果：

```powershell
python scripts/run_iwrs_az_downsampling.py --num-samples 1024 --gamma 0.5 --seed 42
```

默认输出目录为：

```text
D:\PYproject\SPADdata\20250430\downsampling_results\20250430_i_wrs_k1024_gamma0p5_seed42
```

目录内只保存逗号分隔的 `A.txt`--`Z.txt`，每行均来自对应输入点云。脚本默认不导入可视化模块、不弹窗、不保存图片。如需直接对已经保存的结果手动弹窗检查，可运行：

```powershell
python -c "from data_read.raw2pointcloud import read_pc, plot_pc; p=read_pc(r'D:\PYproject\SPADdata\20250430\downsampling_results\20250430_i_wrs_k1024_gamma0p5_seed42\A.txt'); plot_pc(p, 'ds')"
```

这会复用项目实际存在的 `data_read/raw2pointcloud.py` 中 `read_pc` 和 `plot_pc(..., mode="ds")`，只调用 `plt.show()`，不会重新降采样或保存图片。批处理脚本本身也保留 `--preview-label A` 入口，适合在首次运行且输出目录尚不存在时处理完成后立即预览。

## 方法来源与复现状态

- SampleNet: Lang et al., *SampleNet: Differentiable Point Cloud Sampling*,
  CVPR 2020，官方 GitHub：`https://github.com/itailang/SampleNet`，本次对照
  commit：`3d20c7a62f6788cc56b68d5367ff25a8a2c13fad`。
- APES: Wu et al., *Attention-Based Point Cloud Edge Sampling*, CVPR 2023，
  官方 GitHub：`https://github.com/JunweiZheng93/APES`，本次对照 commit：
  `988aa892980261d8685dc6734422ed4c0da25a52`。
- I-WRS 的加权无放回键采样依据 Efraimidis and Spirakis,
  *Weighted random sampling with a reservoir*, IPL 2006；未发现作者官方
  GitHub 仓库。

详细逐项审计见 [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md)。结论是：本目录代码
属于面向 SPAD `xyzi` 数据契约的本地适配/重实现，**不是官方 GitHub 的
逐行、完整网络或 checkpoint 兼容复现**。各方法文件顶部已写明论文名称、
GitHub 来源（若存在）、固定审计版本和 BibTeX。
