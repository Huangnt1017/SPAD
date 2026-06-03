# SPAD SNN Imaging Project

## 项目路径
- 模型兼容入口: D:\PYproject\SPAD\SNN_based_method\SNN.py
- 模型实际实现 (activation_based): D:\PYproject\SPAD\SNN_based_method\SNN_new.py
- 损失函数: D:\PYproject\SPAD\SNN_based_method\loss.py
- 统一配置: D:\PYproject\SPAD\SNN_based_method\SNN_config.py
- 训练/测试脚本: D:\PYproject\SPAD\SNN_based_method\scripts\
- 设计文档: D:\PYproject\SPAD\SNN_based_method\reademe.md
- 基线模型: D:\PYproject\SPAD\baseline\SPT.py
- 数据示例: D:\PYproject\SPADdata\0825\frame\

## 环境
- PyTorch: 使用 torchnew 环境
- SpikingJelly: 使用环境中安装的官方 `spikingjelly.activation_based`
- SNN.py 仅保留旧导入路径兼容, 实际转发到 SNN_new.py
- SNN_new.py 使用 activation_based API, cupy 后端自动探测

## 对齐参数 (与 read_spad-master 对齐)
- `T_max = time_threshold = 128` (最大有效 ToF bin)
- `B = batch_size = 8`
- `chunk_size = 64` (每个 chunk 的帧数)
- `pages_per_group = 512` (64×8, 每个训练样本的 raw page 数)
- `depth_range = 128.0` (与 T_max 对齐)
- `C = 32` (工作通道数)

## 当前状态
- SNN.py: 兼容入口, 不包含本地 spikingjelly1/clock_driven 实现
- SNN_new.py: 正弦编码 + 等宽 SpikeBlock + Chunked 处理 + 置信度门控精修头, 已通过测试
  - SpatialRefineHead: 3 通道输入 (归一化 depth + intensity + confidence),
    残差按置信度缩放, 输出 clamp 到有效范围
  - 新增 encoding_mode="lut" 支持可学习 LUT 编码 (embed_dim 可调, 3 种初始化)
  - LUT 模式 forward 返回 lut_smooth / lut_norm 正则 loss 供训练使用
  - forward 输出字典包含 confidence: weight_sum / (weight_sum + 1.0)
- activation_based 版本神经元 step_mode='m'
  - 自动探测 cupy 后端, 不可用时回退到 torch
- loss.py: L_GT + L_SSIM + L_var + L_sparse + L_smooth + LUT 正则, 已通过测试
  - SSIMLoss: 7×7 高斯窗口, depth_range=128, intensity_range=1.0
  - ImageMetrics: MAE / RMSE / SSIM / PSNR, 归一化到 [0,1] 后计算, 不参与梯度
- SNN_config.py: SNNConfig dataclass 统一管理所有可调参数, 已通过测试
  - build_model() / build_loss() / build_metrics() / build_dataloaders() 工厂方法
  - LUT 相关参数 (embed_dim / lut_init / w_lut_*) 仅 encoding_mode="lut" 时生效
  - save/load JSON 序列化; 预置: SINUSOIDAL_DEFAULT / LUT_RBF_16 / LUT_SIN_16 / LUT_RBF_32
- scripts/: 训练 (train.py) / 测试 (test.py) / 单样本推理 (test1.py) / 数据加载 (data.py) / 运行时工具 (runtime.py)
- visualize_encoding.py: 编码可视化工具 (频率响应 / 多帧聚合 / 雾 vs 目标区分度)
  - 支持 t_max 参数化 (不再硬编码 150)
  - 默认使用非均匀频率预设 B (n_freq=8 时)

## 关键设计决策
- 编码: 正弦位置编码 (17 通道, freqs=[1,2,4,6,8,12,16,24]), 不用标量 tof/128 (避免幅度偏差)
  - 非均匀频率方案 adj=0.689, 比等差方案 (adj=0.299) 精细分辨率高 2.3×
  - 消融备选: LUT 编码 (embed_dim=16, rbf 初始化), 参数仅多 2416 个
- 结构: 等宽 (T/C/64×64 全程不变), 不用时间 U-Net (P 帧独立采样, PLIF 自然积累)
- 深度输出: Gated Moment (depth 来自原始 tof 加权) + 置信度门控空间精修头
  - 置信度 = weight_sum / (weight_sum + 1.0), 反映累积光子证据充分程度
  - 精修残差按置信度缩放: 低置信度区域保留粗估, 高置信度区域允许大幅修正
- 显存: fp16+checkpoint, chunk=64, B=8 → ~6.5GB (24GB GPU)
