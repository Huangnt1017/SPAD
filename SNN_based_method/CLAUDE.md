# SPAD SNN Imaging Project

## 项目路径
- 模型代码: D:\PYproject\SPAD\SNN_based_method\SNN.py
- 损失函数: D:\PYproject\SPAD\SNN_based_method\loss.py
- 设计文档: D:\PYproject\SPAD\SNN_based_method\reademe.md
- SpikingJelly: D:\PYproject\SPAD\spikingjelly\clock_driven\ (旧版 clock_driven API)
- 神经元构造: D:\PYproject\SPAD\utils\pointnet_utils.py (build_spike_node)
- 基线模型: D:\PYproject\SPAD\baseline\SPT.py
- 数据示例: D:\PYproject\SPADdata\0825\frame\

## 环境
- PyTorch 2.1.2+cu121
- SpikingJelly 旧版 (clock_driven, 非 activation_based)
- build_spike_node 使用 MultiStep*Node + cupy backend

## 当前状态
- SNN.py: 正弦编码 + 等宽SpikeBlock + Chunked处理 + 空间精修头, 已通过测试
  - 新增 encoding_mode="lut" 支持可学习 LUT 编码 (embed_dim 可调, 3种初始化: sinusoidal/rbf/random)
  - LUT 模式 forward 返回 lut_smooth / lut_norm 正则 loss 供训练使用
- loss.py: L_GT + L_SSIM + L_var + L_sparse + L_smooth + LUT正则, 已通过测试
  - SSIMLoss: 7×7 高斯窗口, depth_range=150, intensity_range=1.0
  - ImageMetrics: MAE / RMSE / SSIM / PSNR, 归一化到 [0,1] 后计算, 不参与梯度
- SNN_config.py: SNNConfig dataclass 统一管理所有可调参数, 已通过测试
  - build_model() / build_loss() / build_metrics() 工厂方法
  - LUT 相关参数 (embed_dim / lut_init / w_lut_*) 仅 encoding_mode="lut" 时生效
  - save/load JSON 序列化; 预置: SINUSOIDAL_DEFAULT / LUT_RBF_16 / LUT_SIN_16 / LUT_RBF_32
- visualize_encoding.py: 编码可视化工具 (频率响应 / 多帧聚合 / 雾vs目标区分度)
- 待做: 数据加载器、训练脚本、真实数据验证

## 关键设计决策
- 编码: 正弦位置编码 (17通道, freqs=[1,2,4,6,8,12,16,24]), 不用标量tof/150 (避免幅度偏差)
  - 非均匀频率方案 adj=0.689, 比等差方案 (adj=0.299) 精细分辨率高 2.3×
  - 消融备选: LUT 编码 (embed_dim=16, rbf初始化), 参数仅多 2416 个
- 结构: 等宽 (T/C/64×64全程不变), 不用时间U-Net (P帧独立采样, PLIF自然积累)
- 深度输出: Gated Moment (depth来自原始tof加权) + 空间精修残差头
- 显存: fp16+checkpoint, chunk=128, B=8 → ~6.5GB (24GB GPU)
