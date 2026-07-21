# SPAD 网络结构可编辑汇总 PPT

最终生成与渲染日期：2026-07-20

## 唯一 PPT 文件

- `SPAD_model_architectures_editable_combined.pptx`：12 页可编辑汇总版。

不再生成或维护多个单页 PPT。所有模块、文字、箭头和三次 Bézier 曲线均为 PowerPoint 原生可编辑对象，逐页预览中不包含嵌入到 PPT 内的截图。

## 页面顺序

1. 原始 `GraphResidual`：EdgeConv + 局部 Q/K/V 注意力；
2. 正式 `GraphResidual-GCN`：双图 GraphSAGE + SE + 双残差；
3. Controls 总览：B0–B7 硬结构消融 + B8 参数匹配算子对照；
4. B1：关闭坐标图分支；
5. B2：关闭 SE；
6. B3：关闭坐标残差；
7. B4：max 改为 mean 聚合；
8. B5：KNN 包含自身点；
9. B6：关闭 feature residual；
10. B7：同时关闭两条显式坐标增强路径；
11. B8：GraphSAGE vs 参数匹配 EdgeCNN；
12. C 系列：`lambda_obj` 敏感性。

## 实验设计说明

- B0–B7 是硬结构消融：被关闭的模块或路径会从前向计算中物理移除，而不是用恒等占位掩盖；
- B8 是算子级受控实验，不属于简单删模块：保持 KNN、双分支、聚合、SE、fusion、feature/coord residual、MLP head 与训练协议不变，仅将局部算子从 GraphSAGE 替换为 CNN 式 EdgeCNN；
- B8 两侧完整模型参数量严格匹配：GraphSAGE 与 EdgeCNN 均为 `1,331,745`，用于隔离分析图卷积消息传递规则本身的贡献；
- B8 页面采用蓝色 GraphSAGE / 橙色 EdgeCNN 对照，并标注 seed 42/43 与分析指标；
- C 系列正式冻结网格为 `{0, 0.25, 0.5, 1.0}`，其中 `0` / `0.5` 分别复用 A2 / A3。

## 预览文件

- `SPAD_model_architectures_preview.pdf`：LibreOffice 渲染 PDF；
- `SPAD_model_architectures_contact_sheet.png`：12 页联系表；
- `preview_slide_01.png` 至 `preview_slide_12.png`：180 DPI 逐页预览。

## 统一配色

- 蓝色：特征分支与 GraphSAGE 算子；
- 青色：坐标/物理分支；
- 紫色：注意力、SE、feature residual；
- 橙色：融合、坐标残差与 EdgeCNN 对照算子；
- 绿色：有效输出；
- 红色虚线：关闭时从模型中物理移除的模块或路径。

## 重新生成

```powershell
& "D:\Anaconda3\envs\torchnew\python.exe" `
  "scripts\generate_model_architecture_ppt.py" `
  --output-dir "model\ppt"
```

## 重新渲染

```powershell
& "D:\Anaconda3\envs\torchnew\python.exe" `
  "scripts\render_ppt_preview.py" `
  --input "model\ppt\SPAD_model_architectures_editable_combined.pptx" `
  --output-dir "temp\ppt_render\manual" `
  --dpi 180
```
