# SPAD 项目编码规范

本项目所有 Python 代码改动**必须**遵守以下规则。完整 skill 定义见 `.claude/skills/`。

## 注释语言
- **修改、新增或修改代码注释一律用清晰的技术中文**。
- 不使用模糊词或翻译腔；术语保留英文（如 `forward`、`tensor`、`logits`、`bbox`）。

## 注释义务（不可省略）
1. **形状变换前**必须注释：`reshape / view / permute / transpose / flatten / squeeze / unsqueeze / cat / stack`
   - 写明形状走向，如 `# (B, N, 4) → (B, 4, N)`。
2. **公共函数/类**必须有 docstring，包含 `Args / Returns / Raises`（仅当相关时）。
3. **张量重的代码**docstring 里写关键 shape，例如 `pred: [B, 3] 中心点 / gt: [B, 6] 角点`。
4. **数据流走向**——尤其是数组在管道中形状变化的关键节点——必须注释。
5. **非显然的常量**说明其来源/含义。
6. **临时逻辑**用 `TODO(name) YYYY-MM-DD: 下一步动作` 标记。
7. **不要写"什么"型注释**（代码已说明）；写"为什么/怎么/边界条件"。
8. **注释语言**必须使用中文给出详细注释，在类或函数正式开始前介绍其作用

## 命名（PEP 8）
- 模块/函数/方法/变量/属性: `snake_case`
- 公共类: `CapWords`；私有类: `_CapWords`
- 模块级常量: `UPPER_SNAKE_CASE`
- 布尔: `is_xxx / has_xxx / enable_xxx`
- **禁止**模糊名: `data / result / temp / value / item / dict / list / d / r / t / tmp`（小作用域单字母可: `i j k x y z`）
- 用语义名: `points / features / labels / logits / valid_mask / sample_indices / pred_centers`

## 类型 & 异常
- 新加的公共函数和重要内部辅助加 type hints。
- 入口/边界先验证 shape/dtype/range/必需键，错误信息要可执行（说出实际值与期望值）。
- 不静默吞异常；要么带上下文处理，要么再抛。

## 改动边界
- 保留现有公共接口行为；只在任务明确要求时改。
- 最小局部改动 > 大范围重写；不顺手重构无关组件。
- 重复逻辑提取为可复用辅助；优先纯函数。
- 当改动代码逻辑后及时更新注释。

## Point cloud 管道工作流
- 改动前先理清数据契约：raw → parsed → dataset 样本 → batch tensor → model 输入 → loss 目标 → 评估输出。
- 训练失败时先查：数据解析 / batching / 目标编码 / loss 兼容性，**最后**才动模型架构。
- 增强若改几何，确认 label 同步对齐。

## `__main__` 显存测试规约
当 baseline 文件含 `if __name__ == "__main__":` 时：
- 打印 GPU 型号与显存。
- 批量扫 `[4, 8, 16, 32]`，对每个 size：重建模型 → 上 CUDA → forward+backward → 打印峰值显存。
- 捕获 `torch.cuda.OutOfMemoryError`，报失败 batch 后停止扫描。
- 支持 tuple / dict / tensor 三种返回构造临时 loss。

## 长文件快速排错
1. 先扫 imports / 类与函数签名 / 配置常量 / 装饰器 / 继承 / TODO / debug print。
2. 给出≤3 个高嫌疑区。
3. 只读这几个窄窗，定位根因。
4. 给出聚焦修改，不做大改。

## 完成定义
- 端到端 sanity 跑通，或明确指出剩余阻塞。
- 注释+docstring 解释了意图、数据流、shape 期望。
- 命名/类型/异常处理符合上面规则。
- 无关文件无关行为不动。
