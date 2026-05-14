
---
name: python-code-assistant
description: 当用户请求涉及 Python 代码编写、修改、审查、注释或架构设计时，自动提供符合 PEP 8、类型安全、文档完善的代码。
allowed-tools: Read, Write, Bash
triggers:
  # 英文触发词（核心）
  - "python"
  - "code"
  - "function"
  - "class"
  - "module"
  - "package"
  - "import"
  - "script"
  - "implement"
  - "refactor"
  - "debug"
  - "optimize"
  - "type hint"
  - "docstring"
  - "comment"
  - "exception"
  - "decorator"
  - "generator"
  - "async"
  - "lambda"
  - "algorithm"
  - "data structure"
  - "test"
  - "pytest"
  - "unittest"
  - "mock"
  - "ci/cd"
  - "pipeline"
  - "api"
  - "rest"
  - "sdk"
  - "library"
  - "dependency"
  # 中文触发词（辅助）
  - "代码"
  - "函数"
  - "类"
  - "模块"
  - "调用"
  - "实现"
  - "重构"
  - "调试"
  - "优化"
  - "注释"
  - "类型提示"
  - "文档字符串"
  - "异常"
  - "装饰器"
  - "测试"
  - "接口"
  - "库"
  - "依赖"
  - "功能"
  - "复现"
---

# Python 代码编写与注释规范 (Code & Documentation Standards)

## 适用范围
当用户请求涉及 Python 代码的编写、修改、注释、审查、测试或架构设计时，始终遵循本文件定义的规范。所有生成的代码必须**可直接运行**、**类型安全**、**文档完整**。

---

## 1. 代码风格 (Code Style)

### 1.1 基础格式
- 严格遵循 **PEP 8**。
- 缩进使用 **4 个空格**，禁止使用 tab。
- 每行最多 **100 个字符**（文档字符串或注释可放宽至 120）。
- 文件末尾有且仅有一个换行符。
- 导入顺序：标准库 → 第三方库 → 本地模块，每组之间空一行。
  ```python
  import os
  import sys
  from typing import Optional

  import numpy as np
  import torch

  from my_package import utils
  ```

### 1.2 命名规范
- 模块/包：`lower_with_under.py`
- 类：`CapWords`（如 `PointCloudProcessor`）
- 函数/方法：`lower_with_under()`（如 `compute_intensity`）
- 变量：`lower_with_under`
- 常量：`CAPS_WITH_UNDER`（模块级）
- 私有成员：以单下划线 `_` 开头（protected）、双下划线 `__` 开头（name mangling，非必要不用）

---

## 2. 类型提示 (Type Hints)
所有公共函数、方法、类属性**必须**有完整的类型提示。使用内置泛型或 `typing` 模块。
```python
from typing import List, Tuple, Optional, Union
import torch

def fit_bayesian_model(
    points: torch.Tensor,          # shape: (N, 4) [x, y, z, intensity]
    prior_strength: float = 1.0,
    n_iter: int = 1000
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    ...
    """
    ...
```

---

## 3. 文档字符串 (Docstrings)

一律使用 **Google 风格** docstring，包含：
- 简短描述（第一行）
- 详细说明（可选，空行后）
- Args: 参数列表，每项包含类型和解释
- Returns: 返回值类型及解释
- Raises: 可能抛出的异常
- Example (可选): 使用示例

```python
def denoise_lidar(
    raw_data: np.ndarray,
    fog_model: str = "gamma"
) -> np.ndarray:
    """使用贝叶斯方法去除 LiDAR 点云中的雾气噪声。

    该函数对 64x64 像素、128 时间 bin 的输入张量进行逐像素的
    伽马-高斯混合分布拟合，并返回去噪后的强度图像。

    Args:
        raw_data: 形状为 (64, 64, 128) 的数组，值为光子计数强度。
        fog_model: 雾分布模型，可选 "gamma" 或 "exp".

    Returns:
        与输入同形状的数组，体素强度为去除雾气分量后的期望值。

    Raises:
        ValueError: 如果 raw_data 包含 NaN 或形状不为 (64, 64, 128).
    """
    ...
```

---

## 4. 注释 (Comments)

### 4.1 代码注释原则
- 解释 **“为什么”** 这样做，而不是重复代码干了什么。
- 复杂算法、临时解决方案（workaround）、性能敏感处必须添加注释。
- 注释与代码同行或单独一行，使用 `# ` 开头（井号后一个空格）。
- 避免无意义的注释（如 `# 赋值给 x`）。
- TODO/FIXME 标签格式：
  ```python
  # TODO(作者): 采用量子隧穿改进此处 MCMC 跳跃，预计 v1.2 完成
  # FIXME(作者): 当 NaN 出现时回退到先验，参见 issue #42
  ```

### 4.2 类型注释补充
对于复杂类型，可使用 `# type:` 辅助说明（虽然推荐使用原生 type hints，但在遗留代码或动态场景可用）：
```python
result = get_config()  # type: dict[str, Any]
```

---

## 5. 异常与错误处理 (Exception Handling)
- 捕获**具体的**异常，严禁使用裸露的 `except:`。
- 在 docstring 的 Raises 中记录可能传播的异常。
- 自定义异常类继承自 `Exception`，类名以 `Error` 结尾。
```python
class PointCloudFormatError(Exception):
    """当点云文件格式不符合预期时抛出。"""
    pass
```

---

## 6. 模块与包设计 (Module Design)
- 模块职责单一，功能相近的函数/类归入同一模块。
- `__init__.py` 中明确导出公共 API，使用 `__all__`。
- 所有配置项集中放在 `config.py` 或 `settings.py` 中，使用 pydantic 或 dataclass 管理。
- 处理 I/O、算法、可视化分开为独立模块：`io.py`, `algorithms.py`, `viz.py`。

---

## 7. 测试 (Testing)
- 使用 `pytest` 框架。
- 测试文件命名：`test_<模块名>.py`，放置在 `tests/` 目录下。
- 函数测试命名：`test_<函数名>_<场景>_<预期结果>`。
- 对外部依赖（文件系统、网络、数据库）使用 `unittest.mock` 或 `pytest-mock`。

---

## 8. 性能与安全提醒
- 避免在循环中进行重复属性访问（如 `len(list)` 挪到循环外）。
- 大数据集使用生成器表达式 (`(...)`) 而非列表推导式 (`[...]`)。
- 禁止使用 `eval()` 或 `exec()` 处理用户输入。
- 文件路径拼接使用 `pathlib.Path`。

---

## 9. 代码生成示例
当被要求“创建一个从点云中提取特征的类”时，应输出如下结构：
```python
"""点云特征提取模块，提供 PointCloudFeatureExtractor 类。"""
from pathlib import Path
from typing import Optional
import numpy as np

class PointCloudFeatureExtractor:
    """从 LiDAR 点云中计算几何与强度特征。"""
    def __init__(self, voxel_size: float = 0.1) -> None:
        self.voxel_size = voxel_size

    def extract(self, points: np.ndarray) -> np.ndarray:
        """提取特征向量。

        Args:
            points: (N, 4) 数组 [x, y, z, intensity].

        Returns:
            (N, D) 特征矩阵.
        """
        ...
```

---

## 注意事项
- 本 Skill 为**补充性**规范，不覆盖项目已有特定风格（如 `black` 格式化配置），但需在回答中提醒用户保持一致性。
- 当用户未指定风格时，默认使用本文档的约定。
```

---

### 📌 补充说明：语言对 Skill 的影响

- **Skill 正文**：中英文混写完全没问题，AI 理解两种语言的能力都很强。关键是将规则描述得**清晰无歧义**。
- **触发词 (triggers)**：建议 **英文优先，中文做补充**。因为代码领域的术语在 prompt 中更多以英文出现（如 “refactor this function”），但中文用户也会说“帮我注释这个函数”。同时列出两者能保证最大化触发率。
- 如果你的工作场景全是中文交互，可以把 `triggers` 全部换成中文词汇。

将上述代码块保存为 `.github/skills/python-code-assistant/SKILL.md`，然后在你的项目文件夹中使用 Copilot Chat，AI 就会自动继承这套“代码质量守则”。