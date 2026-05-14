---
name: python-code-assistant
description: 当用户请求涉及 Python 代码编写、修改、审查、注释或架构设计时，自动提供符合 PEP 8、类型安全、文档完善的代码。
allowed-tools: Read, Write, Bash
triggers:
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
  - "命名"
  - "变量"
---

# Python 代码编写与注释规范

## 适用范围
所有生成的 Python 代码必须**可直接运行**、**类型安全**、**命名语义化**、**文档完整**。本规范适用于函数、类、方法、模块、变量等一切代码元素。

---

## 1. 命名规范

### 1.1 通用规则
- 名称必须**自解释**——读者无需查看实现即可理解其用途
- 单字母变量仅允许: 循环索引 `i, j, k`、坐标 `x, y, z`、批量维度缩写 `B, N, C`
- 禁止: 拼音命名、无意义缩写(`d`, `r`, `t`)、拼写错误

### 1.2 分类命名表

| 元素 | 风格 | 示例 |
|---|---|---|
| 模块 / 文件 | `lower_with_under.py` | `pointnet_utils.py`, `data_augment.py` |
| 公共类 | `CapWords` | `PointTransformerLayer`, `SetAbstraction` |
| 私有类 | `_CapWords` | `_CustomBatchNorm` |
| 公共函数 | `lower_with_under()` | `compute_loss()`, `farthest_point_sample()` |
| 私有/内部函数 | `_lower_with_under()` | `_init_weights()`, `_wkv_forward()` |
| 公共方法 | `lower_with_under()` | `forward()`, `build_loss_func()` |
| 参数 / 局部变量 | `lower_with_under` | `num_classes`, `embed_dim`, `drop_path_rate` |
| 模块级常量 | `CAPS_WITH_UNDER` | `PROJECT_ROOT`, `DEFAULT_GRID_SIZE` |
| 属性名 | `lower_with_under` | `self.in_channels`, `self.num_heads` |
| 布尔变量 | `is_` / `has_` / `enable_` 前缀 | `is_training`, `has_bias`, `enable_checkpoint` |
| 计数量 | `num_` / `count_` 前缀 | `num_points`, `count_samples` |

### 1.3 禁止的命名 (模糊通名)
`data`, `input`, `output`, `result`, `temp`, `value`, `item`, `info`, `array`, `list`, `dict`, `d`, `r`, `t`, `tmp` —— 一律替换为语义明确的具体名称。

---

## 2. 文档字符串 — 强制标准

### 2.1 必须包含 docstring 的元素
| 元素 | 要求 |
|---|---|
| 模块文件头 | 说明用途 + 列出主要导出 |
| 公共类 | 含 `__init__` 各参数 + 至少 1 个 Example |
| `__init__` 方法 | 列出所有构造参数及其用途 |
| `forward` 方法 | Args 含张量形状 `(B,N,C)`, Returns 含形状 |
| 公共函数/方法 | 含 Args/Returns/Raises(视情况) |
| 私有复杂方法(>10行或≥3分支) | 至少一行用途说明 |

### 2.2 模块头标准模板
```python
"""模块用途简述.

主要导出:
    ClassName  — 一句话说明
    func_name  — 一句话说明
"""
```

### 2.3 类 docstring 标准模板
```python
class MyModule(nn.Module):
    """一句话概述.

    详细描述架构流程.

    Args:
        in_channels: 输入通道数.
        out_channels: 输出通道数.

    Example:
        >>> m = MyModule(3, 64)
        >>> y = m(x)  # (B, 64, N)
    """
```

### 2.4 函数/方法 docstring 模板
```python
def farthest_point_sample(xyz, npoint):
    """最远点采样 (FPS).

    贪心选取 npoint 个相互距离最远的采样点.

    Args:
        xyz: (B, N, 3) 点云坐标.
        npoint: 采样点数.

    Returns:
        centroids: (B, npoint) 采样点索引 (dtype=long).
    """
```

---

## 3. 内联注释与形状注解

### 3.1 必须注释的位置
1. **张量形状变换** — reshape / permute / transpose 前一行注明目标形状
2. **非显而易见的分支** — 解释为什么有这个分支
3. **魔法数字** — 如 `0.02` / `1e-5` 注明来源
4. **临时方案** — `# TODO(author) YYYY-MM-DD: reason`

```python
# 格式转换: (B, N, 3) → (B, 3, N) 以适应 Conv1d
pts = x[:, :, :3].transpose(1, 2).contiguous()  # (B, 3, N)

# 用 KNN k=16, 平衡感受野与计算开销
idx = knn_point(16, xyz, xyz)  # (B, N, 16)

# truncated_normal std=0.02 来自 ViT 论文
nn.init.trunc_normal_(m.weight, std=0.02)
```

### 3.2 禁止的注释
- `# 赋值` / `# 调用` — 纯语法重复, 直接删除
- 被注释掉的代码 — 使用 git 管理历史, 不要留在文件中

---

## 4. 类内部结构约定

```python
class MyModel(nn.Module):
    """类概述 (必须)."""

    def __init__(self, ...):
        super().__init__()
        # ---- 核心参数 ----
        self.dim = dim            # 特征维度

        # ---- 子模块 ----
        self.embedding = nn.Linear(dim, dim * 4)   # 嵌入层: D → 4D

        # ---- 可学习参数 ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # ---- 初始化 ----
        self.apply(self._init_weights)

    def forward(self, x):
        """前向传播 (...)."""
        ...

    def _init_weights(self, m):
        """权重初始化."""
        ...
```

规则: 方法顺序为 `__init__` → 公开方法 → 私有方法; `__init__` 内用 `# ---- 分组 ----` 分隔; 属性声明必须带行内注释.

---

## 5. 完整代码生成示例

```python
"""PointNet++ Set Abstraction 模块."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """最远点采样 (FPS).

    Args:
        xyz: (B, N, 3) 点云坐标.
        npoint: 采样点数, 必须 <= N.

    Returns:
        centroids: (B, npoint) 采样点索引 (long).

    Raises:
        ValueError: 若 npoint > N.
    """
    B, N, C = xyz.shape
    if npoint > N:
        raise ValueError(f"npoint ({npoint}) > N ({N})")

    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)       # 初始化为极大值
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)  # (B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)            # (B, N)
        mask = dist < distance         # 只更新距离变小的点
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


class SetAbstraction(nn.Module):
    """PointNet++ Set Abstraction (SA) 模块.

    流程: FPS → BallQuery → 中心归一化 → MLP(Conv2d) → 最大池化.

    Args:
        npoint: FPS 采样点数. None 时 group_all=True.
        radius: Ball Query 半径. group_all 时忽略.
        nsample: 每组最大近邻数.
        in_channel: 输入特征通道 (不含 xyz 的 3 维).
        mlp: MLP 输出通道列表, e.g. [64, 64, 128].
        group_all: 全部点作为一组 (最后一层 SA).

    Example:
        >>> sa = SetAbstraction(512, 0.2, 32, 1, [64, 64, 128])
        >>> new_xyz, new_feat = sa(xyz, points)
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
    ) -> None:
        super().__init__()
        self.npoint = npoint               # FPS 采样点数
        self.radius = radius               # Ball Query 半径
        self.nsample = nsample             # 每组最大近邻数
        self.group_all = group_all         # 全局分组标志

        # MLP: 输入 = 归一化坐标(3) + 点特征(in_channel)
        layers: List[nn.Module] = []
        last_channel = in_channel + 3
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,
        points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            xyz: (B, 3, N) 点云坐标.
            points: (B, D, N) 点特征, 可为 None.

        Returns:
            new_xyz: (B, 3, S) 采样后坐标.
            new_points: (B, C_out, S) 池化后特征.
        """
        B, _, N = xyz.shape
        xyz_nc = xyz.transpose(1, 2).contiguous()            # (B, N, 3)
        if points is not None:
            points_nc = points.transpose(1, 2).contiguous()   # (B, N, D)
        else:
            points_nc = None

        if self.group_all:
            new_xyz_nc, new_points = self._group_all(xyz_nc, points_nc)
        else:
            new_xyz_nc, new_points = self._group(xyz_nc, points_nc)

        # Conv2d 处理: (B, C, K, S)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2)[0]          # 最大池化 → (B, C_out, S)

        new_xyz = new_xyz_nc.transpose(1, 2).contiguous()     # (B, 3, S)
        return new_xyz, new_points
```

---

## 6. 自我审查清单

| # | 检查项 | 达标标准 |
|---|---|---|
| 1 | 模块头 docstring | 写明用途 + 主要导出 |
| 2 | 公共类 docstring | 含 Args + Example |
| 3 | `__init__` docstring | 所有构造参数均已列出 |
| 4 | `forward` docstring | Args/Returns 含张量形状 `(B,N,C)` |
| 5 | 公共方法 docstring | >5 行必须有 Args/Returns |
| 6 | 私有方法 docstring | >10 行或 ≥3 分支必须有 |
| 7 | 类型提示 | 所有公开函数/方法的参数与返回值 |
| 8 | 张量形状注释 | 每次 reshape/permute/transpose 前一行 |
| 9 | 变量名语义化 | 禁止 x/data/result/temp 等通名 |
| 10 | 魔力数字有注释 | 如 0.02/1e-5 有来源 |
| 11 | 分支有注释 | if/else 意图可见 |
| 12 | TODO 规范 | `# TODO(name) YYYY-MM-DD: why` |
| 13 | 命名风格 | 类 CapWords, 函数 lower_under, 常量 CAPS |
| 14 | 布尔前缀 | is_/has_/enable_ |
| 15 | 导入顺序 | 标准库 → 第三方 → 本地 |

> **验收标准**: 任何一位未接触过该代码的同事，只看 docstring + 注释即可理解: 每个类/函数的作用、每个参数的含义、输入输出的张量形状.
