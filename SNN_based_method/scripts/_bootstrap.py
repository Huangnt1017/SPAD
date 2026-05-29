"""SNN 脚本的路径自举工具。

这些脚本既要支持从项目根目录直接运行:
    python SNN_based_method/scripts/train.py

也要支持以包模块方式运行:
    python -m SNN_based_method.scripts.train

直接运行脚本时, Python 默认只把 scripts/ 放进 sys.path, 会导致
``SNN_based_method`` 这个包名不可见。因此这里统一把项目根目录加入
sys.path, 让包内绝对导入保持稳定。
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def ensure_project_root_on_path() -> None:
    """确保项目根目录在 sys.path 中, 便于直接运行 scripts/*.py。"""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

