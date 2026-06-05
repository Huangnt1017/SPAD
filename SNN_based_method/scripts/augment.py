"""兼容入口: 真实实现已迁移到 ``SNN_based_method.utils.augment``。"""

from __future__ import annotations

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.utils.augment import *  # noqa: F401,F403
