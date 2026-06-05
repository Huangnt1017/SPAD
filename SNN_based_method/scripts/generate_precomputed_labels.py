"""兼容入口: 真实实现已迁移到 ``SNN_based_method.utils.generate_precomputed_labels``。"""

from __future__ import annotations

import sys

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.utils.generate_precomputed_labels import *  # noqa: F401,F403
from SNN_based_method.utils.generate_precomputed_labels import main, main_without_cli


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
