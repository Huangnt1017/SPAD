# Project Instructions (Auto-generated)

> This file is editable —DeepSeek TUI loads it as the project context pack.

**Project:** SPAD — 单光子激光雷达目标检测与分布建模研究

## Navigation
- `main.py` — 项目入口
- `scripts/train.py` — 训练脚本 (含 save_checkpoint 已迁移至 utils/checkpoint)
- `scripts/test.py` — 测试脚本
- `utils/data.py` — 数据集加载
- `utils/loss.py` — 多任务损失
- `utils/checkpoint.py` — checkpoint 存储/恢复
- `model/readme.md` — 核心研究框架文档 (任务1/2/3)
- `model/graph_residual.py` — 图残差多任务网络 (主模型)
- `baseline/` — 对比基线模型 (DGCNN, PointNet++, PointTransformer 等)

## Skill Files
> DeepSeek TUI loads skills from the **global** `## Skills` block in the system prompt.
> When a Skill's path points to `C:\Users\10721\.deepseek\skills\...`, that is the canonical source.
> Project-local skills at `.github/skills/` may be stale copies — prefer the global path.

### pointcloud-3d-workflows (canonical: global)
**Trigger**: Always use when working with point clouds, raw-to-point-cloud conversion, dataset preparation, model selection, training loops, evaluation, or inference in this project.

**Core Rules (summary)**:
1. **Validate inputs early**: check sample shapes, label counts, coordinate ranges, batch output before coding.
2. **Do not swallow exceptions silently**: handle with context or re-raise with actionable detail.
3. **Preserve data contracts**: dataset → model → loss → metrics must stay aligned.
4. **Comments**: technical Chinese, explain intent/data flow, document tensor shape transitions (e.g. `(B,N,4)→(B,3,N)`).
5. **Deterministic behavior**: use seed where expected, avoid non-reproducible ops.
6. **Minimal & local changes**: do not refactor unrelated code.
7. **Review before done**: public functions need docstrings with Args/Returns/Raises; complex conditions need rationale comments; a short sanity run must complete without shape/dtype/device errors.
8. **Rapid Defect Identification**: for files >200 lines, structural scan first → read key snippets → root cause analysis → concrete fix (≤15 lines).
9. **PyTorch API compatibility**: `torch.cuda.get_device_properties` property names vary (total_memory vs total_mem). `torch.utils.checkpoint` may not exist on old versions. Always use hasattr/fallback try-except when accessing CUDA/checkpoint APIs.

### python-code-assistant (canonical: global)
- PEP 8, 4-space indent, 100-char lines
- Google-style docstrings (Args, Returns, Raises)
- Explicit type hints on all public functions
- Comments explain *why*, not *what*
- `pathlib.Path` for file paths, no bare `except:`

## Common Pitfalls / Reminders
1. **checkpoint API**: use `from utils.checkpoint import save_checkpoint` — not `scripts/train.save_checkpoint`.
2. **cache invalidation**: `__pycache__/` and `.split_cache.json` may need manual cleanup when data/labels change.
3. **CUDA memory**: the `GraphResidualMultiTaskNet` uses chunked EdgeFeature (max_chunk_size=4) + gradient checkpoint (use_checkpoint=True). Default k=16. Test before increasing batch_size.
4. **SPADdata path**: `D:\PYproject\SPADdata\2025-04-30-dpc\` with subdirs A-Z.

**Tree:**
```
DIR: .git
DIR: .github
  DIR: skills
    DIR: pointcloud-3d-workflows
      FILE: SKILL.md
    DIR: python-code-assistant
      FILE: SKILL.md
DIR: HMC
  FILE: HMC_sample.py
  FILE: void.py
DIR: baseline
  FILE: 3DETR.py
  FILE: DCT.py
  FILE: DGCNN.py
  FILE: PointMLP.py
  FILE: PointNet++.py
  FILE: PointTransformer.py
DIR: checkpoints
DIR: data_read
  FILE: ILOF.py
  FILE: pc_infrared.py
  FILE: raw2pc.py
  FILE: raw2pc2.py
  FILE: raw2pointcloud.py
  FILE: test.py
DIR: logs
FILE: main.py
DIR: model
  FILE: readme.md
  FILE: graph_residual.py
  FILE: __init__.py
DIR: scripts
  FILE: __init__.py
  FILE: test.py
  FILE: test1.py
  FILE: train.py
DIR: utils
  FILE: SSIM.py
  FILE: checkpoint.py
  FILE: data.py
  FILE: data_augment.py
  FILE: loss.py
```
