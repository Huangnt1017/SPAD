---
name: pointcloud-3d-workflows
description: Build, debug, and refine 3D point cloud pipelines. Use when working on point cloud classification, semantic segmentation, raw-to-point-cloud conversion, dataset preparation, file parsing, augmentation, sampling, model wiring, training loops, evaluation, inference, tensor shape debugging, or GPU memory checks in 3D workflows.
---

# Point Cloud 3D Workflows

Work from the data contract outward: raw input, parsed point cloud, dataset sample, batch tensors, model input, loss target, and evaluation output.

## Working process
1. Identify the task type first: classification, segmentation, preprocessing, training, evaluation, or inference.
2. Inspect the input format before coding, including path layout, columns, point dimensions, feature channels, and label encoding.
3. Trace the pipeline end to end: raw data -> conversion -> dataset -> augmentation -> model -> loss -> metrics.
4. Prefer the smallest testable fix that preserves existing contracts.
5. Validate assumptions early with sample shapes, dtypes, coordinate ranges, and label counts.

## Task rules
- Use a classification pipeline when labels are per-object or per-cloud.
- Use a segmentation pipeline when labels are per-point.
- If the raw input is not yet a point cloud, fix or add the conversion step before modifying model code.
- If training fails, check parsing, batching, target encoding, and loss compatibility before changing architecture.
- If augmentation changes geometry, confirm labels stay aligned.

## Coding rules
- Write comments in clear technical Chinese when adding or updating comments.
- Keep changes local and avoid refactoring unrelated components.
- Add type hints to new public functions and important internal utilities.
- Validate shape, dtype, range, and required keys early with clear errors.
- Preserve deterministic behavior when the workflow depends on seeds, splits, or reproducible sampling.
- Do not swallow exceptions silently; either handle them with context or re-raise actionable errors.
- Use semantic names such as `points`, `features`, `labels`, `logits`, `valid_mask`, and `sample_indices`.

## Comment and docstring rules
- Add docstrings to public functions, dataset adapters, and model-facing utilities.
- Describe tensor shape transitions near layout changes such as `(B, N, 4) -> (B, 4, N)`.
- Explain non-obvious branches, sampling logic, masking rules, and label alignment assumptions.
- Mark temporary logic with `TODO(name) YYYY-MM-DD: next action`.
- Remove stale comments when logic changes.

## Validation checklist
- Load a small sample and confirm point count, feature dimensions, and label shape.
- Check normalization, sampling, and augmentation behavior on at least one batch.
- Verify dataset outputs match the model contract exactly.
- Confirm model outputs and loss targets use compatible shapes, dtypes, and label spaces.
- Run a short sanity pass to catch device, shape, and dtype issues.
- Review metrics and failure cases using the same label space as training.

## GPU memory check
- When a model file has a local `if __name__ == "__main__":` validation block, add a lightweight CUDA memory smoke test when relevant.
- Report GPU name and total memory.
- Sweep a small batch-size set such as `4, 8, 16, 32` until OOM or completion.
- For each batch size, recreate the model, move it to CUDA, run forward and backward once, then print peak allocated memory.
- Catch `torch.cuda.OutOfMemoryError`, report the failing batch size, and stop the sweep cleanly.
- Support common model outputs such as a tensor, a tuple, or a dict when building the temporary loss.

## Rapid defect scan for long files
- First scan imports, function and class signatures, config constants, decorators, inheritance, and `TODO` or debug prints.
- Produce at most three likely defect areas before reading deeper.
- Then read only the narrow code region around each candidate and identify the root cause directly.
- Suggest focused modifications instead of broad rewrites.

## Definition of done
- The pipeline completes a short end-to-end sanity run, or the remaining blocker is explicitly identified.
- Comments and docstrings explain intent, data flow, and shape expectations.
- New code follows naming, typing, and error-handling rules above.
- Unrelated files and behaviors are left unchanged.
