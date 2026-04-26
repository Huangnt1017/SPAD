---
name: pointcloud-3d-workflows
description: 'Build and debug 3D point cloud pipelines for classification, semantic segmentation, data augmentation, file loading, preprocessing, and neural network training. Use when working with point clouds, raw-to-point-cloud conversion, dataset preparation, model selection, training loops, evaluation, or inference.'
argument-hint: 'task description, data format, or model goal'
user-invocable: true
---

# 3D Point Cloud Workflows

## When to Use
Use this skill for repeatable 3D point cloud work across the SPAD project, especially when the task involves:
- point cloud classification
- semantic segmentation
- data augmentation and sampling
- point cloud file reading and conversion
- neural network training, evaluation, or inference
- debugging data pipelines or tensor shape mismatches

## Working Process
1. Identify the task type first: classification, semantic segmentation, data preparation, training, or inference.
2. Inspect the input data format and source files before coding.
3. Decide the smallest valid pipeline for the task:
   - classification: one label per cloud or object
   - segmentation: one label per point
   - augmentation: preserve labels while modifying geometry or sampling
   - file reading: verify paths, delimiters, point dimensions, and label alignment
4. Trace the data path end to end: raw data -> point cloud conversion -> dataset loader -> augmentations -> model -> loss -> metrics.
5. Prefer local, testable changes over broad refactors.
6. Validate assumptions early by checking sample shapes, label counts, coordinate ranges, and batch output.

## Code Comment Requirements
- Write comments in clear technical Chinese.
- Prefer comments that explain intent and data flow, not obvious syntax.
- For public APIs, include docstrings with: purpose, Args, Returns, and Raises.
- For internal helpers, add short comments only at key logic boundaries and non-obvious constraints.
- For complex conditions or loops, explain why the branch exists and what invariant it protects.
- Explicitly describe tensor shape transitions when data changes form (for example, `(B, N, 4) -> (B, 3, N)`).
- Mark temporary solutions with this exact format: `// TODO(username) YYYY-MM-DD: reason and next action`.
- Avoid stale comments: update or remove comments when logic changes.

## Code Writing Standards
- Keep changes minimal and local; do not refactor unrelated code.
- Preserve existing data contracts between dataset, model, loss, and metrics.
- Use explicit type hints for new public functions and key internal utilities.
- Validate inputs early with clear error messages (shape, dtype, range, missing keys).
- Keep deterministic behavior where expected (seed usage, split behavior, augmentation reproducibility).
- Prefer pure helper functions for reusable geometry/math operations.
- Do not swallow exceptions silently; either handle with context or re-raise with actionable detail.
- Ensure naming reflects semantics (`logits`, `box_preds`, `box_targets`, `valid_mask`, etc.).

## Decision Rules
- If the labels are per-object, use a classification pipeline.
- If the labels are per-point, use a semantic segmentation pipeline.
- If the raw input is not yet a point cloud, add or update the conversion step before model work.
- If the model fails to train, check file parsing, batch shapes, label encoding, and loss-target compatibility before changing architecture.
- If augmentation changes geometry, confirm labels remain aligned and unchanged where required.

## Implementation Checklist
1. Load a small sample and print the point count, feature dimensions, and label shape.
2. Confirm normalization, sampling, and augmentation behavior on one batch.
3. Verify the dataset returns the exact tensors expected by the model.
4. Choose or adapt a network suited to the task and input shape.
5. Connect the training loop with a loss function and metrics that match the labels.
6. Run a short training pass to catch shape, dtype, or convergence issues.
7. Review outputs, confusion patterns, and segmentation masks or predicted classes.

## Review Checklist
1. Public functions include docstrings with Args, Returns, and Raises.
2. Key data-flow steps are commented (loader -> augment -> model -> loss -> metrics).
3. Complex conditions/loops include rationale comments.
4. Temporary logic is tagged using `// TODO(username) YYYY-MM-DD`.
5. Tensor shapes are consistent across dataset, model forward, loss, and evaluation.
6. Top-k, IoU, and AP metrics handle edge cases (small class counts, empty subsets, invalid boxes).
7. Logging/reporting exposes enough context to debug failed samples or skipped evaluations.
8. A short sanity run completes without shape, dtype, or device errors.

## Rapid Defect Identification Protocol (for files >200 lines)
Use this protocol when you need to **quickly locate defects** in a long pipeline script without reading it line by line. It is especially effective for debugging data loaders, augmentation chains, training loops, or evaluation code.

### Step 1: Structural Scan (token cost ≤ 10% of file length)
Scan the file **only** for these structures – do not read the full code:
- Imported libraries (note suspicious third‑party package versions)
- Class/function definition names and parameter lists (do not inspect function bodies)
- Global variables, configuration constants
- Decorators, inheritance relationships
- Any `TODO`, `FIXME`, `print` or `log` statements

Output a list of **up to 3 candidate problem areas**. Examples for point‑cloud code:
- “`learning_rate` initialized to `0` on line 45”
- “Possible point‑cloud shape mismatch in preprocessing lines 120–150”
- “`try‑except` on line 380 swallows the exception”
- “Incorrect normalization factor in `PointCloudAugment` line 210”
- “Label index off‑by‑one in segmentation mask creation line 95”

### Step 2: Read Key Code Snippets
For each candidate area, read **only the 10–20 lines** around that location. For example:
“Read the code from lines 120 to 150, and focus on point tensor shape changes inside the loop.”

### Step 3: In‑Depth Analysis
After reading the short snippet, provide a concise analysis:
- Identify the root cause of the error directly.
- Suggest a concrete modification (total output ≤ 15 lines).
- Do not explain basic syntax; do not repeat code unless highlighting a specific change.

### Step 4 (Optional)
If the issue is still unresolved, ask whether you should scan another candidate area.

### Prohibited Actions
- Do not output meaningless phrases like “Let’s analyze this step by step”
- Do not output duplicate content from the entire file
- Do not ask the user to provide additional information from “the entire file” (for example, printing all variables)

**User cooperation:** You must provide the file, allow only a structural scan, and follow the instruction to paste the short code snippets for the candidate areas.

## Definition of Done
- The modified pipeline passes a short end-to-end run.
- Comments are concise, accurate, and focused on intent/data flow.
- New code follows naming, typing, and error-handling standards above.
- No unrelated files or behaviors are changed.

## Quality Checks
- Input files are parsed consistently.
- Augmentations do not break label correspondence.
- Model input and output shapes match the dataset contract.
- Training loss decreases on a small sanity-check subset.
- Evaluation metrics are reported in the same label space used during training.

## Common Outputs
- dataset loader or parser fixes
- augmentation utilities
- training scripts and evaluation loops
- classification and segmentation model wiring
- debugging notes for point cloud shape and label alignment issues

## References
- See the repo utilities and data readers for existing point cloud conversion and training patterns.