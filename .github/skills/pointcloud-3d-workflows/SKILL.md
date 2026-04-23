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