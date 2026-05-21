---
name: python-code-assistant
description: Write, refactor, debug, review, and document Python code. Use when working on Python functions, classes, modules, scripts, tests, type hints, exceptions, comments, docstrings, API wrappers, dependency integration, or code quality improvements that should follow PEP 8, clear naming, and maintainable structure.
---

# Python Code Assistant

Provide Python code that is directly runnable, clearly named, type-safe where practical, and easy to maintain.

## Core rules
- Prefer minimal, local changes over broad rewrites.
- Follow PEP 8 and keep naming semantic and self-explanatory.
- Add type hints to new public functions, methods, and important internal helpers.
- Validate inputs early and raise clear exceptions instead of failing silently.
- Preserve existing public behavior unless the task explicitly requires a change.

## Naming rules
- Use `snake_case` for modules, functions, methods, variables, and attributes.
- Use `CapWords` for public classes and `_CapWords` only for private internal classes when needed.
- Use `UPPER_SNAKE_CASE` for module-level constants.
- Use boolean prefixes such as `is_`, `has_`, or `enable_`.
- Avoid vague names such as `data`, `result`, `temp`, `value`, `item`, `dict`, `list`, `d`, `r`, `t`, and `tmp` unless the scope is trivial.
- Reserve single-letter names for very small local scopes such as loop indices `i`, `j`, `k` or conventional coordinates `x`, `y`, `z`.

## Documentation rules
- Add a module docstring when creating a new module or substantially restructuring one.
- Add docstrings to public classes and public functions.
- For public callables, include `Args`, `Returns`, and `Raises` when relevant.
- For tensor-heavy or array-heavy code, describe important shapes in docstrings and nearby comments.
- For non-obvious branches, explain intent rather than restating syntax.

## Comment rules
- Keep comments concise and technical.
- Comment shape transitions before `reshape`, `view`, `permute`, `transpose`, or similar layout changes.
- Explain the reason for important constants when they are not self-evident.
- Mark temporary logic with `TODO(name) YYYY-MM-DD: next action`.
- Do not leave commented-out code in files.

## Preferred structure
- Order class methods as `__init__`, public methods, then private helpers.
- Keep helpers small and focused.
- Extract repeated logic into local reusable helpers instead of duplicating branches.
- Prefer pure functions for reusable transformations when state is unnecessary.

## Output checklist
- The code runs or is internally consistent with the surrounding codebase.
- Names reflect intent and data meaning.
- Public APIs are documented.
- Type hints match real inputs and outputs.
- Error messages are actionable.
- Comments clarify data flow, invariants, or shape changes.
