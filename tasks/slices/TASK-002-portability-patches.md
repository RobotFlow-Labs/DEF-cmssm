# TASK-002 — Upstream Portability Patches
- Priority: P0
- Depends on: TASK-001

## Scope
Patch upstream import/path issues while preserving source behavior.

## Acceptance
1. `proposed.*` imports are remapped to local package namespace.
2. `backbone.covnextV2` typo corrected to `backbone.convnextV2`.
3. Hardcoded absolute paths replaced by config/path resolver.
