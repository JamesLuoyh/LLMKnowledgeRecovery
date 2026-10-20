# Test-Iterate Loop — LLMKnowledgeRecovery

**Started:** 2026-10-20 11:13 EST
**Terminal state:** PASS

## Baseline

`uv run pytest -q`: 182 passed. The repair is driven by the implementation-plan review: the isolated unit tests pass, but the command graph, provenance, cache compatibility, GPU execution, and multi-context paths are incomplete.

## Iterations

| # | Hypothesis | Fix | Result |
|---|---|---|---|
| 1 | Request-signature serialization passed dataclass instances to the canonical JSON freezer. | Serialize each intervention definition through its field mapping. | New failure: probe-run exclusion rejected the optimizer batch-size field. |
| 2 | Probe-run identity correctly rejects extraction execution `batch_size`, but the optimizer profile reused that ambiguous key. | Record the scientific probe minibatch as `minibatch_rows` in the optimizer specification. | New failure: legacy summary pointer was no longer written. |
| 3 | Existing callers read `runs/summary.json` even though artifacts now live in a content-addressed bundle. | Keep a lightweight root summary containing the bundle ID while preserving immutable bundle artifacts. | Unit suite passed; fresh end-to-end import exposed a circular package import. |
| 4 | Importing the integration module first loaded `qwen -> extraction.__init__ -> identity -> qwen`. | Make the package-level `build_row_key` export lazy so position helpers do not import the model adapter recursively. | Fresh ten-question pipeline completed all stages and saved 80 predictions. |
| 5 | A stable run bundle still let a repeated fit overwrite selection artifacts, and report exports were keyed only by the training run. | Add immutable attempt IDs under the stable run ID; key evaluation artifacts and report exports by evaluation identity. | Two repeated fits produced distinct attempts under one run ID; ECE/OOD-only changes kept that run ID and produced a new evaluation ID. |
| 6 | Cache locking covered shard/index publication but not the collection manifest, and stale-request rejection used an unsupported mismatch label. | Publish the collection manifest under the store lock and classify request-signature drift as an identity incompatibility. | Four concurrent writers produced one shard/index entry; stale configuration was rejected without loading the model. |
| 7 | Repeated supplied contexts could collapse to a variant-level instance label and assume universal instance coverage. | Preserve instance IDs/probabilities in views and predictions; calculate each instance rate over its actual covered base questions. | Distinct instances remained in the view and reported denominators of two and one as expected. |

## Final verification

- `UV_PROJECT_ENVIRONMENT=/tmp/kr-venv UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`: 183 passed, including new variable-context and immutable-attempt regressions.
- `git diff --check`: passed.
- Fresh `validate -> prepare -> audit-tokens -> extract -> diagnose -> build-view -> train -> evaluate -> report`: passed with 80 saved predictions and CSV/LaTeX/JSON exports.
- Repeated training: same run ID, distinct immutable attempt IDs.
- ECE 15 -> 20 plus OOD-view change: same run ID, distinct evaluation IDs.
- Four-process same-row cache write: one sealed shard and one index entry.
- Cache-only stale request: rejected as `CacheIncompatible` without model construction.
- Variable context coverage: instance denominators remained distinct and instance IDs survived into task views.

## Memory-bug flags

- `[MEMORY-BUG-RISK]` The real Qwen adapter now transfers the frozen model and input tensors to CUDA when available, but this host exposed no real Qwen/GPU fixture. The integration mock and CPU suite pass; a hardware smoke test remains advisable before a long extraction.
