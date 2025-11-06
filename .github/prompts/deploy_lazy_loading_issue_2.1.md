## Deploy Prompt: Issue 2.1 — Lazy Loading of Datasets

Document Version: 1.0
Date: 2025-10-21
Owner: Backtester / Data Engineering

Purpose
-------
This prompt describes the work to implement, test, document, and deploy the "Lazy Loading of Datasets" improvement (BacktestResults backlog item 2.1). Use this file as the authoritative checklist and developer guide for the feature's implementation PRs and deployment.

Background
----------
Currently `Backtester/BacktestResults.py` eagerly loads all dataset files during `TestResults` initialization. For `data/selection3/test-4` this leads to 60 datasets × ~50MB JSON each → multi-GB RAM usage and slow startup. Lazy loading will improve memory usage and responsiveness by loading dataset files only when accessed and evicting least-recently-used items from an in-memory cache.

Goal
----
- Add a safe, backward-compatible lazy-loading mode to `TestResults` and `StrategyResults`.
- Provide an LRU cache with configurable size and context-manager-based automatic cleanup.
- Add unit tests, docs, and a small benchmark verifying memory and performance improvements.

Scope
-----
In-scope:
- Implement `lazy: bool` and `cache_size: int` parameters for `TestResults.__init__`.
- Implement `LazyDict` (or equivalent) loader used for `datasets` and `strategies` access.
- Add `__enter__` / `__exit__` to `TestResults` for resource cleanup.
- Add unit tests and a small benchmark script in `tests/`.
- Add usage examples and brief documentation in `docs/Results/`.

Out-of-scope:
- Changing the on-disk results format (JSON -> Parquet).
- Implementing thread-safe caches for concurrent access (can be follow-up P2).

Design Summary
--------------
- `TestResults(test_path, lazy=False, cache_size=10)` — default remains eager to preserve behavior.
- When `lazy=True`:
  - Load only lightweight metadata (test settings, manifest of datasets and strategies).
  - Use `LazyDict` to map names -> loader callback. `__getitem__` loads on first access.
  - Maintain `_datasets_cache` dict + `_cache_order` list (LRU). When cache size exceeded, evict oldest.
  - Provide `invalidate_cache()` and explicit `preload(datasets=...)` helpers.
  - Implement `__enter__`/`__exit__` to clear caches automatically.

API Contract (short)
-------------------
- Inputs: test_path (str|Path), lazy (bool), cache_size (int >=1)
- Outputs: same public API as before (properties/methods) with identical return types
- Error modes: KeyError when datasets/strategies not found; ValueError for invalid params

Acceptance Criteria
-------------------
- [ ] `TestResults(test_path, lazy=True)` initializes quickly and only reads metadata files.
- [ ] Accessing `test.datasets['dataset_1']` loads only that dataset into memory.
- [ ] `cache_size` limits in-memory datasets; adding (cache_size + 1)-th dataset evicts LRU item.
- [ ] `with TestResults(..., lazy=True) as test:` clears caches on exit.
- [ ] Backward compatibility: `TestResults(..., lazy=False)` behaves exactly as before.
- [ ] Unit tests cover lazy init, load-on-access, eviction, context cleanup, and backward compatibility.
- [ ] Benchmark showing memory reduction (example target: <100MB for test-4 summary operations) and faster summary startup.

Detailed Tasks
--------------
1. Code changes (Primary)
   - File: `Backtester/BacktestResults.py`
     - Add `lazy` and `cache_size` params to `TestResults.__init__`.
     - Factor metadata loading into `_load_metadata()`.
     - Implement `LazyDict` class (internal helper) and use for `self._datasets` & `self._strategies` when lazy.
     - Implement `_load_single_dataset(name)` using existing file parsers and add LRU caching logic.
     - Implement `__enter__` and `__exit__` for automatic cleanup.
     - Add `invalidate_cache()` and `preload(datasets=[])` convenience methods.

2. Tests
   - Create `tests/test_backtest_results_lazy.py` with the following tests:
     - test_lazy_initialization_is_lightweight
     - test_load_single_dataset_on_access
     - test_lru_eviction_behavior
     - test_context_manager_clears_cache
     - test_backwards_compatibility_eager_mode
   - Add small fixtures (synthetic mini-test with 3 small dataset files) under `tests/fixtures/`.

3. Documentation
   - Update `docs/Results/BacktestResults_UserGuide.md` with a "Lazy Loading" subsection and examples.
   - Add a small `docs/Results/BacktestResults_Lazy_Example.md` with usage and memory benchmark results.

4. Benchmark script
   - Add `tools/benchmarks/lazy_memory_benchmark.py` that:
     - Measures process memory (use `tracemalloc` and `psutil` if available).
     - Times eager vs lazy initialization for `data/selection3/test-4` summary operations.
     - Prints a small report.

5. CI / Dev dependency updates
   - Add `pytest` to `requirements-dev.txt` (or `requirements.txt` temporarily) if not present.
   - Add a GitHub Actions job (optional follow-up) to run the lazy feature tests.

Implementation Notes & Constraints
--------------------------------
- Keep public method signatures and return types unchanged.
- Use simple LRU structures (list for order + dict for cache) to avoid heavy dependencies.
- If `psutil`/`tracemalloc` not available on CI, benchmark step should skip gracefully.
- Avoid parallel/threaded loading in this initial change.

PowerShell Commands (developer workstation)
-----------------------------------------
Before running any Python command, activate the project virtualenv in PowerShell:

```powershell
& .venv\Scripts\Activate.ps1
```

Run tests for the new module:

```powershell
pytest -q tests/test_backtest_results_lazy.py
```

Run the benchmark (if psutil installed):

```powershell
python tools/benchmarks/lazy_memory_benchmark.py --test-path data/selection3/test-4
```

Quick local smoke test (interactive):

```powershell
python - <<'PY'
from Backtester.BacktestResults import TestResults
with TestResults('data/selection3/test-4', lazy=True, cache_size=3) as t:
    print('Strategies:', list(t.list_strategies())[:5])
    # Load single dataset and show memory footprint hint
    ds = t.strategies[next(iter(t.list_strategies()))].list_datasets()[0]
    _ = t.strategies[next(iter(t.list_strategies()))].datasets[ds]
    print('Loaded dataset:', ds)
PY
```

Testing & Acceptance
--------------------
- Unit tests must pass locally and in CI for Python 3.8+.
- Tests should include synthetic small fixtures to make them fast (<30s overall).
- Benchmark reports must show meaningful memory reductions for `test-4` summary case.

PR Template / Checklist (copy into PR description)
-------------------------------------------------
- Short summary of change: Implement lazy loading for datasets in `TestResults`.
- Files changed:
  - `Backtester/BacktestResults.py` (core changes)
  - `tests/test_backtest_results_lazy.py` (+fixtures)
  - `docs/Results/BacktestResults_UserGuide.md` (docs)
  - `tools/benchmarks/lazy_memory_benchmark.py` (benchmark)
- Checklist:
  - [ ] Code follows repo style (flake8/black recommended)
  - [ ] Unit tests added and passing
  - [ ] Documentation updated with usage examples
  - [ ] Benchmark script added and results attached to PR if possible
  - [ ] Backward compatibility verified (`lazy=False` behavior unchanged)
  - [ ] Reviewer(s): @backend-lead, @data-eng

Rollout Plan
------------
1. Merge PR to a feature branch and run tests in CI.
2. Run benchmark on a large machine (optional) and attach results.
3. Merge to `develop` / mainline after approvals.
4. Announce to team; include usage notes and cache_size guidance.

Rollback Plan
-------------
- If runtime exceptions occur in production, revert the PR.
- Alternatively, provide a hotfix to revert to `lazy=False` default if immediate rollback not possible.

Risks & Mitigations
-------------------
- Risk: Subtle behavioral change when datasets are mutated by external code.
  - Mitigation: Document cache semantics and provide `invalidate_cache()` method.
- Risk: Cache thrashing with small cache_size.
  - Mitigation: Add warnings when `cache_size < 3` and include guidance in docs.
- Risk: Some callers depend on eager-loading side-effects.
  - Mitigation: Keep `lazy=False` as default and run integration tests.

Estimated Effort & Priority
---------------------------
- Estimated effort: 3 days (High priority, P0)
- Owner: data-eng / backtester maintainer

Notes / References
------------------
- Related backlog: `docs/Results/BacktestResults_Improvements.md` (section 2.1)
- Backtrader docs for analyzers: https://www.backtrader.com/docu/

Contact
-------
If anything is unclear, leave a question on the PR or contact the owner in the repo’s team channel.
