Task: Produce two Markdown deliverables that document and improve the `BacktestResults` class.

Sources to inspect:
- All test output files under the repository `data/selection3/test-4` directory (use representative files).
- Backtrader official documentation: https://www.backtrader.com/docu/
- Local implementation: `Backtester/BacktestResults.py` (and any related modules in `Backtester/`).

Deliverables (place in `docs/`):
A) `BacktestResults_UserGuide.md`
   - Title and one-paragraph summary.
   - Quick start (3–5 lines) showing how to load a backtest result and create a basic report.
   - API reference (public methods and properties; inputs/outputs for each).
   - Three worked examples:
       1. Generate a P&L/time-series chart from a single backtest result.
       2. Build a comparative report across multiple runs (aggregated metrics).
       3. Export summary stats to CSV/JSON.
   - Common pitfalls and troubleshooting (file-format mismatches, missing dependencies).
   - Minimal reproducibility checklist (commands to run, required files).
   - One small code snippet that runs with existing repo dependencies; if not feasible, clearly state why and provide a minimal mock example.

Verification:
- Include a short checklist at the end of each doc that can be ticked once examples run successfully.
- If examples cannot be executed due to missing data or dependencies, list the exact missing items and suggested next actions.

Assumptions and constraints:
- Assume Backtrader version compatible with repo's `requirements.txt`; if version mismatches appear, note them.
- If `Backtester/BacktestResults.py` is spread across multiple files or lacks docstrings, infer reasonable method signatures and document assumptions.

Output format: Markdown files in `docs/` with the filenames above. Use headings, short code blocks, and numbered lists where appropriate.