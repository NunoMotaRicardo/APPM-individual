Task: Produce two Markdown deliverables that document and improve the `BacktestResults` class.

Sources to inspect:
- All test output files under the repository `data/selection3/test-4` directory (use representative files).
- Backtrader official documentation: https://www.backtrader.com/docu/
- Local implementation: `Backtester/BacktestResults.py` (and any related modules in `Backtester/`).
- documentation for local implementation: `docs/Results/`

Deliverables (place in `docs/Results/`):
`BacktestResults_Improvements.md`
   - Executive summary of current limitations (1–3 bullets).
   - Feature backlog organized by category (e.g., API ergonomics, serialization, plotting, performance, testing):
       - For each feature:
           * Title
           * Problem statement / motivation
           * Proposed design or change
           * Acceptance criteria (tests, docs, example)
           * Estimated effort: Low / Medium / High
           * Priority: P0 / P1 / P2
   - Suggested quick wins (changes that can be implemented in <1 day).
   - Suggested medium-term items (1–3 days) and long-term design work.
   - One sample implementation sketch for a top-priority improvement (pseudo-code or function signature).

Verification:
- Include a short checklist at the end of each doc that can be ticked once examples run successfully.
- If examples cannot be executed due to missing data or dependencies, list the exact missing items and suggested next actions.

Assumptions and constraints:
- Assume Backtrader version compatible with repo's `requirements.txt`; if version mismatches appear, note them.
- If `Backtester/BacktestResults.py` is spread across multiple files or lacks docstrings, infer reasonable method signatures and document assumptions.

Output format: Markdown files in `docs/` with the filenames above. Use headings, short code blocks, and numbered lists where appropriate.