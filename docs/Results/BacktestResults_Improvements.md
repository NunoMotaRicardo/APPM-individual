# BacktestResults Class - Improvements Backlog

**Document Version:** 1.0  
**Date:** October 21, 2025  
**Module:** `Backtester/BacktestResults.py`  
**Current Lines of Code:** 554

---

## Executive Summary

### Current Limitations

1. **Memory Efficiency**: Loading 60 datasets (each 50MB+ JSON) simultaneously can consume 3GB+ RAM; no streaming/lazy-loading support
2. **API Ergonomics**: No chainable methods, limited filtering options, verbose data access patterns requiring multiple nested calls
3. **Serialization & Export**: No built-in export to common formats (Parquet, HDF5, Excel); JSON reload is slow for large results

### Quick Stats

- **Total Feature Items**: 24
- **Quick Wins** (< 1 day): 8 items
- **Medium-term** (1-3 days): 11 items  
- **Long-term** (> 3 days): 5 items
- **P0 (Critical)**: 3 items
- **P1 (High)**: 12 items
- **P2 (Nice-to-have)**: 9 items

---

## Feature Backlog

### Category 1: API Ergonomics & Usability

#### 1.1 Chainable/Fluent API Methods

**Priority:** P1 | **Effort:** Medium (2 days)

**Problem Statement:**  
Current API requires multiple nested calls and temporary variables:
```python
# Current verbose approach
test = TestResults('data/test-4')
strategy = test.strategies['GEM1']
dataset = strategy.datasets['dataset_1']
returns = dataset.get_returns()
```

**Proposed Design:**  
Add fluent interface with method chaining:
```python
# Proposed fluent API
test = TestResults('data/test-4')
returns = (test
    .strategy('GEM1')
    .dataset('dataset_1')
    .get_returns())

# Or with filtering
all_gem_returns = (test
    .strategies(filter=lambda name: name.startswith('GEM'))
    .get_datasets_returns()
    .aggregate('mean'))
```

**Acceptance Criteria:**
- [ ] Add `.strategy(name)` method to `TestResults` returning `StrategyResults`
- [ ] Add `.dataset(name)` method to `StrategyResults` returning `DatasetResults`
- [ ] All filter methods accept callable predicates
- [ ] Methods return `self` where appropriate for chaining
- [ ] Unit tests cover chaining scenarios
- [ ] Documentation with 3+ examples
- [ ] Backward compatibility maintained

---

#### 1.2 Context Manager for Large Test Loading

**Priority:** P0 | **Effort:** Low (0.5 days)

**Problem Statement:**  
Loading large test results doesn't clean up memory properly; users must manually manage resources.

**Proposed Design:**
```python
# Proposed context manager
with TestResults.load('data/test-4', lazy=True) as test:
    for strategy_name in test.list_strategies():
        perf = test.strategies[strategy_name].get_performance()
        # Process...
# Memory automatically released here
```

**Acceptance Criteria:**
- [ ] Implement `__enter__` and `__exit__` methods
- [ ] Add `lazy` loading option (loads strategies on-demand)
- [ ] Proper cleanup of loaded DataFrames
- [ ] Works with nested context managers
- [ ] Unit tests verify memory cleanup
- [ ] Documentation example added

---

#### 1.3 Smart Index/Filter Access

**Priority:** P1 | **Effort:** Low (1 day)

**Problem Statement:**  
No way to filter datasets/strategies by pattern or criteria without manual loops.

**Proposed Design:**
```python
# Proposed smart filtering
test = TestResults('data/test-4')

# Filter by regex pattern
gem_strategies = test.get_strategies(pattern='GEM[124]')

# Filter by performance metric
best_sharpe = test.get_strategies(
    filter={'sharpe_ratio__gt': 1.5},
    aggregator='mean'
)

# Get top N datasets by return
top_datasets = test.strategies['GEM1'].get_datasets(
    sort_by='annual_return',
    ascending=False,
    limit=10
)
```

**Acceptance Criteria:**
- [ ] Add `pattern` parameter accepting regex
- [ ] Add `filter` parameter with dict-based queries
- [ ] Add `sort_by`, `ascending`, `limit` parameters
- [ ] Support common operators: `__gt`, `__lt`, `__gte`, `__lte`, `__eq`
- [ ] Unit tests for filter combinations
- [ ] Performance benchmark (should be <100ms for 60 datasets)

---

#### 1.4 Batch Export Methods

**Priority:** P1 | **Effort:** Medium (1.5 days)

**Problem Statement:**  
No built-in way to export all results to files; users write custom export loops.

**Proposed Design:**
```python
# Proposed batch export
test = TestResults('data/test-4')

# Export all strategy performance to single Excel file
test.export_performance(
    'output/test-4-performance.xlsx',
    format='excel',
    sheets={
        'summary': 'strategies_bt_performance',
        'detailed': 'strategies_stats',
        'by_dataset': 'datasets_performance'
    }
)

# Export all returns to Parquet (memory-efficient)
test.export_returns(
    'output/returns/',
    format='parquet',
    partition_by='strategy'  # Creates one file per strategy
)
```

**Acceptance Criteria:**
- [ ] Support formats: CSV, Excel, Parquet, HDF5
- [ ] Export methods: `export_performance()`, `export_returns()`, `export_weights()`, `export_orders()`
- [ ] Handle large files (streaming for >100MB)
- [ ] Progress callback for long exports
- [ ] Auto-create output directories
- [ ] Unit tests with sample data
- [ ] Documentation with format comparison table

---

#### 1.5 Property-Based Access

**Priority:** P2 | **Effort:** Low (0.5 days)

**Problem Statement:**  
Inconsistent access patterns (`test.name` vs `test.list_strategies()`).

**Proposed Design:**
```python
# Current mixed approach
print(test.name)  # Property
strategies = test.list_strategies()  # Method

# Proposed consistent properties
print(test.name)  # Property
print(test.strategies_list)  # Property
print(test.datasets_list)  # Property
print(test.num_strategies)  # Property
print(test.num_datasets)  # Property
```

**Acceptance Criteria:**
- [ ] Add `@property` decorators for common accessors
- [ ] Maintain backward compatibility (keep old methods)
- [ ] Add deprecation warnings to old methods
- [ ] Update all documentation examples
- [ ] Unit tests verify both access patterns work

---

### Category 2: Performance & Memory Optimization

#### 2.1 Lazy Loading of Datasets

**Priority:** P0 | **Effort:** High (3 days)

**Problem Statement:**  
`TestResults.__init__` loads all 60 datasets (60MB each) = 3.6GB RAM immediately, even if user only needs summary stats.

**Proposed Design:**
```python
# Add lazy loading with caching
class TestResults:
    def __init__(self, test_path, lazy=False):
        # ...
        if lazy:
            self._datasets_loaded = False
            self._datasets_cache = {}
        else:
            self._load_datasets()  # Current behavior
    
    @property
    def datasets(self):
        if self._lazy and not self._datasets_loaded:
            self._load_datasets()
        return self._datasets
    
    def _load_single_dataset(self, name):
        if name in self._datasets_cache:
            return self._datasets_cache[name]
        # Load from file...
        self._datasets_cache[name] = dataset
        return dataset
```

**Acceptance Criteria:**
- [ ] `lazy=True` parameter in `TestResults.__init__`
- [ ] Datasets loaded on first access
- [ ] LRU cache for most-recently-used datasets (configurable size)
- [ ] Memory usage reduced by 90% for summary operations
- [ ] Benchmark showing <500MB RAM for 60 datasets with lazy=True
- [ ] Documentation with memory comparison table
- [ ] Unit tests verify lazy behavior

---

#### 2.2 Incremental DataFrame Construction

**Priority:** P1 | **Effort:** Medium (2 days)

**Problem Statement:**  
Methods like `get_datasets_returns()` use repeated `pd.concat()` in loop, which is O(n²) for large n.

**Proposed Design:**
```python
# Current slow approach
def get_datasets_returns(self, dataset_names=[]):
    returns = pd.DataFrame()
    for dataset_name in dataset_names:  # 60 iterations
        # ...
        returns = pd.concat([returns, returns_df], axis=1)  # Creates new DF each time
    return returns

# Proposed efficient approach
def get_datasets_returns(self, dataset_names=[]):
    returns_list = []
    for dataset_name in dataset_names:
        # ...
        returns_list.append(returns_df)
    return pd.concat(returns_list, axis=1)  # Single concat
```

**Acceptance Criteria:**
- [ ] Refactor all methods using repeated `pd.concat()`
- [ ] Benchmark showing 5-10x speedup for 60 datasets
- [ ] No changes to return types or API
- [ ] Unit tests verify identical output
- [ ] Performance regression tests added

---

#### 2.3 Parallel Loading for Multi-Strategy Results

**Priority:** P2 | **Effort:** Medium (2 days)

**Problem Statement:**  
Loading 4 strategy results (each 50MB JSON) is serial; could use multiprocessing.

**Proposed Design:**
```python
from concurrent.futures import ThreadPoolExecutor

class TestResults:
    def _load_strategies(self, parallel=False):
        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._load_single_strategy, f): f
                    for f in strategy_files
                }
                for future in as_completed(futures):
                    strategy = future.result()
                    self.strategies[strategy.name] = strategy
        else:
            # Current serial loading...
```

**Acceptance Criteria:**
- [ ] `parallel=False` parameter in `TestResults.__init__`
- [ ] ThreadPoolExecutor for I/O-bound JSON loading
- [ ] Configurable `max_workers`
- [ ] Benchmark showing 2-3x speedup for 4+ strategies
- [ ] Graceful fallback if thread pool fails
- [ ] Unit tests with mocked file loading

---

#### 2.4 Cached Property Decorators

**Priority:** P2 | **Effort:** Low (1 day)

**Problem Statement:**  
Repeated calls to `get_performance()` recompute same statistics each time.

**Proposed Design:**
```python
from functools import lru_cache

class StrategyResults:
    @lru_cache(maxsize=None)
    def get_performance(self):
        stats = self._compute_performance()
        # ...
        return stats.to_frame().T
    
    def invalidate_cache(self):
        \"\"\"Call this if returns data changes\"\"\"
        self.get_performance.cache_clear()
```

**Acceptance Criteria:**
- [ ] Add `@lru_cache` to all expensive computed properties
- [ ] Add `invalidate_cache()` methods
- [ ] Benchmark showing 100x speedup for repeated access
- [ ] Unit tests verify cache invalidation
- [ ] Documentation notes about caching behavior

---

### Category 3: Serialization & Data Formats

#### 3.1 Native Parquet Support

**Priority:** P1 | **Effort:** Medium (2 days)

**Problem Statement:**  
JSON format is slow to parse and large (50MB+ per strategy). Parquet would be 10x faster and 5x smaller.

**Proposed Design:**
```python
# Save results as Parquet during backtest
class BacktestFramework:
    def save_results(self, results, path, format='json'):
        if format == 'parquet':
            self._save_as_parquet(results, path)
        else:
            self._save_as_json(results, path)  # Current

# Load from Parquet
test = TestResults('data/test-4', format='auto')  # Detects parquet vs json
```

**Acceptance Criteria:**
- [ ] Add Parquet save/load in `BacktestFramework`
- [ ] `TestResults` auto-detects format from file extension
- [ ] Benchmark: Parquet load 10x faster than JSON
- [ ] Benchmark: Parquet files 5x smaller than JSON
- [ ] Handle nested structures (weights history, orders)
- [ ] Optional PyArrow dependency
- [ ] Unit tests for round-trip save/load

---

#### 3.2 Database Backend Support

**Priority:** P2 | **Effort:** High (3 days)

**Problem Statement:**  
No way to query backtest results across multiple tests without loading all into memory.

**Proposed Design:**
```python
# Save to SQLite/PostgreSQL
from Backtester.BacktestResults import ResultsDatabase

db = ResultsDatabase('sqlite:///backtests.db')
db.save_test(TestResults('data/test-1'))
db.save_test(TestResults('data/test-2'))
db.save_test(TestResults('data/test-3'))

# Query across all tests
query = \"\"\"
    SELECT strategy_name, AVG(sharpe_ratio) as avg_sharpe
    FROM strategy_performance
    WHERE test_date > '2025-01-01'
    GROUP BY strategy_name
    ORDER BY avg_sharpe DESC
\"\"\"
results = db.query(query)
```

**Acceptance Criteria:**
- [ ] New `ResultsDatabase` class
- [ ] Support SQLite and PostgreSQL
- [ ] Schema for tests, strategies, datasets, performance
- [ ] Save/load methods preserving all data
- [ ] Query builder helper methods
- [ ] Migration scripts for schema changes
- [ ] Unit tests with in-memory SQLite
- [ ] Documentation with query examples

---

#### 3.3 HDF5 Time-Series Store

**Priority:** P2 | **Effort:** Medium (2 days)

**Problem Statement:**  
Pandas `HDFStore` is optimal for large time-series (returns, weights), but not used.

**Proposed Design:**
```python
import pandas as pd

# Save to HDF5
test = TestResults('data/test-4')
with pd.HDFStore('test-4.h5') as store:
    for strategy_name, strategy in test.strategies.items():
        store[f'{strategy_name}/returns'] = strategy.get_returns()
        store[f'{strategy_name}/performance'] = strategy.get_performance()

# Fast querying
with pd.HDFStore('test-4.h5') as store:
    # Load only specific date range
    returns = store.select('GEM1/returns', where='date > "2020-01-01"')
```

**Acceptance Criteria:**
- [ ] Add `.to_hdf()` method to `TestResults`, `StrategyResults`
- [ ] Support date-range queries
- [ ] Handle multi-index DataFrames
- [ ] Compression support
- [ ] Benchmark: 20x faster than JSON for time-series queries
- [ ] Unit tests for save/load/query
- [ ] Documentation with query examples

---

### Category 4: Plotting & Visualization

#### 4.1 Built-in Plotting Methods

**Priority:** P1 | **Effort:** Medium (2 days)

**Problem Statement:**  
No built-in plotting; users must write matplotlib code manually.

**Proposed Design:**
```python
# Proposed plotting API
test = TestResults('data/test-4')

# Automatic P&L chart
test.strategies['GEM1'].plot(
    figsize=(12, 6),
    show_drawdown=True,
    save_as='GEM1_performance.png'
)

# Compare multiple strategies
test.plot_comparison(
    strategies=['GEM1', 'GEM2', 'HRPP'],
    metric='cumulative_returns',
    aggregator='mean'
)

# Correlation matrix heatmap
test.plot_correlation_matrix(
    save_as='strategy_correlation.png'
)
```

**Acceptance Criteria:**
- [ ] Add `.plot()` method to `StrategyResults`
- [ ] Add `.plot_comparison()` to `TestResults`
- [ ] Support metrics: returns, cumulative_returns, drawdown, weights
- [ ] Configurable style (seaborn, ggplot, etc.)
- [ ] Save to file support (PNG, PDF, SVG)
- [ ] Unit tests verify plot creation
- [ ] Documentation with gallery of examples

---

#### 4.2 Interactive Plotly Dashboards

**Priority:** P2 | **Effort:** Medium (2 days)

**Problem Statement:**  
Static matplotlib plots lack interactivity for exploring 60 datasets.

**Proposed Design:**
```python
# Proposed interactive dashboard
test = TestResults('data/test-4')

# Launch interactive dashboard
test.dashboard(
    port=8050,
    metrics=['sharpe_ratio', 'max_drawdown', 'annual_return']
)
# Opens browser at http://localhost:8050 with:
# - Strategy selector dropdown
# - Dataset multi-select
# - Metric comparison charts
# - Hover tooltips with details
```

**Acceptance Criteria:**
- [ ] Plotly Dash backend
- [ ] Interactive strategy/dataset selection
- [ ] Hover tooltips with detailed stats
- [ ] Export charts as PNG from browser
- [ ] Responsive layout for mobile
- [ ] Optional dependency (`pip install backtest-results[dash]`)
- [ ] Documentation with screenshot examples

---

#### 4.3 PyFolio Integration

**Priority:** P1 | **Effort:** Medium (1.5 days)

**Problem Statement:**  
PyFolio provides advanced tearsheets, but requires manual data formatting.

**Proposed Design:**
```python
# Proposed PyFolio integration
test = TestResults('data/test-4')
strategy = test.strategies['GEM1']

# Automatic PyFolio format conversion
returns, positions, transactions = strategy.to_pyfolio()

# Direct tearsheet generation
strategy.create_pyfolio_tearsheet(
    live_start_date='2020-01-01',
    save_as='GEM1_tearsheet.pdf'
)
```

**Acceptance Criteria:**
- [ ] Add `.to_pyfolio()` method returning (returns, positions, transactions)
- [ ] Map order history to PyFolio transaction format
- [ ] Map weights history to positions format
- [ ] Add `.create_pyfolio_tearsheet()` convenience method
- [ ] Handle missing data gracefully
- [ ] Optional dependency (`pip install backtest-results[pyfolio]`)
- [ ] Unit tests verify format compatibility
- [ ] Documentation with example tearsheet

---

### Category 5: Data Validation & Quality

#### 5.1 Schema Validation on Load

**Priority:** P1 | **Effort:** Low (1 day)

**Problem Statement:**  
No validation of JSON structure; corrupt files cause cryptic errors deep in code.

**Proposed Design:**
```python
from pydantic import BaseModel, ValidationError

class TestSettingsSchema(BaseModel):
    universe_name: str
    backtest_duration: int
    lookback_periods: int
    num_datasets: int
    # ...

class TestResults:
    def _load_json(self, path, schema=None):
        data = json.load(open(path))
        if schema:
            try:
                validated = schema(**data)
                return validated.dict()
            except ValidationError as e:
                raise ValueError(f\"Invalid schema in {path}: {e}\")
        return data
```

**Acceptance Criteria:**
- [ ] Pydantic schemas for test_settings, datasets_info, strategy results
- [ ] Validate on load with clear error messages
- [ ] Optional `strict=False` parameter to skip validation
- [ ] Unit tests with invalid JSON files
- [ ] Documentation listing required fields

---

#### 5.2 Data Integrity Checks

**Priority:** P1 | **Effort:** Medium (1.5 days)

**Problem Statement:**  
No checks for data consistency (e.g., return dates matching dataset dates).

**Proposed Design:**
```python
# Proposed validation API
test = TestResults('data/test-4')

# Run integrity checks
report = test.validate(checks=[
    'dates_sequential',
    'no_missing_returns',
    'weights_sum_to_one',
    'final_value_matches_returns',
    'all_orders_executed'
])

print(report.summary())
# Issues found:
# - Strategy GEM1, dataset_5: weights sum to 0.98 at date 2020-05-01
# - Strategy HRPP, dataset_12: missing return at date 2021-03-15
```

**Acceptance Criteria:**
- [ ] `validate()` method with pluggable checks
- [ ] Built-in checks: dates, weights, returns, final_value, orders
- [ ] Detailed report with affected datasets
- [ ] Warning vs. error severity levels
- [ ] Unit tests for each check type
- [ ] Documentation explaining each check

---

#### 5.3 Missing Data Reports

**Priority:** P2 | **Effort:** Low (1 day)

**Problem Statement:**  
No easy way to see which strategies/datasets have incomplete data.

**Proposed Design:**
```python
# Proposed missing data report
test = TestResults('data/test-4')

missing = test.report_missing_data()
print(missing)
# Output:
# Strategy        Dataset      Missing Returns  Missing Weights  Missing Orders
# GEM1            dataset_5              2             0              0
# GEM2            dataset_12             0             1              3
# HRPP            -                      0             0              0
```

**Acceptance Criteria:**
- [ ] `report_missing_data()` returns DataFrame summary
- [ ] Count missing values per metric per dataset
- [ ] Option to export report as CSV
- [ ] Unit tests with datasets having missing data

---

### Category 6: Testing & Documentation

#### 6.1 Comprehensive Unit Tests

**Priority:** P0 | **Effort:** Medium (2 days)

**Problem Statement:**  
No unit tests for `BacktestResults.py`; changes risk breaking existing code.

**Proposed Design:**
```python
# tests/test_backtest_results.py
import pytest
from Backtester.BacktestResults import TestResults, StrategyResults, DatasetResults

class TestTestResults:
    @pytest.fixture
    def sample_test(self):
        return TestResults('tests/fixtures/sample_test')
    
    def test_load_test_settings(self, sample_test):
        assert sample_test.name == 'sample_test'
        assert sample_test.num_datasets == 10
    
    def test_list_strategies(self, sample_test):
        strategies = sample_test.list_strategies()
        assert len(strategies) > 0
        assert 'GEM1' in strategies
    
    # ... 50+ more tests
```

**Acceptance Criteria:**
- [ ] 80%+ code coverage for all three classes
- [ ] Test fixtures with small sample data
- [ ] Tests for happy path and error cases
- [ ] Parametrized tests for aggregator functions
- [ ] CI/CD integration (GitHub Actions)
- [ ] Coverage report badge in README

---

#### 6.2 Jupyter Notebook Examples

**Priority:** P1 | **Effort:** Low (1 day)

**Problem Statement:**  
Documentation has code examples but no runnable notebooks.

**Proposed Design:**
Create `examples/` folder with notebooks:
- `01_quick_start.ipynb` - Load and explore results
- `02_performance_comparison.ipynb` - Compare strategies
- `03_advanced_filtering.ipynb` - Complex queries
- `04_exporting_results.ipynb` - Export to various formats
- `05_custom_analysis.ipynb` - Advanced use cases

**Acceptance Criteria:**
- [ ] 5+ notebooks covering common use cases
- [ ] Notebooks run end-to-end without errors
- [ ] Use real sample data from `data/selection3/test-4/`
- [ ] Include explanatory markdown cells
- [ ] Output cells showing results
- [ ] Link from main documentation

---

#### 6.3 API Reference Auto-Generation

**Priority:** P2 | **Effort:** Low (0.5 days)

**Problem Statement:**  
Documentation manually maintained; often out-of-sync with code.

**Proposed Design:**
```python
# Add docstrings to all methods
class TestResults:
    def get_strategies_bt_performance(
        self, 
        strategy_names: list = [],
        aggregator: str = \"mean\"
    ) -> pd.DataFrame:
        \"\"\"Get aggregated Backtrader performance metrics across datasets.
        
        Args:
            strategy_names: List of strategy names to include. Empty = all.
            aggregator: Aggregation method ('mean', 'median', 'max', 'min')
        
        Returns:
            DataFrame with strategies as rows, metrics as columns
        
        Raises:
            ValueError: If aggregator is not recognized
        
        Example:
            >>> test = TestResults('data/test-4')
            >>> perf = test.get_strategies_bt_performance(aggregator='median')
            >>> print(perf['sharpe_ratio'])
        \"\"\"
        # ...
```

Then use Sphinx autodoc:
```bash
\$ sphinx-apidoc -o docs/api Backtester/
\$ sphinx-build docs/ docs/_build/html
```

**Acceptance Criteria:**
- [ ] Google-style docstrings for all public methods
- [ ] Sphinx configuration
- [ ] Auto-generated HTML docs
- [ ] Host on ReadTheDocs or GitHub Pages
- [ ] Examples in docstrings that run via doctest

---

### Category 7: Advanced Features

#### 7.1 Custom Metric Registration

**Priority:** P2 | **Effort:** Medium (2 days)

**Problem Statement:**  
Users want custom performance metrics not in Backtrader analyzers.

**Proposed Design:**
```python
# Proposed custom metric API
from Backtester.BacktestResults import register_metric

@register_metric('ulcer_index')
def compute_ulcer_index(returns: pd.Series) -> float:
    \"\"\"Ulcer Index = sqrt(mean(squared drawdowns))\"\"\"
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    return np.sqrt((drawdowns ** 2).mean())

# Now available in get_performance()
test = TestResults('data/test-4')
perf = test.strategies['GEM1'].get_performance()
print(perf['ulcer_index'])  # Automatically computed
```

**Acceptance Criteria:**
- [ ] Decorator `@register_metric(name)`
- [ ] Metrics computed lazily on first access
- [ ] Clear error if metric computation fails
- [ ] Unit tests with sample custom metrics
- [ ] Documentation with 3+ examples

---

#### 7.2 Benchmark Comparison

**Priority:** P1 | **Effort:** Medium (1.5 days)

**Problem Statement:**  
No built-in way to compare strategy performance vs. benchmark (e.g., SPY).

**Proposed Design:**
```python
# Proposed benchmark API
test = TestResults('data/test-4')

# Load benchmark data
benchmark_returns = pd.read_csv('spy_returns.csv', index_col='date')

# Compare all strategies to benchmark
comparison = test.compare_to_benchmark(
    benchmark_returns,
    metrics=['sharpe_ratio', 'alpha', 'beta', 'information_ratio']
)
print(comparison)
# Strategy  Sharpe (Strategy)  Sharpe (Benchmark)  Alpha    Beta
# GEM1           2.13              0.85            0.08    0.92
# ...
```

**Acceptance Criteria:**
- [ ] `compare_to_benchmark()` method
- [ ] Compute alpha, beta, information ratio vs. benchmark
- [ ] Handle date alignment issues
- [ ] Plot strategy vs. benchmark cumulative returns
- [ ] Unit tests with synthetic benchmark
- [ ] Documentation with real SPY example

---

#### 7.3 Multi-Test Aggregation

**Priority:** P2 | **Effort:** High (3 days)

**Problem Statement:**  
No way to aggregate results across multiple test runs (e.g., test-1, test-2, test-3, test-4).

**Proposed Design:**
```python
# Proposed multi-test API
from Backtester.BacktestResults import TestCollection

collection = TestCollection([
    'data/selection3/test-1',
    'data/selection3/test-2',
    'data/selection3/test-3',
    'data/selection3/test-4'
])

# Aggregate performance across all tests
overall = collection.aggregate_performance(
    by='strategy',
    aggregator='mean',
    confidence_interval=0.95
)
print(overall)
# Strategy  Mean Sharpe  Std Dev  95% CI Lower  95% CI Upper
# GEM1          2.05       0.18      1.71          2.39
```

**Acceptance Criteria:**
- [ ] New `TestCollection` class
- [ ] Load multiple tests efficiently
- [ ] Compute aggregate statistics with confidence intervals
- [ ] Handle missing strategies/datasets gracefully
- [ ] Plot distributions across tests
- [ ] Unit tests with multiple sample tests
- [ ] Documentation with practical example

---

## Quick Wins (< 1 Day)

Prioritized by impact/effort ratio:

1. **Context Manager for Large Test Loading** (P0, 0.5 days) - Immediate memory benefit
2. **Property-Based Access** (P2, 0.5 days) - Better API consistency
3. **API Reference Auto-Generation** (P2, 0.5 days) - Reduces doc maintenance
4. **Smart Index/Filter Access** (P1, 1 day) - Huge usability improvement
5. **Schema Validation on Load** (P1, 1 day) - Prevents frustrating errors
6. **Missing Data Reports** (P2, 1 day) - Simple but useful
7. **Jupyter Notebook Examples** (P1, 1 day) - Helps adoption
8. **Cached Property Decorators** (P2, 1 day) - Free performance boost

**Total Quick Wins:** 6 days of work for 8 major improvements

---

## Medium-Term Items (1-3 Days)

Organized by category priority:

### Must-Have (P0-P1)
1. Chainable/Fluent API Methods (P1, 2 days)
2. Batch Export Methods (P1, 2 days)
3. Lazy Loading of Datasets (P0, 3 days)
4. Incremental DataFrame Construction (P1, 2 days)
5. Native Parquet Support (P1, 2 days)
6. Built-in Plotting Methods (P1, 2 days)
7. PyFolio Integration (P1, 1.5 days)
8. Data Integrity Checks (P1, 1.5 days)
9. Comprehensive Unit Tests (P0, 2 days)
10. Benchmark Comparison (P1, 1.5 days)

### Nice-to-Have (P2)
11. Parallel Loading for Multi-Strategy Results (P2, 2 days)
12. HDF5 Time-Series Store (P2, 2 days)
13. Interactive Plotly Dashboards (P2, 2 days)
14. Custom Metric Registration (P2, 2 days)

**Total Medium-Term:** 28.5 days for all P0-P1 items

---

## Long-Term Design Work (> 3 Days)

These require architectural changes:

1. **Database Backend Support** (P2, 3 days) - Enables cross-test queries
2. **Multi-Test Aggregation** (P2, 3 days) - Statistical analysis across runs
3. **Lazy Loading Refactor** (P0, 3 days) - Foundation for memory efficiency

**Total Long-Term:** 9 days

---

## Sample Implementation: Lazy Loading (Top Priority)

### Design Sketch

```python
# Backtester/BacktestResults.py

class TestResults:
    \"\"\"
    Top-level container for backtest run data.
    
    Args:
        test_path: Path to test directory
        lazy: If True, only load metadata; load datasets/strategies on-demand
        cache_size: Max number of datasets to keep in memory (for lazy mode)
    \"\"\"
    
    def __init__(
        self, 
        test_path: str | Path, 
        lazy: bool = False,
        cache_size: int = 10
    ):
        self.test_path = Path(test_path).resolve()
        self._lazy = lazy
        self._cache_size = cache_size
        
        # Always load metadata (lightweight)
        self.test_settings = self._load_json(
            self.test_path / 'test_settings.json',
            label='test settings'
        )
        self.name = self.test_settings.get('test_name', self.test_path.stem)
        self.num_datasets = int(self.test_settings.get('num_datasets', None))
        # ... other metadata
        
        # Lazy loading setup
        if lazy:
            self._datasets_cache = {}
            self._cache_order = []  # LRU tracking
            self._datasets_manifest = self._build_datasets_manifest()
            self._strategies = LazyDict(self._load_strategy)
        else:
            # Current eager loading
            self._load_datasets()
            self._load_strategies()
    
    def _build_datasets_manifest(self) -> Dict[str, Path]:
        \"\"\"Build lightweight index of dataset names -> file paths\"\"\"
        datasets_info = self._load_json(
            self.test_path / 'datasets_info.json',
            label='datasets info'
        )
        manifest = {}
        for dataset_name in datasets_info.keys():
            dataset_path = self.test_path / 'datasets' / f\"{dataset_name}.csv\"
            if dataset_path.exists():
                manifest[dataset_name] = dataset_path
        return manifest
    
    @property
    def datasets(self):
        \"\"\"Lazy-load datasets on first access\"\"\"
        if self._lazy and not hasattr(self, '_datasets'):
            self._datasets = LazyDict(self._load_single_dataset)
            # Populate with manifest keys
            for name in self._datasets_manifest.keys():
                self._datasets._manifest[name] = None  # Not loaded yet
        return self._datasets
    
    def _load_single_dataset(self, name: str):
        \"\"\"Load a single dataset with LRU caching\"\"\"
        if name in self._datasets_cache:
            # Move to end of cache_order (most recent)
            self._cache_order.remove(name)
            self._cache_order.append(name)
            return self._datasets_cache[name]
        
        # Load from disk
        dataset_path = self._datasets_manifest.get(name)
        if not dataset_path:
            raise KeyError(f\"Dataset {name} not found\")
        
        dataset_data = pd.read_csv(dataset_path, index_col=0, header=[0, 1])
        
        # Add to cache
        self._datasets_cache[name] = dataset_data
        self._cache_order.append(name)
        
        # Evict oldest if cache full
        if len(self._cache_order) > self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._datasets_cache[oldest]
        
        return dataset_data
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        \"\"\"Cleanup: clear caches and loaded data\"\"\"
        if hasattr(self, '_datasets_cache'):
            self._datasets_cache.clear()
        if hasattr(self, '_datasets'):
            self._datasets.clear()
        return False  # Don't suppress exceptions


class LazyDict(dict):
    \"\"\"Dict that loads values on first access using a loader function\"\"\"
    
    def __init__(self, loader_func):
        super().__init__()
        self._loader = loader_func
        self._manifest = {}  # Available keys
    
    def __getitem__(self, key):
        if key not in self._manifest:
            raise KeyError(key)
        
        if not super().__contains__(key):
            # Load on first access
            value = self._loader(key)
            super().__setitem__(key, value)
        
        return super().__getitem__(key)
    
    def keys(self):
        return self._manifest.keys()
    
    def __contains__(self, key):
        return key in self._manifest
```

### Usage Examples

```python
# Example 1: Memory-efficient summary statistics
with TestResults('data/test-4', lazy=True) as test:
    # Only loads metadata (~1 KB)
    print(f\"Test: {test.name}\")
    print(f\"Strategies: {test.list_strategies()}\")
    
    # Loads only strategy JSON files (~200 KB total), not datasets
    perf = test.get_strategies_bt_performance(aggregator='mean')
    print(perf)
# Memory automatically released

# Example 2: Process datasets one at a time
test = TestResults('data/test-4', lazy=True, cache_size=5)
for strategy_name in test.list_strategies():
    strategy = test.strategies[strategy_name]  # Lazy load strategy
    for dataset_name in strategy.list_datasets():
        # Lazy load dataset (only 5 in memory at once)
        returns = strategy.datasets[dataset_name].get_returns()
        # Process returns...
        # Oldest dataset auto-evicted from cache when 6th is loaded

# Example 3: Explicit eager loading when needed
test = TestResults('data/test-4', lazy=False)  # Current behavior
# All data loaded immediately
```

### Acceptance Criteria Checklist

- [ ] `lazy=True` parameter in `TestResults.__init__`
- [ ] `cache_size` parameter controls LRU cache
- [ ] Context manager support (`__enter__`/`__exit__`)
- [ ] `LazyDict` class for on-demand loading
- [ ] Datasets loaded only when accessed
- [ ] Strategies loaded only when accessed
- [ ] LRU eviction when cache full
- [ ] Memory usage: <100MB for lazy mode vs. 3GB+ for eager mode (measured with `tracemalloc`)
- [ ] Backward compatibility: `lazy=False` behaves identically to current code
- [ ] Unit tests:
  - Test lazy loading triggers on first access
  - Test cache eviction with `cache_size=2`
  - Test context manager cleanup
  - Test that metadata always loads
- [ ] Documentation:
  - Update User Guide with lazy loading example
  - Add memory comparison table
  - Explain cache size tuning
- [ ] Benchmark:
  - Measure load time for summary stats: <2 seconds (lazy) vs. 30 seconds (eager)
  - Memory profile showing 95% reduction

### Implementation Steps

1. **Day 1 Morning:** Implement `LazyDict` class with unit tests
2. **Day 1 Afternoon:** Add lazy loading to `_load_datasets()`
3. **Day 2 Morning:** Add LRU cache with eviction logic
4. **Day 2 Afternoon:** Add context manager support
5. **Day 3 Morning:** Add lazy loading to `_load_strategies()`
6. **Day 3 Afternoon:** Write comprehensive tests and documentation

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Lazy loading breaks existing code | Keep `lazy=False` as default; add deprecation warning in v2.0 |
| Cache thrashing with small `cache_size` | Warn if `cache_size < num_accessed_datasets / 2` |
| Thread-safety issues with cache | Use `threading.RLock` for cache operations if multi-threading added |

---

## Verification Checklist

After implementing improvements, verify:

### Quick Wins
- [ ] Context manager works and cleans up memory
- [ ] Property-based access works alongside method calls
- [ ] API docs auto-generate from docstrings
- [ ] Smart filtering returns correct results for patterns
- [ ] Schema validation catches invalid JSON
- [ ] Missing data report shows accurate counts
- [ ] Jupyter notebooks run end-to-end
- [ ] Cached properties speed up repeated calls

### Medium-Term
- [ ] Fluent API chains correctly
- [ ] Batch exports produce valid files in all formats
- [ ] Lazy loading reduces memory by >90%
- [ ] Incremental DataFrame construction 5x faster
- [ ] Parquet files load 10x faster than JSON
- [ ] Built-in plots render correctly
- [ ] PyFolio integration produces valid tearsheet
- [ ] Data integrity checks catch known issues
- [ ] Unit tests achieve >80% coverage
- [ ] Benchmark comparison computes alpha/beta correctly

### Long-Term
- [ ] Database backend stores and retrieves all data
- [ ] Multi-test aggregation computes confidence intervals
- [ ] Lazy loading architecture supports streaming

---

## Missing Dependencies & Next Actions

### Current Missing Items

1. **Test Data:**
   - ✅ Available: `data/selection3/test-4/` with 60 datasets and 4 strategies
   - ✅ File structure validated and complete

2. **Development Environment:**
   - ✅ Python 3.8+ installed
   - ✅ Dependencies in `requirements.txt`
   - ❌ Missing: `pytest` for unit tests (add to `requirements-dev.txt`)
   - ❌ Missing: `sphinx` for docs (add to `requirements-dev.txt`)

3. **Documentation:**
   - ✅ User Guide complete (`BacktestResults_UserGuide.md`)
   - ✅ Implementation docs available
   - ❌ Missing: API reference (needs Sphinx setup)
   - ❌ Missing: Jupyter notebook examples

4. **Testing:**
   - ❌ Missing: Unit test suite (`tests/test_backtest_results.py`)
   - ❌ Missing: Test fixtures (`tests/fixtures/`)
   - ❌ Missing: CI/CD configuration (`.github/workflows/`)

### Suggested Next Actions

#### Immediate (Week 1)
1. Create `requirements-dev.txt` with dev dependencies:
   ```
   pytest>=7.0
   pytest-cov>=4.0
   sphinx>=5.0
   sphinx-rtd-theme>=1.0
   black>=22.0
   flake8>=5.0
   ```

2. Implement P0 items:
   - Context manager (0.5 days)
   - Lazy loading (3 days)
   - Comprehensive unit tests (2 days)

3. Set up CI/CD:
   - GitHub Actions workflow for tests
   - Coverage reporting to Codecov

#### Short-Term (Weeks 2-4)
4. Implement 8 quick wins (6 days total)
5. Create Jupyter notebook examples (1 day)
6. Set up Sphinx docs (0.5 days)

#### Medium-Term (Months 2-3)
7. Implement P1 items from medium-term list (28.5 days)
8. User acceptance testing with real workflows
9. Performance benchmarking and optimization

#### Long-Term (Months 4-6)
10. Implement long-term architectural items (9 days)
11. Database backend pilot
12. Multi-test analysis workflows

---

## Version Compatibility Notes

**Current Implementation:**
- Python: 3.8+ (uses `str | Path` type hints)
- Backtrader: 1.9+ (based on `requirements.txt`)
- Pandas: 1.0+ (uses `pd.to_datetime(..., errors='coerce')`)
- NumPy: 1.18+ (modern array operations)

**Proposed Changes:**
- Add optional dependencies for new features:
  - `pyarrow>=10.0` for Parquet support
  - `plotly>=5.0` for interactive plots
  - `pydantic>=2.0` for schema validation
  - `pyfolio` for tearsheet integration (version TBD, check compatibility)

**Deprecation Plan:**
- v1.x: Add new features with backward compatibility
- v2.0: Make lazy loading default, deprecate old method names
- v3.0: Remove deprecated methods, require Python 3.10+

---

## Summary

This backlog provides **24 feature items** spanning **7 categories**, totaling approximately **60 days of implementation work**. The prioritization focuses on:

1. **Immediate value** (P0): Memory efficiency, testing, stability
2. **High impact** (P1): API ergonomics, performance, usability
3. **Future-proofing** (P2): Advanced features, ecosystem integration

By implementing the **8 quick wins** first (6 days), you can deliver significant improvements to users while building momentum for the larger architectural changes.

The **lazy loading implementation sketch** provides a concrete starting point for the highest-priority item, complete with code, tests, and acceptance criteria.

---

**Document Status:** ✅ Complete  
**Total Feature Items:** 24  
**Total Estimated Effort:** ~60 days  
**Ready for:** Review, prioritization, sprint planning  
**Contact:** See project documentation for questions

