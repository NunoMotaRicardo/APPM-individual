# BacktestResults User Guide

## Overview

The `BacktestResults` module provides a comprehensive framework for loading, analyzing, and reporting on backtest results from algorithmic trading strategies. The module consists of three main classes that work together to organize and present backtest data in a structured, accessible manner:

- **`TestResults`**: Top-level container for all backtest run data, including test settings, universe configuration, datasets, and strategy results
- **`StrategyResults`**: Manages results for a specific trading strategy across multiple datasets
- **`DatasetResults`**: Handles detailed results for a single dataset within a strategy, including returns, weights, orders, and asset values

This module is designed to work seamlessly with Backtrader-based backtesting frameworks and provides easy access to performance metrics, portfolio analytics, and time-series data for comprehensive strategy evaluation.

---

## Quick Start

### Loading and Basic Reporting (3-5 Lines)

```python
from Backtester.BacktestResults import TestResults

# Load backtest results from a test directory
test = TestResults("data/selection3/test-4")

# Get aggregated performance statistics for all strategies
performance_stats = test.get_strategies_bt_performance(aggregator="mean")
print(performance_stats)
```

---

## API Reference

### Class: `TestResults`

**Constructor:**
```python
TestResults(test_path: str | Path)
```

**Parameters:**
- `test_path`: Path to the test directory containing backtest results

**Attributes:**
- `test_path` (Path): Resolved absolute path to the test directory
- `results_path` (Path): Path to the results subdirectory
- `test_settings` (Dict): Test configuration from `test_settings.json`
- `universe_name` (str): Name of the asset universe
- `name` (str): Test name
- `num_assets` (int): Number of assets in universe
- `lookback_periods` (int): Number of lookback periods
- `duration` (int): Backtest duration
- `random_seed` (int): Random seed for reproducibility
- `num_datasets` (int): Number of datasets generated
- `universe_settings` (Dict): Universe configuration and asset information
- `datasets` (Dict): Dictionary of dataset metadata
- `strategies` (Dict): Dictionary of StrategyResults objects

**Public Methods:**

#### `list_datasets() -> List[str]`
Returns a list of all dataset names.

**Returns:** List of dataset name strings

#### `get_datasets_info(dataset_names: list = []) -> Dict`
Get metadata for specified datasets.

**Parameters:**
- `dataset_names` (list): List of dataset names. If empty, returns all datasets.

**Returns:** Dictionary with dataset metadata (name, path, start_date, end_date, symbols)

#### `get_datasets_data(datasets_filter_list: list = [], column: str = "") -> Dict`
Load actual dataset price/volume data from CSV files.

**Parameters:**
- `datasets_filter_list` (list): List of datasets to load. If empty, loads all.
- `column` (str): Specific column to extract (e.g., "close", "volume"). If empty, returns all columns.

**Returns:** Dictionary with dataset names as keys and DataFrames as values

#### `list_strategies() -> List[str]`
Returns a list of all strategy names.

**Returns:** List of strategy name strings

#### `get_strategies_stats(strategy_names: list = []) -> pd.DataFrame`
Get computed performance statistics for specified strategies.

**Parameters:**
- `strategy_names` (list): List of strategy names. If empty, returns all strategies.

**Returns:** DataFrame with performance metrics (annual_return, sharpe_ratio, CAGR, etc.)

#### `get_strategies_bt_performance(strategy_names: list = [], aggregator: str = "mean") -> pd.DataFrame`
Get aggregated Backtrader performance metrics across datasets.

**Parameters:**
- `strategy_names` (list): List of strategy names. If empty, returns all strategies.
- `aggregator` (str): Aggregation method ("mean", "median", "max", "min")

**Returns:** DataFrame with aggregated performance metrics per strategy

---

### Class: `StrategyResults`

**Constructor:**
```python
StrategyResults(payload: Dict, lookback_periods: int)
```

**Parameters:**
- `payload`: Dictionary containing strategy results data
- `lookback_periods`: Number of lookback periods to skip

**Attributes:**
- `name` (str): Strategy name
- `datasets` (Dict): Dictionary of DatasetResults objects

**Public Methods:**

#### `list_datasets() -> List[str]`
Returns a list of all dataset names for this strategy.

**Returns:** List of dataset name strings

#### `get_datasets_returns(dataset_names: list = []) -> pd.DataFrame`
Get returns time series for specified datasets.

**Parameters:**
- `dataset_names` (list): List of dataset names. If empty, returns all.

**Returns:** DataFrame with returns for each dataset as columns

#### `get_datasets_bt_performance(dataset_names: list = []) -> pd.DataFrame`
Get Backtrader performance metrics for specified datasets.

**Parameters:**
- `dataset_names` (list): List of dataset names. If empty, returns all.

**Returns:** DataFrame with performance metrics for each dataset

#### `get_datasets_performance(dataset_names: list = []) -> pd.DataFrame`
Get computed performance statistics for specified datasets.

**Parameters:**
- `dataset_names` (list): List of dataset names. If empty, returns all.

**Returns:** DataFrame with computed performance metrics

#### `get_datasets_weights(dataset_names: list = []) -> Dict`
Get portfolio weight history for specified datasets.

**Parameters:**
- `dataset_names` (list): List of dataset names. If empty, returns all.

**Returns:** Dictionary with dataset names as keys and weight DataFrames as values

#### `get_datasets_asset_values(dataset_names: list = [], component: str = None) -> Dict`
Get asset value history (prices, positions, values, portfolio).

**Parameters:**
- `dataset_names` (list): List of dataset names. If empty, returns all.
- `component` (str): Specific component to return ('prices', 'positions', 'values', 'portfolio'). If None, returns all.

**Returns:** Dictionary with dataset names as keys and component data as values

#### `get_returns() -> pd.DataFrame`
Get mean returns across all datasets.

**Returns:** DataFrame with averaged returns

#### `get_performance() -> pd.DataFrame`
Get performance statistics computed from averaged returns.

**Returns:** DataFrame with performance metrics

---

### Class: `DatasetResults`

**Constructor:**
```python
DatasetResults(name: str, payload: Dict, lookback_periods: int, strategy_name: str)
```

**Parameters:**
- `name`: Dataset name
- `payload`: Dictionary containing dataset results
- `lookback_periods`: Number of lookback periods to skip
- `strategy_name`: Name of the parent strategy

**Attributes:**
- `name` (str): Dataset name
- `strategy_name` (str): Parent strategy name
- `lookback_periods` (int): Lookback periods
- `performance_bt` (Dict): Backtrader performance metrics
- `annual_returns` (Dict): Annual returns by year
- `final_value` (float): Final portfolio value
- `returns` (pd.DataFrame): Time series of returns
- `weights_history` (pd.DataFrame): Portfolio weight history
- `orders` (pd.DataFrame): Order history
- `asset_values_history` (Dict): Asset value history components

**Public Methods:**

#### `get_performance() -> pd.DataFrame`
Get computed performance statistics.

**Returns:** DataFrame with one row containing performance metrics

#### `get_bt_performance() -> pd.DataFrame`
Get Backtrader performance metrics.

**Returns:** DataFrame with Backtrader metrics plus final_value

#### `get_returns() -> pd.DataFrame`
Get returns time series.

**Returns:** DataFrame with date index and return column

#### `get_weights() -> pd.DataFrame`
Get portfolio weights history.

**Returns:** DataFrame with dates as index and assets as columns

#### `get_orders() -> pd.DataFrame`
Get order execution history.

**Returns:** DataFrame with order details (order_id, created_date, executed_date, asset, size, price, etc.)

#### `get_asset_values(component: str = None) -> Dict | pd.DataFrame`
Get asset value history data.

**Parameters:**
- `component` (str): Specific component ('prices', 'positions', 'values', 'portfolio'). If None, returns all.

**Returns:** 
- If component is None: Dictionary with all components
- If component specified: DataFrame for that component

---

## Worked Examples

### Example 1: Generate P&L/Time-Series Chart from Single Backtest Result

```python
import matplotlib.pyplot as plt
from Backtester.BacktestResults import TestResults

# Load test results
test = TestResults("data/selection3/test-4")

# Get strategy results for a specific strategy
strategy_name = "GEM1"
strategy = test.strategies[strategy_name]

# Get returns for a specific dataset
dataset_name = "dataset_1"
dataset_result = strategy.datasets[dataset_name]
returns = dataset_result.get_returns()

# Calculate cumulative returns
cumulative_returns = (1 + returns['return']).cumprod()

# Plot P&L over time
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns.index, cumulative_returns.values)
plt.title(f'{strategy_name} - {dataset_name}: Cumulative P&L')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'docs/pnl_{strategy_name}_{dataset_name}.png')
plt.show()

# Print summary statistics
performance = dataset_result.get_performance()
print(f"\n{strategy_name} - {dataset_name} Performance:")
print(performance.T)
```

### Example 2: Build Comparative Report Across Multiple Runs (Aggregated Metrics)

```python
import pandas as pd
from Backtester.BacktestResults import TestResults

# Load multiple test results
test_dirs = ["data/selection3/test-1", "data/selection3/test-2", "data/selection3/test-4"]
comparison_data = []

for test_dir in test_dirs:
    try:
        test = TestResults(test_dir)
        
        # Get aggregated performance for all strategies
        performance = test.get_strategies_bt_performance(aggregator="mean")
        performance['test_name'] = test.name
        performance['num_datasets'] = test.num_datasets
        performance['duration'] = test.duration
        
        comparison_data.append(performance)
    except FileNotFoundError:
        print(f"Warning: Test directory {test_dir} not found")
        continue

# Combine all results
if comparison_data:
    comparison_df = pd.concat(comparison_data, axis=0)
    
    # Reorder columns for better readability
    key_cols = ['test_name', 'num_datasets', 'duration']
    metric_cols = [col for col in comparison_df.columns if col not in key_cols]
    comparison_df = comparison_df[key_cols + metric_cols]
    
    # Display comparison
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE COMPARISON ACROSS TESTS")
    print("="*80)
    print(comparison_df.to_string())
    
    # Save to CSV
    comparison_df.to_csv('docs/strategy_comparison.csv')
    print("\nComparison saved to: docs/strategy_comparison.csv")
else:
    print("No valid test results found")
```

### Example 3: Export Summary Stats to CSV/JSON

```python
import json
import pandas as pd
from Backtester.BacktestResults import TestResults

# Load test results
test = TestResults("data/selection3/test-4")

# ========== Export 1: Strategy-level summary to CSV ==========
strategy_stats = test.get_strategies_stats()
strategy_stats.to_csv('docs/strategy_summary_stats.csv')
print(f"Strategy stats exported to: docs/strategy_summary_stats.csv")

# ========== Export 2: Detailed dataset-level metrics to CSV ==========
detailed_results = []

for strategy_name in test.list_strategies():
    strategy = test.strategies[strategy_name]
    
    for dataset_name in strategy.list_datasets():
        dataset = strategy.datasets[dataset_name]
        
        # Get performance metrics
        perf = dataset.get_performance()
        if not perf.empty:
            perf['strategy'] = strategy_name
            perf['dataset'] = dataset_name
            detailed_results.append(perf)

if detailed_results:
    detailed_df = pd.concat(detailed_results, axis=0)
    detailed_df.to_csv('docs/detailed_dataset_results.csv', index=False)
    print(f"Detailed results exported to: docs/detailed_dataset_results.csv")

# ========== Export 3: Test configuration to JSON ==========
config_summary = {
    'test_name': test.name,
    'universe_name': test.universe_name,
    'num_assets': test.num_assets,
    'lookback_periods': test.lookback_periods,
    'duration': test.duration,
    'random_seed': test.random_seed,
    'num_datasets': test.num_datasets,
    'strategies': test.list_strategies(),
    'datasets': test.list_datasets()
}

with open('docs/test_configuration.json', 'w') as f:
    json.dump(config_summary, f, indent=2)
print(f"Configuration exported to: docs/test_configuration.json")

# ========== Export 4: Portfolio weights to CSV (for first dataset) ==========
strategy_name = test.list_strategies()[0]
dataset_name = test.strategies[strategy_name].list_datasets()[0]
weights = test.strategies[strategy_name].datasets[dataset_name].get_weights()

if not weights.empty:
    weights.to_csv(f'docs/weights_{strategy_name}_{dataset_name}.csv')
    print(f"Weights exported to: docs/weights_{strategy_name}_{dataset_name}.csv")

print("\n✓ All exports completed successfully!")
```

---

## Common Pitfalls and Troubleshooting

### 1. File Format Mismatches

**Problem:** `FileNotFoundError` or `KeyError` when loading results

**Causes:**
- Missing required files: `test_settings.json`, `datasets_info.json`, `universe_settings.json`
- Incorrect directory structure
- Strategy result files (`.json`) missing in `results/` folder

**Solutions:**
```python
from pathlib import Path

# Verify directory structure before loading
test_path = Path("data/selection3/test-4")
required_files = [
    test_path / "test_settings.json",
    test_path / "datasets_info.json",
    test_path.parent / "universe_settings.json"
]

for file in required_files:
    if not file.exists():
        print(f"ERROR: Missing required file: {file}")
    else:
        print(f"✓ Found: {file}")
```

### 2. Large Result Files

**Problem:** Result JSON files are too large (>50MB) and cannot be read

**Solution:** The module handles large files gracefully, but if you encounter memory issues:
```python
# Load only specific strategies
test = TestResults("data/selection3/test-4")

# Work with one strategy at a time
for strategy_name in test.list_strategies():
    strategy = test.strategies[strategy_name]
    # Process strategy...
    del strategy  # Free memory
```

### 3. Missing Dependencies

**Problem:** `ImportError` or `ModuleNotFoundError`

**Solution:** Ensure all required packages are installed:
```bash
pip install numpy pandas backtrader matplotlib seaborn plotly
```

Check your current installation:
```python
import sys
print(f"Python version: {sys.version}")

required_modules = ['numpy', 'pandas', 'backtrader', 'matplotlib']
for module in required_modules:
    try:
        __import__(module)
        print(f"✓ {module} installed")
    except ImportError:
        print(f"✗ {module} NOT installed")
```

### 4. Date Index Issues

**Problem:** Returns or weights DataFrames have wrong date ranges

**Cause:** Lookback periods are automatically removed from returns data

**Solution:** Be aware that the first `lookback_periods` bars are removed:
```python
test = TestResults("data/selection3/test-4")
print(f"Lookback periods: {test.lookback_periods}")

# Returns start AFTER lookback period
dataset = test.strategies["GEM1"].datasets["dataset_1"]
returns = dataset.get_returns()
print(f"Returns start date: {returns.index.min()}")
print(f"Returns end date: {returns.index.max()}")
```

### 5. Empty DataFrames

**Problem:** Methods return empty DataFrames

**Causes:**
- No data available for the requested component
- Strategy didn't execute on the dataset
- Aggregation filters out all data

**Solution:** Always check if DataFrames are empty before processing:
```python
performance = strategy.get_performance()
if performance.empty:
    print("Warning: No performance data available")
else:
    print(performance)
```

### 6. Backtrader Version Compatibility

**Problem:** Metrics or results don't match expected Backtrader output

**Solution:** Verify Backtrader version compatibility:
```python
import backtrader as bt
print(f"Backtrader version: {bt.__version__}")

# Expected version from requirements.txt
# backtrader (any recent version compatible with Python 3.8+)
```

---

## Minimal Reproducibility Checklist

Use this checklist to ensure you can successfully load and analyze backtest results:

### Prerequisites
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (see `requirements.txt`)
- [ ] Backtest has been executed and results saved

### File Structure Verification
```bash
# Verify your test directory has this structure:
data/selection3/test-4/
├── test_settings.json          # ✓ Required
├── datasets_info.json          # ✓ Required
├── datasets/                    # ✓ Required
│   ├── dataset_1.csv
│   ├── dataset_2.csv
│   └── ...
└── results/                     # ✓ Required
    ├── EqualWeight_results.json
    ├── GEM1_results.json
    └── ...
data/selection3/
├── universe_settings.json       # ✓ Required (parent directory)
└── universe_info.csv            # ✓ Required (parent directory)
```

### Quick Validation Script
```python
from pathlib import Path
from Backtester.BacktestResults import TestResults

def validate_test_directory(test_path: str):
    """Validate that a test directory has all required files."""
    test_path = Path(test_path)
    issues = []
    
    # Check test directory exists
    if not test_path.exists():
        return f"❌ Test directory does not exist: {test_path}"
    
    # Check required files
    required_files = {
        'test_settings.json': test_path / 'test_settings.json',
        'datasets_info.json': test_path / 'datasets_info.json',
        'datasets folder': test_path / 'datasets',
        'results folder': test_path / 'results',
        'universe_settings.json': test_path.parent / 'universe_settings.json',
        'universe_info.csv': test_path.parent / 'universe_info.csv'
    }
    
    for name, path in required_files.items():
        if not path.exists():
            issues.append(f"❌ Missing: {name} at {path}")
        else:
            print(f"✓ Found: {name}")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(issue)
        return False
    
    # Try loading
    try:
        test = TestResults(test_path)
        print(f"\n✓ Successfully loaded test: {test.name}")
        print(f"  - Strategies: {len(test.strategies)}")
        print(f"  - Datasets: {len(test.datasets)}")
        return True
    except Exception as e:
        print(f"\n❌ Error loading test: {e}")
        return False

# Run validation
result = validate_test_directory("data/selection3/test-4")
```

### Running the Examples
```bash
# 1. Activate your environment (if using virtual environment)
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 2. Navigate to project root
cd c:\my-git\DataScience-novaIMS\APPM-individual

# 3. Run example scripts
python -c "from Backtester.BacktestResults import TestResults; test = TestResults('data/selection3/test-4'); print(test.get_strategies_stats())"
```

---

## Verification Checklist

After following this guide, verify that you can:

- [ ] **Load a test result:** `test = TestResults("data/selection3/test-4")` executes without errors
- [ ] **List strategies:** `test.list_strategies()` returns a list of strategy names
- [ ] **List datasets:** `test.list_datasets()` returns a list of dataset names
- [ ] **Get strategy performance:** `test.get_strategies_bt_performance()` returns a DataFrame
- [ ] **Access individual strategy:** `strategy = test.strategies["GEM1"]` works
- [ ] **Get returns:** `strategy.get_returns()` returns a DataFrame with date index
- [ ] **Get weights:** `strategy.datasets["dataset_1"].get_weights()` returns weight history
- [ ] **Get orders:** `strategy.datasets["dataset_1"].get_orders()` returns order history
- [ ] **Get asset values:** `strategy.datasets["dataset_1"].get_asset_values()` returns price/position/value data
- [ ] **Export to CSV:** All export examples run without errors
- [ ] **Generate plots:** Matplotlib plots are created successfully

---

## Known Limitations

1. **Large Result Files:** Strategy result JSON files can exceed 50MB for strategies with many rebalancing periods and assets
2. **Memory Usage:** Loading many large datasets simultaneously can consume significant memory
3. **Date Handling:** All dates are assumed to be in the format stored in the JSON files; timezone information is not preserved
4. **Aggregation Assumptions:** The `get_strategies_bt_performance` method fills NaN values with 0 before aggregation
5. **Performance Calculations:** Custom performance metrics assume 252 trading days per year

---

## Additional Resources

- **Backtrader Documentation:** https://www.backtrader.com/docu/
- **Project Repository:** See local `Backtester/` folder for source code
- **Related Documentation:**
  - `docs/AAA_Documentation.md` - Adaptive Asset Allocation strategy
  - `docs/DRPP_Documentation.md` - Dual Regime Portfolio Protection strategy
  - `docs/GEM_Strategy_Documentation.md` - Global Equities Momentum strategy
  - `docs/HRPP_Implementation_Summary.md` - Hierarchical Risk Parity Portfolio

---

## Contact and Support

For issues or questions:
1. Check this documentation first
2. Review the source code: `Backtester/BacktestResults.py`
3. Examine example test results: `data/selection3/test-*/`
4. Check existing documentation in `docs/` folder

---

*Last Updated: October 2025*
*Compatible with: Python 3.8+, Backtrader 1.9+, Pandas 1.0+*
