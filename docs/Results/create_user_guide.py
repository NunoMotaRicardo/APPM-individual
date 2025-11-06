#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate comprehensive BacktestResults documentation."""

USER_GUIDE = r"""# BacktestResults User Guide

**Comprehensive Guide for Analyzing Backtest Results**

Version 1.0 | October 2025

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Class Hierarchy](#class-hierarchy)
3. [API Reference](#api-reference)
4. [Performance Metrics](#performance-metrics)
5. [Common Workflows](#common-workflows)
6. [Advanced Examples](#advanced-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation & Setup

```python
from Backtester.BacktestResults import TestResults

# Load test results
test = TestResults('data/selection3/test-2')

# Access test information
print(f"Test: {test.name}")
print(f"Universe: {test.universe_name}")
print(f"Datasets: {test.num_datasets}")
print(f"Duration: {test.duration} days")
```

### Basic Usage

```python
# List available resources
strategies = test.list_strategies()
datasets = test.list_datasets()

print(f"Strategies: {strategies}")
print(f"Datasets: {len(datasets)} total")

# Get performance comparison
perf = test.get_strategies_bt_performance()
print(perf)
```

---

## Class Hierarchy

### Three-Level Structure

```
TestResults (Top Level)
├── Test metadata & configuration
├── Universe settings
├── Datasets (DatasetResults) - Universe data only
└── Strategies (dict)
    └── StrategyResults (Strategy Level)
        ├── Aggregated performance across datasets
        └── Datasets (dict)
            └── DatasetResults (Dataset Level)
                ├── Returns, weights, orders
                ├── Asset values tracking
                └── Performance metrics
```

### TestResults
- **Purpose**: Container for entire test
- **Contains**: All strategies, universe data, test configuration
- **Key Methods**: list_strategies(), get_strategies_bt_performance()

### StrategyResults
- **Purpose**: Aggregated strategy results across all datasets
- **Contains**: Mean returns, performance statistics
- **Key Methods**: get_returns(), get_performance(), get_datasets_weights()

### DatasetResults
- **Purpose**: Detailed results for one strategy on one dataset
- **Contains**: Returns, weights, orders, asset values, final value
- **Key Methods**: get_returns(), get_weights(), get_orders(), get_asset_values()

---

## API Reference

### TestResults Methods

#### Information Methods
```python
test.list_strategies() -> List[str]
# Returns names of all tested strategies

test.list_datasets() -> List[str]  
# Returns names of all datasets (universe data)

test.get_datasets_info(dataset_names=[]) -> Dict
# Get metadata for datasets (start_date, end_date, symbols)
```

#### Performance Methods
```python
test.get_strategies_bt_performance(
    strategy_names=[],
    aggregator='mean'
) -> DataFrame
# Get backtrader performance metrics for strategies
# aggregator: 'mean', 'median', 'std'

test.get_strategies_stats(strategy_names=[]) -> DataFrame
# Get computed performance statistics
```

### StrategyResults Methods

#### Returns & Performance
```python
strategy.get_returns() -> DataFrame
# Aggregated returns across all datasets
# Returns DataFrame with 'return' column

strategy.get_performance() -> DataFrame
# Computed performance metrics:
# - annual_return, annual_volatility
# - sharpe_ratio, sortino_ratio
# - max_drawdown, CAGR
# - skew, kurtosis, VaR_5pct

strategy.list_datasets() -> List[str]
# List dataset names for this strategy
```

#### Dataset-Level Access
```python
strategy.get_datasets_returns(dataset_names=[]) -> DataFrame
# Returns for each dataset (columns)

strategy.get_datasets_performance(dataset_names=[]) -> DataFrame
# Performance metrics for each dataset

strategy.get_datasets_weights(dataset_names=[]) -> Dict
# Portfolio weights history for each dataset
# Returns: {dataset_name: weights_df}

strategy.get_datasets_asset_values(
    dataset_names=[],
    component=None
) -> Dict
# Asset-level tracking data
# component: 'prices', 'positions', 'values', 'portfolio', or None for all
```

### DatasetResults Methods

```python
dataset.get_returns() -> DataFrame
# Daily returns with 'return' column

dataset.get_weights() -> DataFrame
# Portfolio weights at each rebalancing
# Columns: date, asset symbols

dataset.get_orders() -> DataFrame
# Order execution history
# Columns: type, symbol, size, created_price, executed_price, etc.

dataset.get_asset_values(component=None) -> Dict | DataFrame
# Asset-level tracking:
# - 'prices': Asset prices at rebalancing
# - 'positions': Shares held
# - 'values': Position values (price × shares)
# - 'portfolio': Portfolio value, cash
# If component specified, returns DataFrame; else returns Dict

dataset.get_performance() -> DataFrame
# Computed performance metrics

dataset.get_bt_performance() -> DataFrame
# Backtrader engine performance metrics
```

### Key Attributes

```python
# TestResults
test.name                 # Test name
test.universe_name        # Universe identifier
test.num_datasets         # Number of datasets
test.lookback_periods     # Lookback period used
test.duration             # Backtest duration (days)
test.random_seed          # Random seed
test.universe_settings    # Dict with universe configuration
test.datasets             # Dict of DatasetResults (universe data)
test.strategies           # Dict of StrategyResults

# StrategyResults
strategy.name             # Strategy name
strategy.datasets         # Dict of DatasetResults

# DatasetResults
dataset.name              # Dataset identifier
dataset.strategy_name     # Strategy name
dataset.lookback_periods  # Lookback period
dataset.final_value       # Final portfolio value
dataset.performance_bt    # Dict of backtrader metrics
dataset.annual_returns_by_year  # DataFrame (if available)
dataset.returns           # DataFrame of daily returns
dataset.weights_history   # DataFrame of portfolio weights
dataset.orders            # DataFrame of orders
dataset.asset_values_history  # Dict of asset tracking DataFrames
```

---

## Performance Metrics

### Backtrader Engine Metrics

Calculated by backtrader during execution:

| Metric | Description | Formula/Notes |
|--------|-------------|---------------|
| `Sharpe ratio` | Risk-adjusted return | (mean - rf) / std × √252 |
| `Sortino ratio` | Downside risk-adjusted | mean / downside_std × √252 |
| `annual return` | Annualized return | Calculated by backtrader |
| `annual volatility` | Annualized std dev | std × √252 |
| `max drawdown` | Maximum peak-to-trough | Largest cumulative loss |
| `Calmar ratio` | Return/drawdown | annual_return / max_drawdown |
| `VaR (5%)` | Value at Risk | 5th percentile of returns |
| `VWR` | Variability-Weighted Return | Backtrader analyzer |
| `SQN` | System Quality Number | Van Tharp's metric |
| `skewness` | Return distribution skew | Scipy skew |
| `kurtosis` | Return distribution kurtosis | Scipy kurtosis |

### Computed Metrics

Calculated by BacktestResults from returns:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| `annual_return` | Annualized return | From log returns |
| `annual_volatility` | Annualized volatility | std(returns) × √252 |
| `max_drawdown` | Maximum drawdown | From cumulative log returns |
| `sharpe_ratio` | Sharpe ratio | mean/std × √252 |
| `sortino_ratio` | Sortino ratio | mean/downside_std × √252 |
| `CAGR` | Compound Annual Growth Rate | (Final/Initial)^(252/days) - 1 |
| `VaR_5pct` | 5% Value at Risk | 5th percentile |
| `mean` | Mean return | returns.mean() |
| `std_dev` | Standard deviation | returns.std() |
| `skew` | Skewness | returns.skew() |
| `kurtosis` | Kurtosis | returns.kurtosis() |

---

## Common Workflows

### 1. Load and Explore

```python
from Backtester.BacktestResults import TestResults

# Load test
test = TestResults('data/selection3/test-2')

# Explore structure
print(f"Test: {test.name}")
print(f"Strategies: {test.list_strategies()}")
print(f"Datasets: {len(test.list_datasets())}")

# Universe information
universe = test.universe_settings['universe_symbols']
print(f"\nUniverse has {len(universe)} symbols")
print(universe.head())
```

### 2. Compare Strategies

```python
import matplotlib.pyplot as plt

# Get performance for all strategies
perf = test.get_strategies_bt_performance()

# Plot key metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

perf['Sharpe ratio'].plot(kind='bar', ax=axes[0,0], title='Sharpe Ratio')
axes[0,0].grid(True, alpha=0.3)

perf['annual return'].plot(kind='bar', ax=axes[0,1], title='Annual Return')
axes[0,1].grid(True, alpha=0.3)

perf['max drawdown'].plot(kind='bar', ax=axes[1,0], title='Max Drawdown')
axes[1,0].grid(True, alpha=0.3)

perf['annual volatility'].plot(kind='bar', ax=axes[1,1], title='Volatility')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3. Analyze Single Strategy

```python
# Get strategy
strategy = test.strategies['GEM']

# Get aggregated returns
returns = strategy.get_returns()
print(f"Total periods: {len(returns)}")

# Get performance
performance = strategy.get_performance()
print(performance[['sharpe_ratio', 'annual_return', 'max_drawdown']])

# Plot cumulative returns
cumulative = (1 + returns['return']).cumprod()
plt.figure(figsize=(14, 7))
plt.plot(cumulative)
plt.title(f'{strategy.name} - Cumulative Returns')
plt.xlabel('Period')
plt.ylabel('Cumulative Return')
plt.grid(True, alpha=0.3)
plt.show()
```

### 4. Deep Dive into Dataset

```python
# Get specific dataset
dataset = strategy.datasets['dataset_1']

print(f"Final Value: ${dataset.final_value:,.2f}")
print(f"Periods: {len(dataset.returns)}")

# Get all components
returns = dataset.get_returns()
weights = dataset.get_weights()
orders = dataset.get_orders()
asset_values = dataset.get_asset_values()

# Analyze orders
print(f"\nOrders executed: {len(orders)}")
print(f"Order types:\n{orders['type'].value_counts()}")

# Plot portfolio value evolution
portfolio_df = asset_values['portfolio']
portfolio_df[['portfolio_value', 'cash']].plot(
    title='Portfolio Evolution',
    figsize=(14, 6)
)
plt.ylabel('Value ($)')
plt.show()
```

### 5. Portfolio Weights Analysis

```python
import seaborn as sns

# Get weights
weights = dataset.get_weights()

# Prepare for heatmap
weights_pivot = weights.set_index('date')

# Create heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    weights_pivot.T,
    cmap='RdYlGn',
    center=0,
    cbar_kws={'label': 'Weight'},
    linewidths=0.5
)
plt.title(f'{strategy.name} - Portfolio Weights Evolution')
plt.xlabel('Rebalancing Date')
plt.ylabel('Asset')
plt.tight_layout()
plt.show()
```

---

## Advanced Examples

### Example 1: Cumulative Returns Comparison

```python
import numpy as np

strategies_to_compare = ['GEM', 'AAA', 'EqualWeight']

plt.figure(figsize=(14, 7))

for strategy_name in strategies_to_compare:
    strategy = test.strategies[strategy_name]
    returns = strategy.get_returns()
    
    # Calculate cumulative
    cumulative = (1 + returns['return']).cumprod()
    
    plt.plot(cumulative.index, cumulative.values,
             label=strategy_name, linewidth=2)

plt.title('Strategy Comparison - Cumulative Returns',
          fontsize=14, fontweight='bold')
plt.xlabel('Period')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Example 2: Drawdown Analysis

```python
def calculate_drawdown(returns):
    """Calculate drawdown from returns series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown

# Analyze drawdowns for multiple strategies
fig, axes = plt.subplots(len(strategies_to_compare), 1,
                         figsize=(14, 4*len(strategies_to_compare)))

for idx, strategy_name in enumerate(strategies_to_compare):
    strategy = test.strategies[strategy_name]
    returns = strategy.get_returns()['return']
    
    dd = calculate_drawdown(returns)
    
    ax = axes[idx] if len(strategies_to_compare) > 1 else axes
    ax.fill_between(dd.index, 0, dd.values, alpha=0.3, color='red')
    ax.plot(dd.index, dd.values, color='darkred', linewidth=1)
    ax.set_title(f'{strategy_name} - Drawdown', fontweight='bold')
    ax.set_ylabel('Drawdown')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()
```

### Example 3: Risk-Return Scatter

```python
import pandas as pd

# Collect data
risk_return_data = []

for strategy_name in test.list_strategies():
    strategy = test.strategies[strategy_name]
    perf = strategy.get_performance().iloc[0]
    
    risk_return_data.append({
        'strategy': strategy_name,
        'return': perf['annual_return'],
        'volatility': perf['annual_volatility'],
        'sharpe': perf['sharpe_ratio']
    })

df = pd.DataFrame(risk_return_data)

# Create scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df['volatility'],
    df['return'],
    s=df['sharpe']*100,  # Size by Sharpe
    alpha=0.6,
    c=df['sharpe'],      # Color by Sharpe
    cmap='viridis'
)

# Add labels
for idx, row in df.iterrows():
    plt.annotate(
        row['strategy'],
        (row['volatility'], row['return']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9
    )

plt.xlabel('Annual Volatility', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.title('Risk-Return Profile (bubble size = Sharpe Ratio)',
          fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Sharpe Ratio')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Example 4: Performance Stability

```python
# Analyze consistency across datasets

strategy = test.strategies['GEM']
dataset_names = strategy.list_datasets()

# Collect performance for each dataset
perf_list = []
for ds_name in dataset_names:
    dataset = strategy.datasets[ds_name]
    perf = dataset.get_performance().iloc[0]
    perf['dataset'] = ds_name
    perf_list.append(perf)

perf_df = pd.DataFrame(perf_list)

# Plot distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics = ['sharpe_ratio', 'annual_return', 'max_drawdown', 'CAGR']
titles = ['Sharpe Ratio', 'Annual Return', 'Max Drawdown', 'CAGR']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx//2, idx%2]
    
    perf_df[metric].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
    ax.axvline(perf_df[metric].mean(), color='red',
              linestyle='--', linewidth=2, label='Mean')
    ax.axvline(perf_df[metric].median(), color='blue',
              linestyle='--', linewidth=2, label='Median')
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(f'{strategy.name} - Performance Stability Across Datasets',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Print statistics
print(f"\nPerformance Statistics for {strategy.name}")
print("=" * 60)
print(perf_df[metrics].describe())
```

### Example 5: Asset Contribution Analysis

```python
# Analyze asset-level data

dataset = strategy.datasets['dataset_1']
asset_values = dataset.get_asset_values()

# Get components
values = asset_values['values']
weights = dataset.get_weights()

# Calculate contribution to return
# (weight × asset_return)
# This is simplified - actual calculation more complex

print(f"Assets tracked: {values.columns.tolist()}")
print(f"Rebalancing events: {len(weights)}")

# Plot asset values evolution
plt.figure(figsize=(14, 8))
values.plot(ax=plt.gca())
plt.title('Asset Values Over Time')
plt.xlabel('Rebalancing Date')
plt.ylabel('Value ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Example 6: Order Execution Analysis

```python
# Analyze order execution patterns

dataset = strategy.datasets['dataset_1']
orders = dataset.get_orders()

# Calculate slippage
orders['slippage'] = orders['executed_price'] - orders['created_price']
orders['slippage_pct'] = (orders['slippage'] / orders['created_price']) * 100

# Separate buy and sell orders
buy_orders = orders[orders['size'] > 0]
sell_orders = orders[orders['size'] < 0]

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Slippage distribution
axes[0].hist(orders['slippage_pct'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Slippage Distribution')
axes[0].set_xlabel('Slippage (%)')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Buy vs Sell slippage
axes[1].hist([buy_orders['slippage_pct'], sell_orders['slippage_pct']],
            bins=30, label=['Buy Orders', 'Sell Orders'],
            alpha=0.7, edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Slippage by Order Type')
axes[1].set_xlabel('Slippage (%)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistics
print(f"Total Orders: {len(orders)}")
print(f"\nOrder Types:")
print(orders['type'].value_counts())
print(f"\nSlippage Statistics:")
print(orders['slippage_pct'].describe())
print(f"\nMean Slippage:")
print(f"  Buy Orders:  {buy_orders['slippage_pct'].mean():.4f}%")
print(f"  Sell Orders: {sell_orders['slippage_pct'].mean():.4f}%")
```

---

## Best Practices

### 1. Efficient Data Loading

```python
# Load once, reuse
test = TestResults('data/selection3/test-2')

# Cache frequently used strategies
gem = test.strategies['GEM']
aaa = test.strategies['AAA']

# Process in batches for large datasets
dataset_names = test.list_datasets()
batch_size = 20

for i in range(0, len(dataset_names), batch_size):
    batch = dataset_names[i:i+batch_size]
    # Process batch
    weights = strategy.get_datasets_weights(batch)
    # Analyze...
```

### 2. Error Handling

```python
def safe_load_test(path):
    """Safely load test results with error handling."""
    try:
        return TestResults(path)
    except FileNotFoundError:
        print(f"Test not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return None
    except Exception as e:
        print(f"Error loading test: {e}")
        return None

test = safe_load_test('data/selection3/test-2')
if test:
    # Proceed with analysis
    pass
```

### 3. Check for Empty Data

```python
# Always verify data exists
returns = dataset.get_returns()
if returns.empty:
    print("No returns data available")
else:
    # Analyze returns
    cumulative = (1 + returns['return']).cumprod()
    cumulative.plot()

# Check for None values
if dataset.annual_returns_by_year is not None:
    print(dataset.annual_returns_by_year)
else:
    print("Annual returns not computed")
```

### 4. Visualization Configuration

```python
# Set consistent style
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# High-quality exports
plt.savefig('output.png', dpi=300, bbox_inches='tight')
```

### 5. Reusable Analysis Functions

```python
def analyze_strategy(strategy, metrics=None):
    """Analyze strategy performance."""
    if metrics is None:
        metrics = ['sharpe_ratio', 'annual_return', 'max_drawdown']
    
    perf = strategy.get_performance()
    return perf[metrics].iloc[0]

# Use for all strategies
comparison = {
    name: analyze_strategy(strat)
    for name, strat in test.strategies.items()
}

import pandas as pd
df = pd.DataFrame(comparison).T
print(df)
```

### 6. Export Results

```python
# Export to CSV
performance = test.get_strategies_bt_performance()
performance.to_csv('results/strategy_performance.csv')

# Export to Excel (multiple sheets)
with pd.ExcelWriter('results/backtest_analysis.xlsx') as writer:
    # Performance summary
    performance.to_excel(writer, sheet_name='Performance')
    
    # Individual strategy returns
    for strategy_name in test.list_strategies():
        strategy = test.strategies[strategy_name]
        returns = strategy.get_returns()
        returns.to_excel(writer, sheet_name=f'{strategy_name}_Returns')
    
    # Universe info
    universe = test.universe_settings['universe_symbols']
    universe.to_excel(writer, sheet_name='Universe')
```

---

## Troubleshooting

### Common Issues & Solutions

**Issue**: `FileNotFoundError: Test path not found`  
**Solution**: Verify test_path exists and contains required files:
- test_settings.json
- datasets_info.json  
- results/*.json files

**Issue**: Empty DataFrames returned  
**Solution**: Check if backtest generated data for that component. Some datasets may not have all data types.

**Issue**: `KeyError` when accessing strategy/dataset  
**Solution**: Use `list_strategies()` and `list_datasets()` to verify names exist.

**Issue**: Memory errors with large tests  
**Solution**: Process datasets in chunks using batch processing pattern.

**Issue**: Plots not showing  
**Solution**: Ensure `plt.show()` is called or use Jupyter inline plotting.

**Issue**: Inconsistent performance metrics  
**Solution**: Check which metrics are from backtrader vs computed. They may differ slightly.

### Debug Helper

```python
def debug_test_structure(test_path):
    """Print test structure for debugging."""
    from pathlib import Path
    import json
    
    path = Path(test_path)
    
    print(f"Test Path: {path}")
    print(f"Exists: {path.exists()}")
    
    if path.exists():
        print(f"\nFiles in test directory:")
        for file in path.glob('*'):
            print(f"  {file.name}: {file.stat().st_size} bytes")
        
        # Check results
        results_path = path / 'results'
        if results_path.exists():
            print(f"\nStrategy result files:")
            for file in results_path.glob('*.json'):
                print(f"  {file.name}")
        
        # Load settings
        settings_file = path / 'test_settings.json'
        if settings_file.exists():
            with open(settings_file) as f:
                settings = json.load(f)
            print(f"\nTest Settings:")
            for key, value in settings.items():
                print(f"  {key}: {value}")

# Use it
debug_test_structure('data/selection3/test-2')
```

---

## Appendix: Complete API

### TestResults
- `__init__(test_path: str | Path)`
- `list_datasets() -> List[str]`
- `list_strategies() -> List[str]`
- `get_datasets_info(dataset_names: list = []) -> Dict`
- `get_datasets_data(datasets_filter_list: list = [], column: str = "") -> Dict`
- `get_strategies_stats(strategy_names: list = []) -> DataFrame`
- `get_strategies_bt_performance(strategy_names: list = [], aggregator: str = "mean") -> DataFrame`

### StrategyResults
- `__init__(payload: Dict, lookback_periods: int)`
- `list_datasets() -> List[str]`
- `get_datasets_returns(dataset_names: list = []) -> DataFrame`
- `get_datasets_bt_performance(dataset_names: list = []) -> DataFrame`
- `get_datasets_performance(dataset_names: list = []) -> DataFrame`
- `get_datasets_weights(dataset_names: list = []) -> Dict`
- `get_datasets_asset_values(dataset_names: list = [], component: Optional[str] = None) -> Dict`
- `get_returns() -> DataFrame`
- `get_performance() -> DataFrame`

### DatasetResults
- `__init__(name: str, payload: Dict, lookback_periods: int, strategy_name: str)`
- `get_performance() -> DataFrame`
- `get_bt_performance() -> DataFrame`
- `get_returns() -> DataFrame`
- `get_weights() -> DataFrame`
- `get_orders() -> DataFrame`
- `get_asset_values(component: Optional[str] = None) -> Dict[str, DataFrame] | DataFrame`

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Maintained by**: Backtest Analysis Team

For questions or suggestions, please create an issue or contact the development team.
"""

# Save to file
with open('docs/BacktestResults_UserGuide.md', 'w', encoding='utf-8') as f:
    f.write(USER_GUIDE)

print("✓ User Guide created: docs/BacktestResults_UserGuide.md")
