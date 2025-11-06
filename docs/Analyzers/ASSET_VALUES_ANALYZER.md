# Asset Values Analyzer Documentation

## Overview

The `AssetValuesAnalyzer` is a custom Backtrader analyzer that tracks the value of all assets at the beginning of each rebalancing period. This provides detailed insights into portfolio composition, asset performance, and position sizing over time.

## Features

The analyzer captures the following information at each rebalancing event:

- **Rebalancing Dates**: Timestamps of when portfolio rebalancing occurs
- **Asset Names**: List of assets in the portfolio
- **Asset Prices**: Current price of each asset at rebalancing time
- **Asset Positions**: Number of shares held for each asset
- **Asset Values**: Dollar value (price × shares) for each asset
- **Portfolio Value**: Total portfolio value including cash
- **Cash**: Available cash at rebalancing time

## Implementation Details

### How It Works

1. **Automatic Integration**: The analyzer is automatically added to all backtests via the `BacktraderPortfolioBacktest` class
2. **Pre-Rebalancing Capture**: Values are captured BEFORE rebalancing orders are placed, showing the current state
3. **Strategy Notification**: The strategy's `notify_asset_values_analyzer()` method is called during each rebalancing event
4. **Data Collection**: Asset prices and positions are gathered from all data feeds and broker state

### Code Location

- **Analyzer Class**: `AssetValuesAnalyzer` in `Backtester/BacktestFramework.py`
- **Notification Method**: `notify_asset_values_analyzer()` in `PortfolioRebalanceStrategy` class
- **Registration**: Added via `cerebro.addanalyzer(AssetValuesAnalyzer, _name='asset_values')`
- **Result Loading**: `get_asset_values()` method in `DatasetResults` class in `Backtester/BacktestResults.py`

## Usage

### Method 1: Direct Access from Backtest Results

After running a backtest, the asset values data is available in the results dictionary:

\\\python
# Run backtest
backtest = BacktraderPortfolioBacktest(
    strategies=bt_strategies, 
    datasets=my_dataset_list,
    rebalance_every=21,
    lookback=126
)
results = backtest.run_backtest()

# Access asset values for a specific strategy and dataset
strategy_name = 'HRPP_ward'
dataset_name = 'dataset_1'
asset_values_data = results[strategy_name][dataset_name]['asset_values_history']

# The data is a dictionary with DataFrames already built
prices_df = asset_values_data['prices']
positions_df = asset_values_data['positions']
values_df = asset_values_data['values']
portfolio_df = asset_values_data['portfolio']
\\\

### Method 2: Using BacktestResults Classes (Recommended)

For saved/loaded test results, use the convenient methods in `DatasetResults` and `StrategyResults`:

\\\python
from Backtester.BacktestResults import TestResults

# Load test results
test = TestResults('data/selection3/test-1')

# Get asset values for a specific dataset
strategy = test.strategies['HRPP_ward']
dataset_result = strategy.datasets['dataset_1']

# Get all asset values components
all_data = dataset_result.get_asset_values()
prices_df = all_data['prices']
positions_df = all_data['positions']
values_df = all_data['values']
portfolio_df = all_data['portfolio']

# Or get a specific component directly
prices_df = dataset_result.get_asset_values('prices')
positions_df = dataset_result.get_asset_values('positions')
values_df = dataset_result.get_asset_values('values')
portfolio_df = dataset_result.get_asset_values('portfolio')

# Get asset values for multiple datasets
all_datasets_values = strategy.get_datasets_asset_values()
dataset1_prices = all_datasets_values['dataset_1']['prices']

# Get specific component for multiple datasets
all_positions = strategy.get_datasets_asset_values(component='positions')
dataset1_positions = all_positions['dataset_1']
\\\

### Data Structure

Each component returns a pandas DataFrame:

**Prices DataFrame:**
\\\
date        AAPL   MSFT   GOOGL
2020-01-15  150.5  180.2  1200.3
2020-02-15  152.1  182.5  1215.8
\\\

**Positions DataFrame:**
\\\
date        AAPL  MSFT  GOOGL
2020-01-15   100    80     15
2020-02-15   105    82     16
\\\

**Values DataFrame:**
\\\
date          AAPL      MSFT      GOOGL
2020-01-15  15050.0  14416.0  18004.5
2020-02-15  15970.5  14965.0  19452.8
\\\

**Portfolio DataFrame:**
\\\
date        portfolio_value    cash
2020-01-15       100000.0    5000.0
2020-02-15       102500.0    4200.0
\\\

## Examples

### Example 1: Analyze Asset Contribution

\\\python
from Backtester.BacktestResults import TestResults

test = TestResults('data/selection3/test-1')
dataset_result = test.strategies['HRPP_ward'].datasets['dataset_1']

# Get asset values and portfolio data
values_df = dataset_result.get_asset_values('values')
portfolio_df = dataset_result.get_asset_values('portfolio')

# Calculate percentage contribution of each asset to portfolio
contributions = values_df.div(portfolio_df['portfolio_value'], axis=0) * 100

# Plot asset contributions over time
import matplotlib.pyplot as plt

contributions.plot(kind='area', stacked=True, figsize=(12, 6))
plt.title('Asset Contributions to Portfolio Over Time')
plt.ylabel('Percentage (%)')
plt.xlabel('Rebalancing Date')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
\\\

### Example 2: Track Position Changes

\\\python
dataset_result = test.strategies['HRPP_ward'].datasets['dataset_1']
positions_df = dataset_result.get_asset_values('positions')

# Calculate position changes between rebalancing periods
position_changes = positions_df.diff()

# Identify which assets were bought/sold
for date in position_changes.index[1:]:  # Skip first row (NaN)
    changes = position_changes.loc[date]
    bought = changes[changes > 0]
    sold = changes[changes < 0]
    
    print(f"\\nRebalancing on {date}:")
    if len(bought) > 0:
        print(f"  Bought: {bought.to_dict()}")
    if len(sold) > 0:
        print(f"  Sold: {sold.to_dict()}")
\\\

### Example 3: Compare Asset Performance Across Strategies

\\\python
test = TestResults('data/selection3/test-1')

strategies = ['HRPP_ward', 'HRPP_single', 'DRPP']
dataset_name = 'dataset_1'

for strategy_name in strategies:
    values_df = test.strategies[strategy_name].datasets[dataset_name].get_asset_values('values')
    portfolio_df = test.strategies[strategy_name].datasets[dataset_name].get_asset_values('portfolio')
    
    # Calculate total return for each asset
    asset_returns = (values_df.iloc[-1] - values_df.iloc[0]) / values_df.iloc[0] * 100
    
    print(f"\\n{strategy_name} - Asset Returns:")
    print(asset_returns.sort_values(ascending=False))
\\\

### Example 4: Concentration Risk Analysis

\\\python
dataset_result = test.strategies['HRPP_ward'].datasets['dataset_1']
values_df = dataset_result.get_asset_values('values')
portfolio_df = dataset_result.get_asset_values('portfolio')

# Calculate weight of each asset at each rebalancing
weights = values_df.div(portfolio_df['portfolio_value'], axis=0)

# Calculate Herfindahl-Hirschman Index (HHI) for concentration
hhi = (weights ** 2).sum(axis=1)

print("Concentration Analysis (HHI):")
print(f"Mean HHI: {hhi.mean():.4f}")
print(f"Min HHI: {hhi.min():.4f} (most diversified)")
print(f"Max HHI: {hhi.max():.4f} (most concentrated)")

# Plot HHI over time
hhi.plot(figsize=(10, 4))
plt.title('Portfolio Concentration Over Time (HHI)')
plt.ylabel('HHI Index')
plt.xlabel('Date')
plt.axhline(y=1/len(values_df.columns), color='r', linestyle='--', label='Equal weight benchmark')
plt.legend()
plt.show()
\\\

## Benefits

1. **Performance Attribution**: Identify which assets contribute most to portfolio performance
2. **Rebalancing Analysis**: Track how positions change over time
3. **Risk Monitoring**: Analyze concentration risk by tracking asset values
4. **Debugging**: Verify that portfolio optimization and rebalancing logic work correctly
5. **Compliance**: Document portfolio composition at key decision points
6. **Easy Access**: Structured DataFrames ready for analysis

## Integration with Existing Analyzers

The Asset Values Analyzer works alongside existing analyzers:

- **PortfolioWeightsAnalyzer**: Tracks target weights (what strategy wants)
- **AssetValuesAnalyzer**: Tracks actual positions and values (what portfolio has)
- **OrderTrackingAnalyzer**: Tracks execution details (how we got there)

Together, these provide a complete picture of portfolio evolution.

## API Reference

### DatasetResults.get_asset_values()

\\\python
def get_asset_values(component: Optional[str] = None) -> Dict[str, pd.DataFrame] | pd.DataFrame
\\\

**Parameters:**
- `component` (str, optional): Which component to return. Options: 'prices', 'positions', 'values', 'portfolio'. If None, returns all components.

**Returns:**
- If `component` is None: Dictionary with keys ['prices', 'positions', 'values', 'portfolio']
- If `component` is specified: DataFrame for that component

### StrategyResults.get_datasets_asset_values()

\\\python
def get_datasets_asset_values(dataset_names: list = [], component: Optional[str] = None) -> dict
\\\

**Parameters:**
- `dataset_names` (list, optional): List of dataset names. If empty, returns all datasets.
- `component` (str, optional): Which component to return. If None, returns all components for each dataset.

**Returns:**
- Dictionary with dataset names as keys and asset values data as values

## Notes

- Values are captured BEFORE rebalancing orders are executed
- This shows the portfolio state at the decision point
- First rebalancing event typically occurs after warmup period
- Asset names list may vary if using dynamic asset selection strategies
- All DataFrames have a datetime index named 'date'
- Empty DataFrames are returned if no asset values data is available

## See Also

- `PortfolioWeightsAnalyzer` - for target portfolio weights
- `OrderTrackingAnalyzer` - for order execution details
- Backtrader Analyzer Documentation: https://www.backtrader.com/docu/analyzers/
