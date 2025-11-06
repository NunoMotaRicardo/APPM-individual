# Short Selling Implementation Guide

## Overview
Short selling capability has been added to three portfolio strategies:
- **Markowitz** (Mean-Variance Optimization)
- **GMVP** (Global Minimum Variance Portfolio)
- **MSRP** (Maximum Sharpe Ratio Portfolio)

By default, all strategies maintain long-only constraints (no negative weights). To enable short selling, you must explicitly pass `allow_short=True` via the strategy configuration.

## Usage Pattern

### Basic Usage (Long-Only, Default)
```python
# Long-only by default - no changes needed
bt_strategies = {
    "Markowitz": (MarkowitzStrategy, {}),
    "GMVP": (GMVPStrategy, {}),
}
```

### Enabling Short Selling
To enable short selling, pass `allow_short=True` through `portfolio_func_kwargs`:

```python
bt_strategies = {
    "Markowitz_Short": (MarkowitzStrategy, {
        'portfolio_func_kwargs': {'allow_short': True}
    }),
    "GMVP_Long": (GMVPStrategy, {
        'portfolio_func_kwargs': {'allow_short': False}  # Explicit long-only
    }),
}
```

### Complete Backtest Example
```python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_Core import MarkowitzStrategy, GMVPStrategy, MaximumSharpeRatioStrategy

bt_strategies = {
    # Long-only strategies (default behavior)
    "Markowitz_Long": (MarkowitzStrategy, {}),
    "GMVP_Long": (GMVPStrategy, {}),
    
    # Short-selling enabled strategies
    "Markowitz_Short": (MarkowitzStrategy, {
        'portfolio_func_kwargs': {'allow_short': True},
        'buy_slippage_buffer': 0.07  # Other strategy params work as usual
    }),
    "GMVP_Short": (GMVPStrategy, {
        'portfolio_func_kwargs': {'allow_short': True},
        'buy_slippage_buffer': 0.05
    }),
    "MSRP_Short": (MaximumSharpeRatioStrategy, {
        'portfolio_func_kwargs': {'allow_short': True},
        'buy_slippage_buffer': 0.05
    }),
}

bt_backtest = BacktraderPortfolioBacktest(
    strategies=bt_strategies,
    datasets=my_dataset_list,
    benchmark=['1-N'],
    rebalance_every=21,
    lookback=126,
    initial_cash=100000,
    commission=0.001,
    short_interest=0.03,  # 3% annual short interest cost
    interest_long=False   # Only charge shorts
)

backtest_results = bt_backtest.run_backtest()
```

## How It Works

### Portfolio Function Parameters
Each portfolio function accepts `allow_short` as a keyword argument:

```python
def gmvp_portfolio_fun(dataset, base_column="adjusted", allow_short=False, **kwargs):
    # ... portfolio optimization logic ...
    if allow_short:
        # Allow negative weights (short positions)
        w = w / np.sum(w)
    else:
        # Long-only constraint
        w = np.abs(w) / np.sum(np.abs(w))
```

### Parameter Passing Flow
1. You specify `portfolio_func_kwargs` in strategy configuration
2. Backtest framework passes these kwargs to the portfolio function
3. Portfolio function uses `allow_short` to modify optimization constraints

## Strategy-Specific Implementations

### Markowitz Strategy
```python
def markowitz_portfolio_fun(dataset, base_column="adjusted", lambda_param=0.5, allow_short=False, **kwargs):
    # ...
    if allow_short:
        constraints = [cp.sum(w) == 1]
    else:
        constraints = [w >= 0, cp.sum(w) == 1]
```
- **Short-selling**: Removes non-negativity constraint `w >= 0`
- **Long-only**: Enforces `w >= 0` in optimization

### GMVP Strategy
```python
def gmvp_portfolio_fun(dataset, base_column="adjusted", allow_short=False, **kwargs):
    # ...
    if allow_short:
        w = w / np.sum(w)  # Normalize allowing negatives
    else:
        w = np.abs(w) / np.sum(np.abs(w))  # Force positive weights
```
- **Short-selling**: Analytical solution normalized directly
- **Long-only**: Takes absolute values before normalization

### MSRP Strategy
```python
def maximum_sharpe_ratio_portfolio_fun(dataset, base_column="adjusted", allow_short=False, **kwargs):
    # ...
    if allow_short:
        w = cp.Variable(n)  # Allow negative weights
    else:
        w = cp.Variable(n, nonneg=True)  # Long-only
```
- **Short-selling**: CVXPY variable without non-negativity constraint
- **Long-only**: Uses `nonneg=True` parameter

## Important Notes

### Short Interest Costs
When using short selling, always configure `short_interest` in the backtest:
```python
bt_backtest = BacktraderPortfolioBacktest(
    # ... other params ...
    short_interest=0.03,  # 3% annual cost for short positions
    interest_long=False   # Don't charge longs (typical for stocks)
)
```

### Margin Requirements
- Short positions may be rejected if insufficient margin
- Backtrader automatically checks available cash/margin
- Consider using lower `buy_slippage_buffer` for long-only strategies

### Strategy Compatibility
Only these three strategies support short selling:
- ✅ **Markowitz** - Easy modification (remove constraint)
- ✅ **GMVP** - Easy modification (adjust normalization)
- ✅ **MSRP** - Easy modification (remove constraint)
- ❌ **Risk Parity** - Not compatible (math requires positive weights)
- ❌ **IVP** - Not compatible (conceptually long-only)
- ❌ **Quintile** - Not compatible (ranking-based)
- ❌ **MDP/MDCP** - Complex modification required

## Comparison Example

Create matched pairs to compare long-only vs short-selling:
```python
bt_strategies = {
    # Long-only versions
    "GMVP_Long": (GMVPStrategy, {
        'portfolio_func_kwargs': {'allow_short': False}
    }),
    
    # Short-selling versions  
    "GMVP_Short": (GMVPStrategy, {
        'portfolio_func_kwargs': {'allow_short': True}
    }),
}
```

Then analyze results:
```python
from Backtester.BacktestResults import TestResults

test_results = TestResults("data/selection3/test-1")
long_results = test_results.strategies['GMVP_Long']
short_results = test_results.strategies['GMVP_Short']

# Compare performance
long_sharpe = long_results.get_datasets_performance()['sharpe_ratio'].mean()
short_sharpe = short_results.get_datasets_performance()['sharpe_ratio'].mean()
print(f"Long-only Sharpe: {long_sharpe:.3f}")
print(f"Short-selling Sharpe: {short_sharpe:.3f}")
```

## See Also
- `SHORT_INTEREST_IMPLEMENTATION.md` - Short interest cost configuration
- `SHORT_INTEREST_EXAMPLES.md` - More usage examples
- `Strategy_Core.py` - Portfolio function implementations
