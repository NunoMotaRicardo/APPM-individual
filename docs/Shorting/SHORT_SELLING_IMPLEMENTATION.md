# Short Selling Implementation

This document describes the short-selling capability added to selected portfolio strategies.

## Overview

Three portfolio strategies now support short positions (negative weights):
1. **Markowitz (MVP)** - Mean-variance portfolio optimization
2. **GMVP** - Global Minimum Variance Portfolio
3. **MSRP** - Maximum Sharpe Ratio Portfolio

All three strategies have an llow_short parameter that defaults to False for backward compatibility.

## Affected Files

- Backtester/Strategy_Core.py: Modified portfolio functions and strategy classes

## Implementation Details

### Portfolio Functions

Each of the three portfolio functions now accepts an llow_short parameter:

`python
def markowitz_portfolio_fun(dataset, base_column="adjusted", lambda_param=0.5, allow_short=False, **kwargs)
def gmvp_portfolio_fun(dataset, base_column="adjusted", allow_short=False, **kwargs)
def maximum_sharpe_ratio_portfolio_fun(dataset, base_column="adjusted", allow_short=False, **kwargs)
`

**When llow_short=False (default):**
- Markowitz: Constraint w >= 0 is applied
- GMVP: Uses 
p.abs(w) to enforce positive weights
- MSRP: Creates variable with 
onneg=True constraint

**When llow_short=True:**
- Markowitz: Removes the w >= 0 constraint, only keeps sum(w) == 1
- GMVP: Removes the absolute value, allows negative weights with sum(w) == 1
- MSRP: Creates unrestricted variable, allows negative weights

### Strategy Classes

The strategy classes now have an llow_short parameter:

`python
class MarkowitzStrategy(PortfolioRebalanceStrategy):
    params = dict(
        PortfolioRebalanceStrategy.params,
        allow_short=False,  # Default: long-only
    )
`

The parameter is passed to the portfolio function via portfolio_func_kwargs:

`python
self.params.portfolio_func_kwargs = {'allow_short': self.params.allow_short}
`

## Usage Examples

### Example 1: Long-only (default behavior)

`python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_Core import MarkowitzStrategy, GMVPStrategy

# Default behavior - long-only
strategies = {
    'Markowitz': MarkowitzStrategy,
    'GMVP': GMVPStrategy
}

backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=dataset_list,
    rebalance_every=21,
    lookback=126
)

results = backtest.run_backtest()
`

### Example 2: Enabling short selling

`python
# Enable short selling for specific strategies
strategies = {
    'Markowitz_Long': (MarkowitzStrategy, {}),
    'Markowitz_Short': (MarkowitzStrategy, {'allow_short': True}),
    'GMVP_Long': (GMVPStrategy, {}),
    'GMVP_Short': (GMVPStrategy, {'allow_short': True}),
}

backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=dataset_list,
    rebalance_every=21,
    lookback=126,
    short_interest=0.03,  # 3% annual short interest (recommended with shorts)
    interest_long=False
)

results = backtest.run_backtest()
`

### Example 3: In notebook with custom parameters

`python
bt_strategies = {
    "MVP": (MarkowitzStrategy, {
        'buy_slippage_buffer': 0.07,
        'allow_short': False  # Long-only
    }),
    "MVP_Short": (MarkowitzStrategy, {
        'buy_slippage_buffer': 0.07,
        'allow_short': True  # Allow shorts
    }),
    "GMVP": (GMVPStrategy, {
        'buy_slippage_buffer': 0.05,
    }),
    "MSRP_Short": (MaximumSharpeRatioStrategy, {
        'buy_slippage_buffer': 0.05,
        'allow_short': True
    }),
}

bt_backtest = BacktraderPortfolioBacktest(
    strategies=bt_strategies, 
    datasets=my_dataset_list,
    benchmark=['1-N'],
    rebalance_every=int(rebalance_days),
    lookback=int(lookback_periods),    
    initial_cash=100000,
    commission=0.001,
    short_interest=0.03,  # Important: add short interest costs
    interest_long=False
)

backtest_results = bt_backtest.run_backtest()
`

## Important Considerations

### 1. Short Interest Costs

When using short selling, you should enable short interest costs:

`python
backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=dataset_list,
    short_interest=0.03,  # 3% annual rate for borrowing costs
    interest_long=False   # Only charge shorts
)
`

See docs/SHORT_INTEREST_IMPLEMENTATION.md for typical rates and details.

### 2. Margin Requirements

Short positions require margin. The framework includes built-in margin protection:
- Uses getcommissioninfo() to calculate total order costs
- Includes uy_slippage_buffer to prevent margin calls
- You may need to increase uy_slippage_buffer for short strategies (e.g., 0.05-0.10)

### 3. Weight Interpretation

With short selling enabled:
- Positive weights = long positions
- Negative weights = short positions
- Sum of weights = 1 (net exposure)
- Sum of absolute weights = gross exposure (can exceed 1)

Example: w = [0.6, 0.8, -0.4] means:
- 60% long in asset 1
- 80% long in asset 2
- 40% short in asset 3
- Net exposure: 100%
- Gross exposure: 180%

### 4. Strategies NOT Modified

The following strategies were not modified because they are either:
- Conceptually incompatible with short selling (IVP)
- Mathematically break with negative weights (Risk Parity)
- Would require significant redesign (Quintile, MDP, MDCP)

**Not supporting short selling:**
- QuintileStrategy
- InverseVolatilityStrategy
- VanillaRiskParityStrategy
- MostDiversifiedStrategy
- MaximumDecorrelationStrategy

## Testing

To test the short-selling feature:

`python
# Create a simple test
from Backtester.Strategy_Core import MarkowitzStrategy
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range('2020-01-01', periods=100)
data = pd.DataFrame({
    'Asset1': np.random.randn(100).cumsum() + 100,
    'Asset2': np.random.randn(100).cumsum() + 100,
}, index=dates)

# Test long-only
from Backtester.Strategy_Core import markowitz_portfolio_fun
weights_long, _ = markowitz_portfolio_fun(data, allow_short=False)
print(f"Long-only weights: {weights_long}")
print(f"All positive? {np.all(weights_long >= 0)}")

# Test short-allowed
weights_short, _ = markowitz_portfolio_fun(data, allow_short=True)
print(f"Short-allowed weights: {weights_short}")
print(f"Has negatives? {np.any(weights_short < 0)}")
`

## Backward Compatibility

✅ **Fully backward compatible** - all existing code continues to work unchanged because:
- llow_short defaults to False
- Existing strategy instantiations without the parameter work as before
- Long-only behavior is preserved by default

## Future Enhancements

Potential future improvements:
- Add leverage constraints (max gross exposure)
- Individual asset short limits
- Sector neutrality constraints
- Long-short dollar neutrality (sum of weights = 0)
