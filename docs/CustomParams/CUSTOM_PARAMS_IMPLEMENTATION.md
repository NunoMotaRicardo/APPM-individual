# Custom Portfolio Function Parameters Implementation

## Overview
This document describes the implementation of custom parameters for portfolio optimization strategies, specifically for DRPP (Downside Risk Parity Portfolio) and HRPP (Hierarchical Risk Parity Portfolio).

## Problem Statement
Previously, custom parameters (like method, 	hreshold, linkage_method) were defined at the strategy instantiation level but were not being passed through to the portfolio optimization functions, causing them to use default values instead.

## Solution

### 1. Framework Changes (BacktestFramework.py)

#### Added portfolio_func_kwargs parameter to strategy params
`python
params: Any = dict(
    rebalance_every=21,
    lookback=252 // 4,
    warmup=None,
    portfolio_func=None,
    dataset_assets=None,
    buy_slippage_buffer=0.01,
    verbose=False,
    portfolio_func_kwargs={},  # NEW: Custom kwargs for portfolio function
)
`

#### Modified calculate_target_weights() to pass kwargs
The portfolio function is now called with unpacked kwargs:
`python
portfolio_func_kwargs = self.params.portfolio_func_kwargs or {}
weights, log_message = self.params.portfolio_func(portfolio_data, **portfolio_func_kwargs)
`

#### Removed global linkage_method parameter
The linkage_method parameter was removed from BacktraderPortfolioBacktest.__init__() as it should be strategy-specific, not global.

### 2. Notebook Usage Pattern (NR-2-RunBacktest.ipynb)

Custom parameters are now passed via the portfolio_func_kwargs dictionary:

`python
bt_strategies = {
    "DRPP": (DownsideRiskParityStrategy, {
        'buy_slippage_buffer': 0.04,
        'portfolio_func_kwargs': {
            'method': 'cvxpy',      # DRPP-specific parameter
            'threshold': 0.0,       # DRPP-specific parameter
        }
    }),
    "HRPP_ward": (HierarchicalRiskParityStrategy, {
        'buy_slippage_buffer': 0.04,
        'portfolio_func_kwargs': {
            'linkage_method': 'ward'  # HRPP-specific parameter
        }
    }),
    "HRPP_single": (HierarchicalRiskParityStrategy, {
        'buy_slippage_buffer': 0.04,
        'portfolio_func_kwargs': {
            'linkage_method': 'single'  # Different linkage method
        }
    }),
}
`

### 3. Multiple Variants of Same Strategy
You can now easily test multiple configurations of the same strategy by giving them different names:
- HRPP_ward, HRPP_single, HRPP_complete, HRPP_average

Each variant will be tracked separately in the results.

## Benefits

1. **Flexibility**: Each strategy instance can have its own custom parameters
2. **Clarity**: Clear separation between strategy-level params and optimization function params
3. **Extensibility**: Easy to add new custom parameters without modifying the framework
4. **Multiple Variants**: Test different parameter configurations side-by-side

## Parameter Flow

`
Notebook Strategy Definition
    ↓
BacktraderPortfolioBacktest.run_single_backtest()
    ↓
cerebro.addstrategy(strategy_class, **strat_params)
    ↓
PortfolioRebalanceStrategy.__init__()
    ↓ (stores in self.params.portfolio_func_kwargs)
PortfolioRebalanceStrategy.calculate_target_weights()
    ↓
portfolio_func(portfolio_data, **portfolio_func_kwargs)
    ↓
DRPP/HRPP portfolio function receives custom parameters
`

## Existing Strategy Compatibility

Strategies without custom parameters continue to work without changes:
`python
"IVP": (InverseVolatilityStrategy, {
    'buy_slippage_buffer': 0.04,
})
# No portfolio_func_kwargs needed - uses defaults
`

## Testing Recommendations

1. Test DRPP with both method='cvxpy' and method='iterative'
2. Test HRPP with all linkage methods: 'single', 'complete', 'average', 'ward'
3. Test DRPP with different threshold values (e.g., 0.0, -0.01, 0.01)
4. Verify results differ between configurations

## Notes

- The portfolio_func_kwargs dict is passed directly to the portfolio function via **kwargs
- Portfolio functions must accept **kwargs to be compatible (all current ones do)
- Empty dict {} is safe default - no parameters passed if not specified
- Parameters are strategy-instance specific, not global

