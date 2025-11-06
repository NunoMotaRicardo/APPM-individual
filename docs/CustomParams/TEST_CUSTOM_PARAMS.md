# Example: Testing Custom Parameters Implementation

## Quick Test Script

`python
# Test that parameters flow correctly
from Backtester.Strategy_DRPP import DownsideRiskParityStrategy
from Backtester.Strategy_HRPP import HierarchicalRiskParityStrategy

# Define strategies with custom parameters
test_strategies = {
    # DRPP with CVXPY method
    "DRPP_cvxpy": (DownsideRiskParityStrategy, {
        'buy_slippage_buffer': 0.04,
        'portfolio_func_kwargs': {
            'method': 'cvxpy',
            'threshold': 0.0,
        }
    }),
    
    # DRPP with iterative method
    "DRPP_iter": (DownsideRiskParityStrategy, {
        'buy_slippage_buffer': 0.04,
        'portfolio_func_kwargs': {
            'method': 'iterative',
            'threshold': -0.01,  # Different threshold
        }
    }),
    
    # HRPP with Ward linkage
    "HRPP_ward": (HierarchicalRiskParityStrategy, {
        'buy_slippage_buffer': 0.04,
        'portfolio_func_kwargs': {
            'linkage_method': 'ward'
        }
    }),
    
    # HRPP with Single linkage
    "HRPP_single": (HierarchicalRiskParityStrategy, {
        'buy_slippage_buffer': 0.04,
        'portfolio_func_kwargs': {
            'linkage_method': 'single'
        }
    }),
}

# Run backtest
bt_backtest = BacktraderPortfolioBacktest(
    strategies=test_strategies,
    datasets=my_dataset_list,
    rebalance_every=15,
    lookback=30,
)

results = bt_backtest.run_backtest()
`

## Verification

Each strategy variant will:
1. Receive its own portfolio_func_kwargs dictionary
2. Pass those kwargs to the portfolio optimization function
3. Use the specified parameters (method, threshold, linkage_method)
4. Generate unique results based on those parameters

## Expected Behavior

- DRPP_cvxpy and DRPP_iter should produce different weights (different optimization methods)
- HRPP_ward and HRPP_single should produce different weights (different clustering methods)
- All variants are tracked separately in results dictionary

