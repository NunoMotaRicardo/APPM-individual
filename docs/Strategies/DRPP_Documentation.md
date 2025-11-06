# Downside Risk Parity Portfolio (DRPP) - Documentation

## Overview
The **Downside Risk Parity Portfolio (DRPP)** strategy is a variant of the traditional Risk Parity approach that focuses exclusively on downside risk rather than total volatility. This is analogous to how the Sortino ratio improves upon the Sharpe ratio by considering only downside deviation.

## Motivation

### Traditional Risk Parity Limitation
Traditional Risk Parity allocates capital so that each asset contributes equally to portfolio risk (variance). However, this approach treats upside and downside volatility equally, which is not aligned with investor preferences:
- Investors don''t mind upside volatility (gains)
- Investors care primarily about downside volatility (losses)

### Downside Risk Parity Solution
DRPP addresses this by:
1. Using **semi-covariance matrix** (only negative returns) instead of full covariance
2. Equalizing **downside risk contributions** rather than total risk contributions
3. Providing better downside protection while allowing upside potential

## Mathematical Formulation

### Downside Semi-Covariance Matrix
For returns r_t, define the downside semi-covariance between assets i and j:

Σ_downside[i,j] = E[(r_i - threshold)^- * (r_j - threshold)^-]

where (x)^- = min(x, 0) keeps only negative deviations.

Default threshold = 0 (mean return), but can be customized.

### Risk Contribution Equation
For portfolio weights w, the downside risk contribution of asset i:

DRC_i = w_i * (Σ_downside * w)_i

### Optimization Objective
Find weights w such that:
- DRC_1 = DRC_2 = ... = DRC_n = (1/n) * Portfolio Downside Risk
- w_i >= 0 (long-only)
- Σw_i = 1 (fully invested)

## Implementation Details

### Two Solution Methods

1. **CVXPY (Default, Recommended)**
   - Uses Sequential Convex Programming (SCP)
   - Linearizes non-convex risk contributions iteratively
   - More robust and guaranteed convergence
   - Better for production use

2. **Iterative Newton-Raphson**
   - Faster but less stable
   - Uses cyclical coordinate descent with dampening
   - Good for quick experiments

### Key Parameters
```python
downside_risk_parity_portfolio_fun(
    dataset,              # Price data DataFrame
    base_column="adjusted",  # Price column to use
    method="cvxpy",       # Optimization method
    threshold=0.0,        # Downside threshold (0 = negative returns)
    max_iter=2000,        # Max iterations for convergence
    tol=1e-6             # Convergence tolerance
)
```

### Robustness Features
- **Regularization**: Adds small identity matrix to ensure PSD
- **Fallback mechanisms**: Returns equal weights if optimization fails
- **PSD checking**: Validates semi-covariance matrix is positive semi-definite
- **Zero volatility handling**: Detects and handles zero downside volatility
- **Convergence monitoring**: Tracks solution quality via coefficient of variation

## Usage Example

### Basic Usage
```python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest, DownsideRiskParityStrategy

# Define strategies to test
strategies = {
    ''Risk Parity'': VanillaRiskParityStrategy,
    ''Downside Risk Parity'': DownsideRiskParityStrategy
}

# Run backtest
backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=dataset_list,
    rebalance_every=63,  # Quarterly
    lookback=252,        # 1 year lookback
    warmup=252          # 1 year warmup
)

results = backtest.run_backtest()
```

### Custom Threshold Example
```python
# Use custom threshold (e.g., risk-free rate or target return)
from Backtester.weights_calculators import downside_risk_parity_portfolio_fun

def custom_drpp(dataset, **kwargs):
    return downside_risk_parity_portfolio_fun(
        dataset, 
        threshold=0.02/252,  # 2% annual threshold, daily
        method="cvxpy"
    )

# Create custom strategy class
class CustomDRPPStrategy(PortfolioRebalanceStrategy):
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = custom_drpp
```

## Expected Behavior

### Compared to Traditional Risk Parity
- **More conservative allocation** to high-volatility downside assets
- **Better downside protection** in bear markets
- **Potentially lower Sharpe ratio** but higher Sortino ratio
- **Different weight distribution** when upside/downside asymmetry exists

### Performance Characteristics
- **Sortino Ratio**: Expected to be higher than traditional RP
- **Maximum Drawdown**: Expected to be lower than traditional RP
- **Upside Capture**: May be lower (more conservative)
- **Downside Capture**: Expected to be better (main objective)

## Technical Notes

### Computational Complexity
- Similar to traditional Risk Parity: O(n²) per iteration
- Typically converges in 10-30 iterations
- CVXPY method: ~0.5-2 seconds for 20-50 assets
- Iterative method: ~0.1-0.5 seconds for 20-50 assets

### Edge Cases Handled
1. **No downside data**: Falls back to regular covariance
2. **Singular matrix**: Adds regularization
3. **Zero downside volatility**: Returns equal weights
4. **Non-convergence**: Validates solution quality before returning

### Limitations
1. **Estimation error**: Downside covariance harder to estimate (fewer observations)
2. **Look-ahead bias**: Uses historical downside, may not predict future
3. **Regime dependency**: Works best in volatile markets with clear downside
4. **Optimization difficulty**: Non-convex problem, local optima possible

## Theoretical Foundation

### Related Literature
- **Sortino & van der Meer (1991)**: Downside risk measures
- **Estrada (2007)**: Mean-semivariance optimization
- **Maillard, Roncalli & Teiletche (2010)**: Risk parity portfolios
- **Choueifaty et al. (2013)**: Properties of risk-based portfolios

### Connection to Other Strategies
- **Vanilla Risk Parity**: DRPP with threshold → -∞
- **Inverse Volatility**: DRPP without correlation structure
- **Minimum CVaR**: DRPP is continuous approximation

## File Locations
- **Weight Calculator**: `Backtester/weights_calculators.py` (lines 647-868)
- **Strategy Class**: `Backtester/BacktestFramework.py` (lines 879-885)

## Author Notes
Created as part of AssetPricingPortfolio research framework.
Follows the same architecture as existing risk-based strategies.
Designed for academic finance studies on downside risk management.

---
Last Updated: 2024-10-16
