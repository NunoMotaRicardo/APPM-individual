# Hierarchical Risk Parity Portfolio (HRPP) - Implementation Summary

## Overview
Successfully implemented the **Hierarchical Risk Parity Portfolio (HRPP)** strategy based on López de Prado's approach, following the methodology demonstrated in the Lab6 notebooks.

## Files Modified

### 1. Backtester/weights_calculators.py
- **Added scipy imports**: rom scipy.cluster.hierarchy import dendrogram, linkage and rom scipy.spatial.distance import squareform
- **Added main function**: hierarchical_risk_parity_portfolio_fun()
- **Added helper functions**:
  - _corr_to_dist(): Converts correlation matrix to distance matrix using López de Prado's metric
  - _ivp_weights(): Inverse Variance Portfolio weights for sub-clusters
  - _cluster_var(): Calculate cluster variance using IVP
  - _hrpp_weights(): Core HRPP algorithm implementation

### 2. Backtester/BacktestFramework.py
- **Added strategy class**: HierarchicalRiskParityStrategy
- Inherits from PortfolioRebalanceStrategy
- Uses hierarchical_risk_parity_portfolio_fun as portfolio function

## Algorithm Details

### Hierarchical Risk Parity (HRPP) Algorithm

1. **Distance Calculation**: Convert correlation matrix to distance using d_ij = √(0.5 × (1 - ρ_ij))

2. **Hierarchical Clustering**: Perform clustering using linkage method (default: 'single')

3. **Quasi-Diagonalization**: Reorder assets based on dendrogram leaf ordering

4. **Recursive Bisection**:
   - Start with all assets as one cluster
   - Recursively split clusters into two halves
   - Calculate variance for each sub-cluster using Inverse Variance Portfolio (IVP)
   - Allocate weights inversely proportional to cluster variance: α = 1 - v_left / (v_left + v_right)
   - Continue until individual assets are reached

5. **Weight Normalization**: Map weights back to original asset order and normalize

## Parameters

- **dataset**: Price data (DataFrame)
- **base_column**: Column to extract from multi-level columns (default: 'adjusted')
- **linkage_method**: Clustering method (default: 'single', options: 'complete', 'average', 'ward')

## Robustness Features

- **Data validation**: Checks for sufficient historical data (min 2 periods)
- **Matrix validation**: Checks for NaN values in correlation/covariance matrices
- **Fallback mechanism**: Returns equal weights if optimization fails
- **Error logging**: Returns log messages for debugging

## Usage Example

\\\python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest, HierarchicalRiskParityStrategy

# Define strategies to test
strategies = {
    'Vanilla Risk Parity': VanillaRiskParityStrategy,
    'Hierarchical Risk Parity': HierarchicalRiskParityStrategy,
    'Downside Risk Parity': DownsideRiskParityStrategy
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
\\\

## Testing

Tested successfully with synthetic data:
- 4 assets, 100 days of price history
- Weights sum to 1.0 exactly
- No errors or warnings
- Produces diversified portfolio weights

## Key Advantages

1. **No estimation error**: Doesn't require expected returns estimation
2. **Hierarchical diversification**: Naturally diversifies across asset clusters
3. **Robust**: More stable than mean-variance optimization
4. **Interpretable**: Tree structure shows clustering relationships

## Comparison with Other Strategies

- **vs. Equal Weight**: HRPP considers correlations and volatilities
- **vs. Risk Parity**: HRPP uses hierarchical structure, more robust to estimation error
- **vs. Markowitz**: HRPP doesn't require expected returns, avoids corner solutions

## References

- López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample
- Implemented based on Lab6 HRPP notebooks in the aulas folder

## Status

✅ Implementation complete
✅ Function tested successfully
✅ Strategy class created
✅ Ready for backtesting
