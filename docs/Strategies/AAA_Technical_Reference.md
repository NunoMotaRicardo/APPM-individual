# Adaptive Asset Allocation (AAA) Strategy - Technical Reference

## Overview

The Adaptive Asset Allocation (AAA) strategy is a quantitative portfolio management approach that combines **momentum-based asset selection** with **minimum-variance optimization** to construct portfolios that balance growth potential with risk management.

**Module**: `Backtester.Strategy_AAA`  
**Main Function**: `adaptive_asset_allocation_portfolio_fun()`  
**Strategy Class**: `AdaptiveAssetAllocationStrategy`  
**Base Class**: `PortfolioRebalanceStrategy` (from BacktestFramework)

---

## Algorithm Description

### Core Logic Flow

```
INPUT: Daily price DataFrame (N assets × T days)
       Parameters: momentum_top_n, momentum_lookback, covariance_lookback, etc.

STEP 1: Data Extraction
├─ Handle MultiIndex columns (extract base_column level)
├─ Identify asset names and count
└─ Validate sufficient history for momentum calculation

STEP 2: Momentum Ranking
├─ Get current price: P_t
├─ Get historical price: P_t-L (L = momentum_lookback)
├─ Calculate momentum score: M_i = P_i(t) / P_i(t-L)
├─ Drop NaN and infinite values
└─ Rank all assets by momentum score

STEP 3: Asset Selection
├─ Select top N assets: A_selected = top(M, momentum_top_n)
└─ Validate selected assets have sufficient data

STEP 4: Return Calculation
├─ For selected assets only:
│  ├─ Calculate log returns: r_i(t) = ln(P_i(t) / P_i(t-1))
│  └─ Extract most recent C observations (C = covariance_lookback)
└─ Drop assets with incomplete return histories

STEP 5: Covariance Estimation
├─ Build covariance matrix: Σ = Cov(r_valid)
└─ Dimensions: n_valid × n_valid

STEP 6: Minimum Variance Optimization
├─ Objective: min w^T Σ w
├─ Constraints: 
│  ├─ Σw_i = 1 (fully invested)
│  └─ 0 ≤ w_i ≤ 1 (long-only, no leverage)
└─ Method: SLSQP (Sequential Least Squares Programming)

STEP 7: Weight Construction
├─ If optimization successful:
│  ├─ Normalize weights to sum = 1
│  └─ Map to full universe (non-selected assets = 0)
└─ If optimization fails:
   └─ Equal-weight selected assets (fallback)

OUTPUT: weight vector w (length N, sums to 1.0), log_message string
```

---

## Mathematical Formulation

### 1. Momentum Score

For asset $i$ at time $t$:

$$M_i(t) = \frac{P_i(t)}{P_i(t - L_{mom})}$$

where:
- $P_i(t)$ = Current price of asset $i$
- $P_i(t - L_{mom})$ = Price $L_{mom}$ trading days ago
- $L_{mom}$ = `momentum_lookback` parameter (default: 132 days ≈ 6 months)
- $M_i(t)$ = Momentum score (price relative)

Higher $M_i$ indicates stronger recent performance.

### 2. Asset Selection

Sort assets by momentum and select top $N_{top}$:

$$\mathcal{A}_{selected} = \arg\max_{|\mathcal{A}| = N_{top}} \{M_1, M_2, \ldots, M_N\}$$

where:
- $N_{top}$ = `momentum_top_n` parameter (default: 5)
- $\mathcal{A}_{selected}$ = Set of selected assets

### 3. Log Returns

For each selected asset $i \in \mathcal{A}_{selected}$:

$$r_i(t) = \ln\left(\frac{P_i(t)}{P_i(t-1)}\right)$$

Extract most recent $C$ observations:

$$\mathbf{R} = \begin{bmatrix} r_1(t-C+1) & \cdots & r_n(t-C+1) \\ \vdots & \ddots & \vdots \\ r_1(t) & \cdots & r_n(t) \end{bmatrix}$$

where $C$ = `covariance_lookback` (default: 22 days ≈ 1 month).

### 4. Covariance Matrix

Estimate sample covariance matrix:

$$\boldsymbol{\Sigma} = \frac{1}{C-1} \mathbf{R}^T \mathbf{R}$$

Dimensions: $n_{valid} \times n_{valid}$ (where $n_{valid} \leq N_{top}$ after removing assets with incomplete data).

### 5. Minimum Variance Optimization

**Objective Function:**

$$\min_{\mathbf{w}} \quad \sigma^2_p = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

**Subject to:**

$$\begin{align}
\sum_{i=1}^{n_{valid}} w_i &= 1 && \text{(fully invested)} \\
0 \leq w_i &\leq 1 \quad \forall i && \text{(long-only)}
\end{align}$$

**Solution Method:** SLSQP (Sequential Least Squares Quadratic Programming) via `scipy.optimize.minimize`.

### 6. Weight Mapping

Map optimized weights back to full universe:

$$w_i^{full} = \begin{cases}
w_i^{opt} & \text{if } i \in \mathcal{A}_{selected} \text{ and valid} \\
0 & \text{otherwise}
\end{cases}$$

**Constraint verification:** $\sum_{i=1}^{N} w_i^{full} = 1$

---

## Implementation Details

### Function Signature

```python
def adaptive_asset_allocation_portfolio_fun(
    dataset: pd.DataFrame,
    base_column: str = "adjusted",
    momentum_top_n: int = 5,
    momentum_lookback: int = 6 * 22,  # 132 days ≈ 6 months
    covariance_lookback: int = 22,     # 22 days ≈ 1 month
    **kwargs
) -> tuple[np.ndarray, str]:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | DataFrame | - | Daily price data (rows=dates, cols=assets) |
| `base_column` | str | `"adjusted"` | Column level to extract from MultiIndex |
| `momentum_top_n` | int | `5` | Number of top momentum assets to select |
| `momentum_lookback` | int | `132` | Momentum calculation period (trading days) |
| `covariance_lookback` | int | `22` | Covariance estimation window (trading days) |
| `**kwargs` | dict | - | Additional arguments (passed by backtester) |

**Trading Day Conversions:**
- 1 month ≈ 22 trading days
- 3 months ≈ 66 trading days
- 6 months ≈ 132 trading days
- 12 months ≈ 252 trading days

### Return Values

| Return | Type | Description |
|--------|------|-------------|
| `weights` | np.ndarray | Portfolio weights (length = num_assets in dataset) |
| `log_message` | str | Diagnostic messages (warnings, errors, fallback notices) |

---

## Edge Cases and Fallback Logic

### 1. Insufficient History

**Condition:** `len(dataset) <= momentum_lookback`

**Action:** Return equal weights across all assets

**Reason:** Cannot compute meaningful momentum scores

### 2. Empty Momentum Series

**Condition:** All momentum scores are NaN or infinite

**Action:** Return equal weights across all assets

**Reason:** No valid momentum data after filtering

### 3. Insufficient Return History

**Condition:** `len(log_returns) < covariance_lookback` or `< 2`

**Action:** Equal-weight selected momentum assets

**Reason:** Covariance estimation requires minimum 2 observations

### 4. Assets with Incomplete Data

**Condition:** Selected assets lack sufficient recent return observations

**Action:** Remove assets with incomplete data, proceed with valid subset

**Effect:** May reduce number of assets below `momentum_top_n`

### 5. Optimization Failure

**Condition:** SLSQP solver fails to converge or raises exception

**Action:** Equal-weight selected valid assets

**Logged:** `"WARNING: AAA - optimizer failed; using equal weights among selected assets."`

### 6. Invalid Optimized Weights

**Condition:** Optimizer returns negative weights or sum ≤ 0

**Action:** Equal-weight selected valid assets

**Reason:** Numerical instability or constraint violation

---

## Optimization Algorithm: SLSQP

### Method Details

**Full Name:** Sequential Least Squares Quadratic Programming

**Source:** `scipy.optimize.minimize(method='SLSQP')`

**Characteristics:**
- Gradient-based optimization
- Handles equality and inequality constraints
- Robust for small-scale quadratic programs (n ≤ 100)
- Iterative refinement of solution

### Initial Guess

Equal weights among selected assets:

$$w_i^{(0)} = \frac{1}{n_{valid}} \quad \forall i$$

### Bounds

Element-wise constraints for each asset:

$$0.0 \leq w_i \leq 1.0 \quad \forall i \in \{1, \ldots, n_{valid}\}$$

### Constraints

Single equality constraint:

```python
{
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1.0
}
```

Enforces: $\sum_{i} w_i = 1$

### Convergence Criteria

Default SLSQP tolerances:
- Function tolerance: `ftol = 1e-6`
- Constraint violation tolerance: `1e-6`
- Maximum iterations: 100

**Success Indicator:** `result.success == True`

---

## Numerical Stability Considerations

### 1. Covariance Matrix Conditioning

**Potential Issue:** Singular or near-singular $\boldsymbol{\Sigma}$ when:
- Assets are highly correlated ($\rho > 0.99$)
- Insufficient observations ($C < n_{valid}$)
- Zero-variance assets in estimation window

**Mitigation:**
- Use `covariance_lookback = 22` (default) to ensure $C \geq n_{valid}$
- SLSQP's built-in regularization handles moderate ill-conditioning
- Fallback to equal weights on optimization failure

### 2. Log Return Calculation

**Potential Issue:** $\ln(x)$ undefined for $x \leq 0$

**Mitigation:**
- Assumes positive prices (standard in financial data)
- `.dropna()` removes any resulting NaN values

### 3. Momentum Score Filtering

**Potential Issue:** Division by zero when $P_i(t - L_{mom}) = 0$

**Mitigation:**
```python
momentum_scores = (current_prices / past_prices).replace([np.inf, -np.inf], np.nan).dropna()
```
Replaces infinities with NaN, then drops invalid scores.

### 4. Weight Normalization

**Potential Issue:** Optimizer may return weights that sum to 0.999999 due to numerical precision

**Mitigation:**
```python
optimal_weights = optimal_weights / optimal_weights.sum()
```
Enforces exact sum-to-one after optimization.

---

## Data Requirements

### Minimum History

| Calculation | Required Length |
|-------------|----------------|
| Momentum | `momentum_lookback + 1` days |
| Returns | `covariance_lookback + 1` days |
| **Total Minimum** | `max(momentum_lookback, covariance_lookback) + 1` |

**Example (default parameters):**
- `momentum_lookback = 132` → need 133 days
- `covariance_lookback = 22` → need 23 days
- **Required:** 133 days of price history

### Data Format

**Expected Structure:**

```python
# Single-level columns (preferred)
dataset = pd.DataFrame({
    'AAPL': [150.0, 151.2, ...],
    'MSFT': [300.0, 302.5, ...],
    'GOOGL': [2800.0, 2820.0, ...]
}, index=pd.DatetimeIndex([...]))

# Multi-level columns (handled automatically)
dataset = pd.DataFrame({
    ('AAPL', 'adjusted'): [150.0, 151.2, ...],
    ('AAPL', 'close'): [150.5, 151.7, ...],
    ('MSFT', 'adjusted'): [300.0, 302.5, ...],
    ('MSFT', 'close'): [301.0, 303.0, ...]
})
```

**Column Extraction:**
- If MultiIndex detected, function extracts `base_column` level (default: `'adjusted'`)
- Falls back to full dataset if extraction fails

---

## Integration with BacktestFramework

### Strategy Class Definition

```python
class AdaptiveAssetAllocationStrategy(PortfolioRebalanceStrategy):
    """AAA strategy using momentum selection + minimum variance weighting."""
    
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = adaptive_asset_allocation_portfolio_fun
```

### Backtest Configuration

Parameters passed via `portfolio_func_kwargs`:

```python
strategies = {
    'AAA_Strategy': (AdaptiveAssetAllocationStrategy, {
        'rebalance_every': 21,          # Rebalance monthly
        'lookback': 132,                # Need 6 months history
        'portfolio_func_kwargs': {
            'base_column': 'adjusted',
            'momentum_top_n': 5,
            'momentum_lookback': 132,
            'covariance_lookback': 22
        }
    })
}
```

### Execution Flow

1. **Backtester Initialization**
   - Strategy registered with `PortfolioRebalanceStrategy` base class
   - `portfolio_func` pointer set to AAA function

2. **Rebalance Trigger**
   - Every `rebalance_every` days (e.g., 21 = monthly)
   - Backtester extracts last `lookback` days of data
   - Passes dataset window to `portfolio_func`

3. **Weight Calculation**
   - `adaptive_asset_allocation_portfolio_fun()` called
   - Returns weights array and log message
   - Backtester applies weights to current portfolio

4. **Order Execution**
   - Orders generated to rebalance from current positions to target weights
   - Considers transaction costs (if configured)

5. **Logging**
   - `log_message` appended to backtest log
   - Warnings visible in results for debugging

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Dominant Factor |
|-----------|------------|-----------------|
| Momentum Calculation | $O(N \cdot L_{mom})$ | N = num_assets |
| Sorting (selection) | $O(N \log N)$ | Sorting all assets |
| Return Calculation | $O(n_{top} \cdot C)$ | $n_{top}$ = selected assets |
| Covariance Estimation | $O(C \cdot n_{valid}^2)$ | Matrix multiplication |
| Optimization | $O(n_{valid}^3)$ | Quadratic program |
| **Total** | $O(N \cdot L_{mom} + n_{valid}^3)$ | For typical N=20, $n_{valid}$=5 |

**Typical Execution Time:**
- N = 20 assets, $L_{mom}$ = 132, C = 22, $n_{valid}$ = 5
- **~10-50 milliseconds** per rebalance on modern hardware

### Memory Usage

| Component | Size |
|-----------|------|
| Dataset | $N \times L_{mom} \times 8$ bytes (float64) |
| Returns Matrix | $n_{valid} \times C \times 8$ bytes |
| Covariance Matrix | $n_{valid}^2 \times 8$ bytes |
| **Total (estimate)** | $\sim$ 1-10 MB for typical parameters |

---

## Comparison to Other Strategies

| Feature | AAA | GEM | HRPP | DRPP |
|---------|-----|-----|------|------|
| **Selection Mechanism** | Momentum ranking | Momentum consistency | Hierarchical risk parity | Risk parity with dual momentum |
| **Weighting Method** | Minimum variance | Equal weight | Risk-based (HRP) | Risk-based (inverse volatility) |
| **Asset Count** | Top N (e.g., 5) | Variable (can go to cash) | All assets | Variable |
| **Optimization** | Quadratic program | None | Hierarchical clustering | None |
| **Defensive Position** | Holds selected assets | Cash/Treasury | Full universe (rebalanced) | Cash/Treasury |
| **Rebalance Frequency** | Monthly (typical) | Monthly | Monthly/Quarterly | Monthly |
| **Risk Focus** | Variance minimization | Downside protection | Diversification | Balanced risk |

---

## References

### Academic Background

1. **Minimum Variance Portfolio**
   - Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77-91.

2. **Momentum Investing**
   - Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*, 48(1), 65-91.

3. **Adaptive Asset Allocation**
   - Butler, A., Philbrick, M., & Gordillo, P. (2012). "Adaptive Asset Allocation." *SSRN Working Paper*.

### Implementation Resources

- **Backtrader Documentation**: https://www.backtrader.com/docu/
- **SciPy Optimization Guide**: https://docs.scipy.org/doc/scipy/reference/optimize.html
- **pandas MultiIndex**: https://pandas.pydata.org/docs/user_guide/advanced.html

---

## Troubleshooting Guide

### Issue: Weights concentrate on single asset

**Symptoms:** One weight ≈ 1.0, others ≈ 0.0

**Causes:**
- Selected assets highly correlated
- One asset has much lower variance
- Short `covariance_lookback` captures transient low volatility

**Solutions:**
1. Increase `covariance_lookback` (e.g., 44 days instead of 22)
2. Add maximum weight constraint to optimization
3. Regularize covariance matrix: $\boldsymbol{\Sigma}_{reg} = \boldsymbol{\Sigma} + \lambda \mathbf{I}$

### Issue: Optimization fails frequently

**Symptoms:** Many "optimizer failed" warnings in logs

**Causes:**
- Singular covariance matrix
- Extreme asset correlations
- Numerical precision issues

**Solutions:**
1. Increase `covariance_lookback` (more observations)
2. Check for duplicate or highly correlated assets in universe
3. Consider CVXPY-based solver (more robust)

### Issue: Performance lags benchmark

**Symptoms:** AAA returns < equal-weight or buy-and-hold

**Causes:**
- Momentum reversal (strategy chases past winners)
- High rebalancing costs
- Short momentum lookback captures noise

**Solutions:**
1. Increase `momentum_lookback` (e.g., 252 days = 12 months)
2. Reduce rebalancing frequency (e.g., quarterly instead of monthly)
3. Add transaction cost analysis to backtest
4. Test with longer historical period (5+ years)

### Issue: Large drawdowns during corrections

**Symptoms:** Strategy loses significant value in bear markets

**Causes:**
- AAA does not include cash/defensive position
- All selected assets may decline together

**Solutions:**
1. Combine with absolute momentum filter (like GEM)
2. Include bond/treasury ETFs in asset universe
3. Add maximum drawdown stop-loss at backtester level
4. Consider hybrid AAA+GEM approach

---

**Document Version:** 1.0  
**Last Updated:** November 5, 2025  
**Author:** Technical documentation for AAA strategy  
**Module:** `Backtester.Strategy_AAA`
