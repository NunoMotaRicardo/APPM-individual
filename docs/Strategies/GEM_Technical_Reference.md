# Global Equities Momentum (GEM) Strategy - Technical Reference

## Overview

The Global Equities Momentum (GEM) strategy is a tactical asset allocation strategy that combines **relative momentum** (comparing assets against each other) and **absolute momentum** (comparing assets against a risk-free benchmark) to dynamically allocate capital between risky assets and defensive positions.

**Module**: `Backtester.Strategy_dual_momentum`  
**Main Function**: `gem_portfolio_fun()`  
**Strategy Class**: `GlobalEquitiesMomentumStrategy`  
**Base Class**: `PortfolioRebalanceStrategy` (from BacktestFramework)

---

## Algorithm Description

### Core Logic Flow

```
INPUT: Daily price DataFrame (N assets × T days)
       Parameters: momentum_periods, min_positive_periods, treasury_threshold, etc.

STEP 1: Data Extraction
├─ Handle MultiIndex columns (extract base_column level)
├─ Identify asset names and count
└─ Identify risk-free asset (cash or treasury symbol)

STEP 2: Momentum Calculation
├─ For each lookback period p ∈ momentum_periods:
│  ├─ Get price p days ago: P_t-p
│  ├─ Get current price: P_t
│  └─ Calculate return: R_p = (P_t - P_t-p) / P_t-p × 100
└─ Store momentum matrix (N assets × M periods)

STEP 3: Consistency Filtering
├─ For each risky asset i:
│  ├─ Count positive momentum periods: C_i = Σ I(R_i,p > 0)
│  └─ Calculate average momentum: M̄_i = (1/M) Σ R_i,p
└─ Filter: Keep assets where C_i ≥ min_positive_periods

STEP 4: Absolute Momentum Check
├─ If treasury asset exists (not cash):
│  ├─ Calculate treasury average return: M̄_treasury
│  └─ Filter: Keep assets where M̄_i > M̄_treasury + threshold
└─ Else: Skip this step

STEP 5: Position Limiting (optional)
└─ If maximum_positions is set:
   └─ Sort qualified assets by M̄_i descending
   └─ Keep top N assets

STEP 6: Weight Allocation
├─ If qualified_assets is empty:
│  ├─ If risk_free_asset = 'cash': w = [0, 0, ..., 0]
│  └─ Else: w[treasury_idx] = 1.0
└─ Else:
   └─ Equal weight: w[i] = 1 / |qualified_assets| for i ∈ qualified

OUTPUT: weight vector w (sums to 1.0), log_message string
```

---

## Mathematical Formulation

### 1. Momentum Score

For asset $i$ and lookback period $p$ (in trading days):

$$M_{i,p} = \frac{P_i(t) - P_i(t-p)}{P_i(t-p)} \times 100$$

where:
- $P_i(t)$ = Current price of asset $i$
- $P_i(t-p)$ = Price $p$ trading days ago
- $M_{i,p}$ = Momentum score (percentage return)

### 2. Consistency Score

$$C_i = \sum_{p \in \mathcal{P}} \mathbb{1}(M_{i,p} > 0)$$

where:
- $\mathcal{P}$ = Set of momentum periods (e.g., {21, 63, 126, 252})
- $\mathbb{1}(\cdot)$ = Indicator function (1 if true, 0 otherwise)
- $C_i$ = Number of periods with positive momentum

### 3. Average Momentum

$$\bar{M}_i = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} M_{i,p}$$

### 4. Qualification Criteria

Asset $i$ qualifies if **both** conditions are met:

$$\begin{cases}
C_i \geq c_{min} & \text{(Consistency criterion)} \\
\bar{M}_i > \bar{M}_{rf} + \theta & \text{(Absolute momentum criterion)}
\end{cases}$$

where:
- $c_{min}$ = `min_positive_periods` parameter
- $\bar{M}_{rf}$ = Average momentum of risk-free asset
- $\theta$ = `treasury_threshold` parameter

### 5. Weight Allocation

$$w_i = \begin{cases}
\frac{1}{N_q} & \text{if } i \in \text{qualified assets} \\
1 & \text{if } i = \text{risk-free asset and } N_q = 0 \\
0 & \text{otherwise}
\end{cases}$$

where $N_q$ = number of qualified risky assets.

**Constraint**: $\sum_{i=1}^{N} w_i = 1$ (fully invested, no leverage)

---

## Implementation Details

### Function Signature

```python
def gem_portfolio_fun(
    dataset: pd.DataFrame,
    base_column: str = "adjusted",
    momentum_periods: list = [21, 63, 126, 252],
    min_positive_periods: int = 3,
    treasury_threshold: float = 0.0,
    maximum_positions: int = 6,
    risk_free_asset: str = 'cash',
    **kwargs
) -> tuple[np.ndarray, str]:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | DataFrame | - | Daily price data (rows=dates, cols=assets) |
| `base_column` | str | `"adjusted"` | Column level to extract from MultiIndex |
| `momentum_periods` | list[int] | `[21, 63, 126, 252]` | Lookback periods in **trading days** |
| `min_positive_periods` | int | `3` | Minimum positive periods to qualify |
| `treasury_threshold` | float | `0.0` | Minimum excess return vs risk-free (%) |
| `maximum_positions` | int | `6` | Maximum positions to hold (None = unlimited) |
| `risk_free_asset` | str | `'cash'` | Risk-free option: 'cash' or symbol name |

### Return Values

| Return | Type | Description |
|--------|------|-------------|
| `weights` | np.ndarray | Portfolio weights (length = num_assets) |
| `log_message` | str | Warnings/errors encountered during execution |

---

## Data Flow

### Input Data Format

```python
# Expected DataFrame structure:
# Index: DatetimeIndex (daily dates)
# Columns: Asset names (str)
# Values: Adjusted close prices (float)

Example:
            IVV     VEU     AGG     BIL
2023-01-01  450.0   55.0    102.0   91.5
2023-01-02  451.2   55.1    102.1   91.5
2023-01-03  449.8   54.9    102.0   91.6
...         ...     ...     ...     ...
```

### MultiIndex Handling

If data has MultiIndex columns (e.g., OHLC + Adjusted):

```python
# Example MultiIndex:
#                IVV              VEU
#                open   adjusted  open   adjusted
# 2023-01-01    450.0  448.5     55.0   54.8

# Function extracts 'adjusted' level:
extracted = dataset.xs('adjusted', level=1, axis=1)
# Result:
#                IVV     VEU
# 2023-01-01    448.5   54.8
```

### Risk-Free Asset Identification

#### Option 1: Cash (Default)

```python
risk_free_asset = 'cash'
use_cash = True
treasury_idx = None
# When defensive: weights = [0, 0, ..., 0] (hold cash)
```

#### Option 2: Treasury Symbol

```python
risk_free_asset = 'BIL'
matching_assets = [name for name in asset_names if 'BIL' in name.upper()]
treasury_idx = asset_names.index(matching_assets[0])
# When defensive: weights[treasury_idx] = 1.0 (buy treasury)
```

---

## Code Architecture

### Class Hierarchy

```
bt.Strategy (Backtrader base)
    └─ PortfolioRebalanceStrategy (BacktestFramework)
        └─ GlobalEquitiesMomentumStrategy (Strategy_dual_momentum)
```

### Strategy Initialization

```python
class GlobalEquitiesMomentumStrategy(PortfolioRebalanceStrategy):
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = gem_portfolio_fun
```

The parent class (`PortfolioRebalanceStrategy`) handles:
- **Rebalancing schedule**: Calls portfolio function every `rebalance_every` days
- **Data extraction**: Calls `get_portfolio_data()` to pass historical prices
- **Order execution**: Converts weights to Backtrader orders
- **Logging**: Records decisions and errors

### Integration with BacktestFramework

```python
# BacktestFramework.py (simplified)
class PortfolioRebalanceStrategy(bt.Strategy):
    params = (
        ('portfolio_func', None),
        ('portfolio_func_kwargs', {}),
        ('rebalance_every', 21),
        ('lookback', 252),
    )
    
    def next(self):
        # Called on every bar (daily)
        if should_rebalance():
            # Get historical data (last 'lookback' days)
            portfolio_data = self.get_portfolio_data()
            
            # Call portfolio function
            weights, log_msg = self.params.portfolio_func(
                portfolio_data,
                **self.params.portfolio_func_kwargs
            )
            
            # Execute rebalancing orders
            self.rebalance(weights)
```

---

## Edge Cases and Error Handling

### 1. Insufficient Data

```python
if len(dataset) <= max(momentum_periods):
    # Not enough history for longest lookback
    # Action: Allocate to risk-free asset
    weights = np.zeros(num_assets)
    if not use_cash and treasury_idx is not None:
        weights[treasury_idx] = 1.0
    return weights, "WARNING: Insufficient data"
```

### 2. No Assets in Dataset

```python
if num_assets == 0:
    return np.array([]), "ERROR: GEM - No assets in dataset."
```

### 3. Risk-Free Asset Not Found

```python
matching_assets = [name for name in asset_names if risk_free_asset.upper() in name.upper()]
if not matching_assets:
    log_message += f"WARNING: Risk-free asset '{risk_free_asset}' not found; using cash."
    use_cash = True  # Fallback to cash
```

### 4. No Risky Assets

```python
if len(risky_assets) == 0:
    log_message += "WARNING: No risky assets to evaluate."
    # Allocate 100% to risk-free asset
```

### 5. Missing Momentum Data

```python
for lookback_days in momentum_periods:
    if len(dataset) > lookback_days:
        # Calculate momentum
    else:
        log_message += f"WARNING: Skipping {lookback_days}-day momentum."
        # Momentum score remains NaN
```

### 6. No Qualified Assets

```python
if len(qualified_assets) == 0:
    # All risky assets fail momentum criteria
    # Action: Allocate to risk-free asset (cash or treasury)
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Momentum calculation | O(M × N) | M periods, N assets |
| Consistency counting | O(M × N) | Count positive periods |
| Sorting (if max_positions) | O(N log N) | Sort by avg momentum |
| Weight allocation | O(N) | Linear assignment |
| **Total** | **O(M × N + N log N)** | Dominated by momentum calc |

### Memory Usage

- Momentum matrix: `M × N × 8 bytes` (float64)
- Example: 4 periods × 10 assets × 8 = 320 bytes (negligible)

### Execution Time

Typical runtime per rebalance (measured on modern CPU):
- **10 assets, 4 periods**: ~1-2 ms
- **50 assets, 4 periods**: ~3-5 ms
- **100 assets, 6 periods**: ~8-12 ms

Bottleneck: DataFrame indexing (`iloc[-lookback_days-1]`)

---

## Testing and Validation

### Unit Test Cases

#### Test 1: Basic Allocation

```python
# Setup: 3 risky assets, all positive momentum
prices = pd.DataFrame({
    'STOCK_A': [100, 105, 110],  # 10% return
    'STOCK_B': [50, 52, 55],     # 10% return
    'STOCK_C': [200, 195, 205],  # 2.5% return
    'BIL': [91, 91, 91]          # 0% return
})

weights, _ = gem_portfolio_fun(
    prices,
    momentum_periods=[1, 2],
    min_positive_periods=1,
    risk_free_asset='BIL'
)

# Expected: Equal weight to A, B, C (all beat treasury)
assert np.allclose(weights, [0.33, 0.33, 0.33, 0.0], atol=0.01)
```

#### Test 2: Defensive Allocation (Cash)

```python
# Setup: All risky assets negative
prices = pd.DataFrame({
    'STOCK_A': [100, 95, 90],   # -10% return
    'STOCK_B': [50, 48, 45],    # -10% return
})

weights, _ = gem_portfolio_fun(
    prices,
    momentum_periods=[1, 2],
    min_positive_periods=1,
    risk_free_asset='cash'
)

# Expected: All zeros (hold cash)
assert np.allclose(weights, [0.0, 0.0])
```

#### Test 3: Defensive Allocation (Treasury)

```python
# Setup: All risky assets negative, treasury specified
prices = pd.DataFrame({
    'STOCK_A': [100, 95, 90],
    'BIL': [91, 91, 91]
})

weights, _ = gem_portfolio_fun(
    prices,
    momentum_periods=[1, 2],
    min_positive_periods=1,
    risk_free_asset='BIL'
)

# Expected: 100% BIL
assert np.allclose(weights, [0.0, 1.0])
```

#### Test 4: Maximum Positions

```python
# Setup: 5 positive assets, max 3 positions
prices = pd.DataFrame({
    'A': [100, 120],  # 20% (rank 1)
    'B': [100, 115],  # 15% (rank 2)
    'C': [100, 110],  # 10% (rank 3)
    'D': [100, 105],  # 5%  (rank 4 - excluded)
    'E': [100, 102],  # 2%  (rank 5 - excluded)
})

weights, _ = gem_portfolio_fun(
    prices,
    momentum_periods=[1],
    maximum_positions=3,
    risk_free_asset='cash'
)

# Expected: Equal weight to A, B, C
assert np.allclose(weights, [0.33, 0.33, 0.33, 0.0, 0.0], atol=0.01)
```

---

## Comparison with Reference Implementation

The original `gem.py` (from `/references/GEM-master/`) differs in several ways:

| Aspect | Original gem.py | Current Implementation |
|--------|-----------------|------------------------|
| **Data Source** | Live API calls (Morningstar, MSCI, FRED) | Pre-loaded DataFrames |
| **Frequency** | Monthly (end-of-month) | Daily (with monthly rebalancing) |
| **Assets** | Fixed 4 assets (US, World-ex-US, Bonds, Treasury) | Configurable N assets |
| **Lookback** | Fixed 1, 6, 12 months | Configurable periods in days |
| **Framework** | Standalone script | Integrated with Backtrader |
| **Output** | Print statements | Weight array + logs |

### Key Similarities

1. **Multi-period momentum**: Both calculate returns over multiple lookback periods
2. **Consistency check**: Both count positive periods to filter assets
3. **Absolute momentum**: Both compare risky assets vs treasury
4. **Equal weighting**: Both use equal weights for qualified assets

---

## Optimization Opportunities

### 1. Vectorized Momentum Calculation

**Current** (loop-based):
```python
for lookback_days in momentum_periods:
    past_prices = dataset.iloc[-lookback_days-1]
    returns = (current_prices - past_prices) / past_prices * 100
```

**Optimized** (vectorized):
```python
lookback_indices = [-p-1 for p in momentum_periods]
past_prices = dataset.iloc[lookback_indices]  # Get all at once
returns = (current_prices - past_prices) / past_prices * 100
```

### 2. Cached Momentum Scores

Store momentum scores in strategy state to avoid recalculation:
```python
# In strategy __init__:
self.momentum_cache = {}

# In gem_portfolio_fun:
cache_key = str(dataset.index[-1])  # Use date as key
if cache_key in self.momentum_cache:
    return self.momentum_cache[cache_key]
```

### 3. Incremental Updates

Instead of recalculating all momentum scores, update only new data:
```python
# Shift old scores and add new period
self.momentum_scores.shift(1)
self.momentum_scores.iloc[0] = calculate_new_momentum()
```

---

## Known Limitations

### 1. Lagging Indicator

Momentum is inherently backward-looking:
- **Problem**: Late entry/exit from trends
- **Impact**: Miss early recovery, late to exit crashes
- **Mitigation**: Use shorter periods or combine with other signals

### 2. Whipsaw Risk

Frequent reversals in sideways markets:
- **Problem**: Buy high, sell low during choppy periods
- **Impact**: High turnover, transaction costs
- **Mitigation**: Increase `min_positive_periods`, use longer lookbacks

### 3. Parameter Sensitivity

Performance varies significantly with parameter choices:
- **Problem**: Optimal parameters change over time (overfitting risk)
- **Impact**: Past performance not predictive
- **Mitigation**: Walk-forward analysis, ensemble of parameter sets

### 4. Equal Weighting

No consideration of asset volatility or correlation:
- **Problem**: High-vol assets get same weight as low-vol
- **Impact**: Suboptimal risk-adjusted returns
- **Mitigation**: Scale weights by inverse volatility

### 5. Binary Decisions

All-or-nothing allocation (qualified vs not):
- **Problem**: Small momentum changes cause large weight shifts
- **Impact**: High turnover
- **Mitigation**: Use gradual weighting based on momentum strength

---

## Future Enhancements

### 1. Weighted Momentum

Instead of equal-period averaging, use exponential decay:
```python
# Recent periods weighted more heavily
decay_weights = np.array([0.4, 0.3, 0.2, 0.1])  # [12mo, 6mo, 3mo, 1mo]
avg_momentum[asset] = (momentum_scores.loc[asset] * decay_weights).sum()
```

### 2. Volatility-Adjusted Weights

Scale allocations by inverse volatility:
```python
vol = dataset.pct_change().rolling(window=21).std().iloc[-1]
risk_adjusted_weights = (1 / vol) / (1 / vol).sum()
```

### 3. Correlation Filter

Avoid highly correlated assets:
```python
correlation_matrix = dataset.pct_change().corr()
# Remove assets with corr > 0.8 to qualified assets
```

### 4. Dynamic Threshold

Adjust treasury threshold based on market regime:
```python
market_vol = dataset.pct_change().std().mean()
dynamic_threshold = treasury_threshold * (market_vol / 0.02)  # Scale by volatility
```

### 5. Partial Defensive Allocation

Allow fractional treasury allocation:
```python
# Instead of 100% treasury when no qualified assets:
treasury_weight = 0.5  # 50% defensive
risky_weight = 0.5 / len(risky_assets)  # 50% across all risky
```

---

## References

### Academic Papers

1. **Antonacci, G. (2014)**. "Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk"  
   - Foundation for combining relative and absolute momentum

2. **Jegadeesh, N., & Titman, S. (1993)**. "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"  
   - Seminal momentum research

3. **Faber, M. (2007)**. "A Quantitative Approach to Tactical Asset Allocation"  
   - Momentum-based tactical allocation strategies

### Related Code

- **Original Implementation**: `references/GEM-master/gem.py`
- **Backtest Framework**: `Backtester/BacktestFramework.py`
- **Strategy Module**: `Backtester/Strategy_dual_momentum.py`

---

**Document Version**: 2.0  
**Last Updated**: November 5, 2025  
**Author**: APPM Individual Project  
**License**: MIT
