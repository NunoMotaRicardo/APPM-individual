# Global Equities Momentum (GEM) Strategy Documentation

## Overview

The Global Equities Momentum (GEM) strategy is a tactical asset allocation approach that combines **relative momentum** (comparing assets against each other) and **absolute momentum** (comparing assets against a risk-free benchmark) to make investment decisions.

## Strategy Logic

### Core Principles

1. **Multi-Period Momentum Evaluation**: Calculate returns over multiple lookback periods (1, 3, 6, and 12 months)
2. **Consistency Filtering**: Select only assets with positive momentum across multiple periods
3. **Absolute Momentum Check**: Compare qualified risky assets against treasury returns
4. **Defensive Allocation**: Switch to treasury when risky assets show weakness

### Implementation Steps

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Calculate Momentum Scores                           │
│   - Compute returns for each asset over 1, 3, 6, 12 months  │
│   - Returns = (Current Price - Past Price) / Past Price     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Filter by Consistency                               │
│   - Count positive momentum periods for each risky asset    │
│   - Require ≥ min_positive_periods (default: 3 out of 4)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Absolute Momentum Check                             │
│   - Compare qualified assets vs treasury average return     │
│   - Filter: avg_momentum > treasury_return + threshold      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Allocate Weights                                    │
│   - If qualified assets exist: Equal weight them            │
│   - If no qualified assets: 100% to treasury                │
└─────────────────────────────────────────────────────────────┘
```

## Parameters

### Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentum_periods` | `[1, 3, 6, 12]` | Lookback periods in months for momentum calculation |
| `min_positive_periods` | `3` | Minimum number of periods with positive momentum required |
| `treasury_threshold` | `0.0` | Minimum excess return over treasury (%) to invest in risky assets |
| `base_column` | `"adjusted"` | Column name to extract from MultiIndex DataFrames |

### Backtest Framework Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rebalance_every` | `21` | Rebalancing frequency (trading days, ~1 month) |
| `lookback` | `252` | Minimum historical data required (trading days, ~1 year) |
| `initial_cash` | `100000` | Starting portfolio value |
| `commission` | `0.001` | Transaction cost (0.1%) |

## Mathematical Formulation

### Momentum Score Calculation

For each asset $i$ and lookback period $p$ (in months):

$$
M_{i,p} = \frac{P_i(t) - P_i(t - p \times 21)}{P_i(t - p \times 21)} \times 100
$$

Where:
- $P_i(t)$ = Current price of asset $i$
- $P_i(t - p \times 21)$ = Price $p$ months ago (~21 trading days per month)
- $M_{i,p}$ = Momentum score as percentage return

### Consistency Score

$$
C_i = \sum_{p \in \{1,3,6,12\}} \mathbb{1}(M_{i,p} > 0)
$$

Where $\mathbb{1}(\cdot)$ is the indicator function (1 if true, 0 if false).

### Asset Qualification

An asset $i$ qualifies if:

$$
C_i \geq \text{min\_positive\_periods} \quad \text{AND} \quad \bar{M}_i > \bar{M}_{\text{treasury}} + \text{threshold}
$$

Where:
- $\bar{M}_i = \frac{1}{4}\sum_{p \in \{1,3,6,12\}} M_{i,p}$ = Average momentum
- $\bar{M}_{\text{treasury}}$ = Average treasury momentum

### Weight Allocation

$$
w_i = \begin{cases} 
\frac{1}{N_q} & \text{if asset } i \text{ is qualified} \\
0 & \text{if asset } i \text{ is risky but not qualified} \\
1 & \text{if asset } i \text{ is treasury and no risky assets qualified}
\end{cases}
$$

Where $N_q$ = number of qualified risky assets.

## Code Structure

### Portfolio Function: `gem_portfolio_fun()`

**Input:**
- `dataset`: DataFrame with price history for all assets
- `base_column`: Column to extract from MultiIndex
- `momentum_periods`: List of lookback periods
- `min_positive_periods`: Consistency threshold
- `treasury_threshold`: Minimum excess return required

**Output:**
- `weights`: NumPy array of portfolio weights (sums to 1.0)
- `log_message`: String with diagnostic information

**Key Logic:**

```python
# 1. Extract data and identify treasury
treasury_assets = [name for name in asset_names 
                   if 'TREAS' in name.upper() or 'BIL' in name.upper()]

# 2. Calculate momentum over multiple periods
for months in momentum_periods:
    lookback_days = months * 21
    past_prices = dataset.iloc[-lookback_days-1]
    returns = ((current_prices - past_prices) / past_prices * 100)
    momentum_scores[months] = returns

# 3. Count positive momentum periods
positive_counts[asset] = (asset_momentum > 0).sum()

# 4. Filter qualified assets
qualified_assets = [asset for asset in risky_assets 
                   if positive_counts[asset] >= min_positive_periods
                   and avg_momentum[asset] > treasury_return + threshold]

# 5. Allocate weights
if len(qualified_assets) == 0:
    weights[treasury_idx] = 1.0  # Defensive position
else:
    # Equal weight qualified assets
    weight_per_asset = 1.0 / len(qualified_assets)
```

### Strategy Class: `GlobalEquitiesMomentumStrategy`

Inherits from `PortfolioRebalanceStrategy` and sets:
- `self.params.portfolio_func = gem_portfolio_fun`

This integrates the GEM logic into the backtest framework.

## Dataset Requirements

### Required Assets

Your dataset **must** include:
1. **Risky Assets**: At least one equity/bond ETF (e.g., IVV, VEU, AGG)
2. **Treasury/Cash Proxy**: Asset with 'TREAS' or 'BIL' in name for defensive allocation

### Data Format

**MultiIndex DataFrame:**

```python
import pandas as pd

# Example structure
columns = pd.MultiIndex.from_product([
    ['BIL', 'IVV', 'VEU', 'AGG'],  # Assets (BIL = treasury)
    ['open', 'high', 'low', 'close', 'adjusted']  # Price columns
])

# Strategy will extract 'adjusted' column by default
```

**Minimum Data Length:**
- At least **252 trading days** (1 year) for 12-month momentum
- More recommended for stability (~500 days / 2 years)

## Usage Example

### Basic Setup

```python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_dual_momentum import GlobalEquitiesMomentumStrategy

# Define strategy configuration
strategies = {
    'GEM_Conservative': (GlobalEquitiesMomentumStrategy, {
        'rebalance_every': 21,  # Monthly
        'lookback': 252,
        'portfolio_func_kwargs': {
            'momentum_periods': [1, 3, 6, 12],
            'min_positive_periods': 3,  # Require 3/4 periods positive
            'treasury_threshold': 0.0
        }
    }),
    'GEM_Aggressive': (GlobalEquitiesMomentumStrategy, {
        'rebalance_every': 21,
        'lookback': 252,
        'portfolio_func_kwargs': {
            'momentum_periods': [3, 6, 12],  # Ignore 1-month noise
            'min_positive_periods': 2,  # Less strict
            'treasury_threshold': 2.0  # Must beat treasury by 2%
        }
    })
}

# Create backtest
backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=[your_data],
    benchmark=['1-N'],  # Equal weight benchmark
    initial_cash=100000,
    commission=0.001
)

# Run
results = backtest.run_backtest()

# Analyze
sharpe_summary = backtest.get_summary_statistics('Sharpe ratio')
print(sharpe_summary)
```

### Parameter Tuning Guide

#### Conservative (Lower Turnover)
- `min_positive_periods`: 3-4 (stricter)
- `treasury_threshold`: 0.0-1.0%
- `momentum_periods`: [1, 3, 6, 12] (all periods)

#### Aggressive (Higher Turnover)
- `min_positive_periods`: 1-2 (lenient)
- `treasury_threshold`: 1.0-3.0% (higher hurdle)
- `momentum_periods`: [6, 12] (longer-term only)

#### Balanced
- `min_positive_periods`: 2-3
- `treasury_threshold`: 0.5-1.5%
- `momentum_periods`: [3, 6, 12] (exclude 1-month)

## Expected Behavior

### Bull Market
- **High momentum consistency** → Multiple risky assets qualify
- **Weights**: Distributed among top-performing equities/bonds
- **Treasury allocation**: 0% (all risky)

### Bear Market
- **Negative momentum** → Few/no assets meet criteria
- **Weights**: 100% treasury (defensive)
- **Effect**: Capital preservation during downturns

### Sideways Market
- **Mixed signals** → Some periods positive, some negative
- **Weights**: Partial risky allocation (1-2 assets) or treasury
- **Effect**: Reduced exposure during uncertainty

## Performance Characteristics

### Strengths
- **Downside Protection**: Switches to treasury during market crashes
- **Trend Following**: Captures sustained bull markets
- **Whipsaw Reduction**: Multi-period requirement filters noise
- **Simplicity**: No complex optimization, transparent rules

### Limitations
- **Late Entry/Exit**: Momentum is a lagging indicator
- **Opportunity Cost**: May miss early recovery phases
- **Parameter Sensitivity**: Performance varies with parameter choices
- **Transaction Costs**: Monthly rebalancing incurs costs

### Typical Metrics (Historical)
- **Sharpe Ratio**: 0.6-1.2 (depends on period)
- **Max Drawdown**: 10-25% (vs 50%+ for buy-and-hold)
- **Win Rate**: 55-65%
- **Turnover**: Low-medium (monthly rebalancing, but holdings stable in trends)

## Comparison with Other Strategies

| Feature | GEM | AAA | Equal Weight (1/N) |
|---------|-----|-----|--------------------|
| **Selection** | Multi-period momentum | Single-period momentum | None |
| **Weighting** | Equal weight | Min variance | Equal weight |
| **Defensive** | Yes (treasury switch) | No | No |
| **Optimization** | None | Quadratic | None |
| **Complexity** | Medium | High | Low |
| **Turnover** | Low-medium | Medium-high | Low |

## Troubleshooting

### Issue: All weights allocated to treasury
**Cause**: No risky assets meet momentum criteria  
**Solution**: 
- Lower `min_positive_periods`
- Reduce `treasury_threshold`
- Check if data includes sufficient bull market periods

### Issue: "Insufficient data" warning
**Cause**: Not enough historical data for 12-month lookback  
**Solution**:
- Ensure dataset has ≥252 trading days
- Reduce `max(momentum_periods)` to 6 months

### Issue: Treasury asset not found
**Cause**: Dataset doesn't include asset with 'TREAS' or 'BIL' in name  
**Solution**:
- Add treasury ETF (BIL, SHY, or custom)
- Or modify `treasury_assets` detection logic

### Issue: High turnover despite low rebalancing frequency
**Cause**: Momentum signals fluctuating month-to-month  
**Solution**:
- Use longer momentum periods (e.g., [6, 12] only)
- Increase `min_positive_periods`
- Add hysteresis (require N consecutive months before switching)

## References

### Academic Papers
1. **Antonacci, G. (2014)**. "Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk"
2. **Faber, M. (2007)**. "A Quantitative Approach to Tactical Asset Allocation"
3. **Jegadeesh, N., & Titman, S. (1993)**. "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"

### Related Resources
- [gem.py source code](../GEM-master/gem.py) - Original momentum calculation logic
- [BacktestFramework.py](BacktestFramework.py) - Integration framework
- [Strategy_AAA.py](Strategy_AAA.py) - Alternative momentum strategy

## Version History

- **v1.0** (2025-10-19): Initial implementation based on gem.py logic
  - Multi-period momentum evaluation
  - Consistency filtering
  - Treasury defensive allocation

## Future Enhancements

### Potential Improvements
1. **Weighted Average Momentum**: Instead of equal weighting periods, use exponential decay
2. **Volatility Adjustment**: Scale momentum by inverse volatility
3. **Correlation Filter**: Avoid highly correlated assets
4. **Dynamic Thresholds**: Adjust treasury_threshold based on market regime
5. **Partial Allocation**: Allow fractional treasury allocation (e.g., 50% defensive)

### Example: Weighted Momentum
```python
# Instead of simple average:
avg_momentum[asset] = asset_momentum.mean()

# Use time-weighted average (recent periods weighted more):
weights = np.array([0.4, 0.3, 0.2, 0.1])  # [12mo, 6mo, 3mo, 1mo]
avg_momentum[asset] = (asset_momentum * weights).sum()
```

---

**Last Updated**: October 19, 2025  
**Author**: APPM Individual Project  
**License**: MIT