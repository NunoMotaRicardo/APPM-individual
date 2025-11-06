# Adaptive Asset Allocation (AAA) Strategy - User Guide

## What is the AAA Strategy?

The Adaptive Asset Allocation (AAA) strategy is a **smart portfolio management** approach that automatically selects the best-performing assets and allocates capital to minimize risk. Think of it as a two-step process:

1. **Pick the winners** - Identifies assets with the strongest recent momentum
2. **Balance the risk** - Allocates more weight to stable performers and less to volatile ones

This strategy aims to capture gains from trending assets while keeping portfolio volatility low through intelligent diversification.

---

## How Does It Work?

### The Big Picture

Imagine you're building a portfolio and you want to:
- Invest in assets that have been going up
- Not put all your eggs in one basket
- Keep your overall portfolio risk low

The AAA strategy does this automatically by:
1. Looking at how much each asset has grown recently (momentum)
2. Picking the top performers
3. Using math to figure out the safest way to split your money among them

### The 4-Step Process

#### Step 1: Calculate Momentum

For each asset (stock, ETF, bond, etc.), calculate how much it has gained over the past **6 months** (132 trading days):

**Example:**
- Asset A: Started at $100, now $120 → Momentum = 1.20 (20% gain) ✅
- Asset B: Started at $50, now $55 → Momentum = 1.10 (10% gain) ✅
- Asset C: Started at $80, now $75 → Momentum = 0.94 (6% loss) ❌

**The higher the momentum score, the better the asset has performed.**

#### Step 2: Select Top Performers

Rank all assets by momentum and select the **top 5** (by default):

**From our example:**
- If you have 10 assets in your universe, AAA picks the 5 with the highest momentum scores
- Assets that have been losing value or underperforming are excluded

**Result:** You're left with only the "winning" assets

#### Step 3: Estimate Risk

For the selected winners, look at their recent volatility (last **1 month**):

- How much does each asset fluctuate day-to-day?
- How do they move together (correlation)?

The strategy builds a **covariance matrix** - a mathematical representation of:
- Individual asset risk (volatility)
- Relationships between assets (do they move together or opposite?)

#### Step 4: Optimize Weights

Use mathematical optimization to find the allocation that:
- **Minimizes portfolio volatility** (variance)
- Keeps you fully invested (weights sum to 100%)
- Only allows long positions (no shorting)

**Example Output:**
```
Asset A: 35% (strong momentum, low volatility)
Asset B: 25% (good momentum, medium volatility)
Asset C: 20% (decent momentum, higher volatility)
Asset D: 15% (acceptable momentum, correlated with A)
Asset E: 5%  (selected but high volatility, gets small weight)
```

The optimization automatically gives larger weights to assets that contribute less to overall portfolio risk.

---

## When to Use AAA

### ✅ Good For

- **Long-term investors** who rebalance monthly or quarterly
- **Diversified portfolios** with 10+ asset choices (stocks, ETFs, bonds, commodities)
- **Risk-conscious investors** who want exposure to winners without excessive volatility
- **Systematic traders** who prefer rule-based strategies over discretion
- **Combining momentum and risk management** in a single framework

### ⚠️ Consider Carefully

- **Short-term trading** - AAA is designed for monthly rebalancing, not daily trading
- **Concentrated portfolios** - Works best with diverse asset universes
- **Bear markets** - AAA stays fully invested; doesn't move to cash during downturns
- **High transaction costs** - Monthly rebalancing can generate trading costs

### ❌ Not Ideal For

- **Always holding specific assets** - AAA may exclude your favorites if momentum is weak
- **Day trading or intraday strategies** - Too slow for high-frequency approaches
- **Very small universes** (< 5 assets) - Limited diversification benefits
- **Crypto or highly volatile assets** - May produce extreme concentration

---

## Setting Up Your First Backtest

### Prerequisites

You need:
1. **Daily price data** for your assets (CSV or DataFrame)
2. **At least 6 months** of history (132 trading days + warm-up period)
3. **Python environment** with required packages: `pandas`, `numpy`, `scipy`, `backtrader`

### Basic Setup (Step-by-Step)

#### 1. Prepare Your Data

Create a CSV file with daily adjusted close prices:

```csv
Date,SPY,QQQ,IWM,EFA,EEM,AGG,GLD,TLT
2023-01-01,380.00,290.00,185.00,65.00,40.00,103.00,170.00,98.00
2023-01-02,382.50,292.00,186.50,65.20,40.30,103.10,171.00,98.50
2023-01-03,381.00,291.50,185.80,65.10,40.10,103.05,170.50,98.30
...
```

**Assets in this example:**
- **SPY**: S&P 500 ETF (US large-cap stocks)
- **QQQ**: NASDAQ-100 ETF (US tech stocks)
- **IWM**: Russell 2000 ETF (US small-cap stocks)
- **EFA**: EAFE ETF (developed international markets)
- **EEM**: Emerging Markets ETF
- **AGG**: Aggregate Bond ETF (US bonds)
- **GLD**: Gold ETF (commodities)
- **TLT**: Long-term Treasury ETF (safe haven)

#### 2. Write the Backtest Code

```python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_AAA import AdaptiveAssetAllocationStrategy
import pandas as pd

# Load your data
data = pd.read_csv('my_portfolio_data.csv', index_col='Date', parse_dates=True)

# Define your strategy configuration
strategies = {
    'AAA_Standard': (AdaptiveAssetAllocationStrategy, {
        'rebalance_every': 21,  # Rebalance monthly (21 trading days)
        'lookback': 132,        # Need 6 months of data for momentum
        'portfolio_func_kwargs': {
            'momentum_top_n': 5,              # Select top 5 assets
            'momentum_lookback': 132,         # 6 months momentum window
            'covariance_lookback': 22,        # 1 month covariance window
            'base_column': 'adjusted'         # Use adjusted prices
        }
    })
}

# Create and run backtest
backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets={'my_data': data},
    benchmark=['1-N'],          # Compare to equal-weight benchmark
    initial_cash=100000,        # Start with $100,000
    commission=0.001            # 0.1% per trade
)

# Run the backtest
results = backtest.run()

# Access results
print(results.summary())
results.plot()
```

#### 3. Run the Backtest

```python
# Execute the backtest (this will take a few seconds to minutes depending on data size)
results.run()

# View performance metrics
print(f"Total Return: {results.total_return:.2%}")
print(f"Annual Return: {results.annual_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Volatility: {results.volatility:.2%}")
```

---

## Understanding the Parameters

### Core Parameters

#### `momentum_top_n` (Default: 5)

**What it does:** Number of top-performing assets to select

**Impact:**
- **Lower values (3)**: More concentrated, potentially higher returns but also higher risk
- **Higher values (10)**: More diversified, smoother returns, lower concentration risk

**When to adjust:**
- Large universe (20+ assets): Increase to 7-10 for better diversification
- Small universe (10 assets): Keep at 3-5 to maintain selectivity
- Very volatile markets: Increase to spread risk

**Example:**
```python
'momentum_top_n': 3   # Aggressive - top 3 only
'momentum_top_n': 5   # Standard - balanced approach
'momentum_top_n': 10  # Conservative - more diversification
```

#### `momentum_lookback` (Default: 132 days ≈ 6 months)

**What it does:** How far back to look when calculating momentum

**Impact:**
- **Shorter periods (66 days = 3 months)**: More responsive to recent trends, may chase noise
- **Longer periods (252 days = 12 months)**: More stable signals, may miss short-term opportunities

**When to adjust:**
- Fast-moving markets: Shorten to 66-88 days (3-4 months)
- Stable, trending markets: Lengthen to 180-252 days (9-12 months)
- High volatility: Use longer periods to smooth out noise

**Example:**
```python
'momentum_lookback': 66   # 3 months - responsive
'momentum_lookback': 132  # 6 months - standard
'momentum_lookback': 252  # 12 months - stable
```

#### `covariance_lookback` (Default: 22 days ≈ 1 month)

**What it does:** How much recent data to use for risk estimation

**Impact:**
- **Shorter periods (10-15 days)**: Captures current market conditions, may be unstable
- **Longer periods (44-66 days)**: More stable estimates, may lag actual conditions

**When to adjust:**
- Rapidly changing volatility: Keep short (10-22 days)
- Stable market environments: Extend to 44-66 days
- Small asset selection: Increase to ensure stable covariance matrix

**Example:**
```python
'covariance_lookback': 10  # 2 weeks - very reactive
'covariance_lookback': 22  # 1 month - standard
'covariance_lookback': 44  # 2 months - stable
```

### Rebalancing Parameters

#### `rebalance_every` (Default: 21 days ≈ 1 month)

**What it does:** How often to recalculate and adjust portfolio weights

**Impact:**
- **More frequent (5-10 days)**: Higher transaction costs, more responsive
- **Less frequent (63-126 days)**: Lower costs, may miss opportunities

**When to adjust:**
- High transaction costs: Increase to 44-63 days (2-3 months)
- Low transaction costs: Can rebalance more frequently (10-21 days)
- Very volatile markets: Consider weekly (5 days) rebalancing

**Example:**
```python
'rebalance_every': 5   # Weekly - high turnover
'rebalance_every': 21  # Monthly - balanced
'rebalance_every': 63  # Quarterly - low turnover
```

---

## Parameter Configurations for Different Goals

### Configuration 1: Aggressive Growth

**Goal:** Maximum returns, willing to accept higher risk and concentration

```python
'portfolio_func_kwargs': {
    'momentum_top_n': 3,              # Concentrate in top 3
    'momentum_lookback': 66,          # Shorter-term momentum (3 months)
    'covariance_lookback': 22,        # Standard risk window
    'base_column': 'adjusted'
}
'rebalance_every': 21  # Monthly rebalancing
```

**Characteristics:**
- High concentration (33% per asset)
- Chases recent winners
- Higher volatility expected
- Suitable for risk-tolerant investors

### Configuration 2: Balanced (Standard)

**Goal:** Good returns with moderate risk

```python
'portfolio_func_kwargs': {
    'momentum_top_n': 5,              # Select top 5
    'momentum_lookback': 132,         # Medium-term momentum (6 months)
    'covariance_lookback': 22,        # Standard risk window
    'base_column': 'adjusted'
}
'rebalance_every': 21  # Monthly rebalancing
```

**Characteristics:**
- Balanced diversification (20% per asset average)
- Stable momentum signals
- Moderate transaction costs
- Suitable for most investors

### Configuration 3: Conservative / Low Volatility

**Goal:** Minimize risk, accept lower returns

```python
'portfolio_func_kwargs': {
    'momentum_top_n': 8,              # More diversification
    'momentum_lookback': 252,         # Long-term momentum (12 months)
    'covariance_lookback': 44,        # Longer risk window for stability
    'base_column': 'adjusted'
}
'rebalance_every': 63  # Quarterly rebalancing
```

**Characteristics:**
- High diversification (12.5% per asset average)
- Very stable signals
- Low turnover / transaction costs
- Suitable for risk-averse investors

### Configuration 4: Rapid Response

**Goal:** Quickly adapt to changing market conditions

```python
'portfolio_func_kwargs': {
    'momentum_top_n': 5,              # Standard selection
    'momentum_lookback': 44,          # Short-term momentum (2 months)
    'covariance_lookback': 10,        # Very recent risk estimate
    'base_column': 'adjusted'
}
'rebalance_every': 10  # Bi-weekly rebalancing
```

**Characteristics:**
- Highly responsive to trends
- Frequent rebalancing (higher costs)
- Can capture short-term opportunities
- Suitable for active traders with low commission

---

## Interpreting Results

### Key Metrics to Monitor

#### 1. Total Return

**What it means:** Overall gain/loss over the backtest period

**Good AAA performance:** 
- Should beat equal-weight benchmark by 2-5% annually
- Competitive with buy-and-hold top performers

#### 2. Sharpe Ratio

**What it means:** Return per unit of risk (higher is better)

**Target values:**
- < 1.0: Poor risk-adjusted returns
- 1.0 - 1.5: Good performance
- > 1.5: Excellent risk-adjusted returns

**AAA typically achieves:** 1.2 - 1.8 in diversified markets

#### 3. Maximum Drawdown

**What it means:** Largest peak-to-trough decline

**Expected for AAA:**
- Bull markets: 10-20% drawdowns
- Bear markets: 30-40% drawdowns (stays fully invested)
- Should be less than equal-weight benchmark

#### 4. Volatility (Annual Standard Deviation)

**What it means:** How much returns fluctuate

**Expected for AAA:**
- 12-18% for diversified equity portfolios
- Should be lower than equal-weight or buy-and-hold
- Minimum variance optimization should reduce this

#### 5. Turnover

**What it means:** How much trading occurs (% of portfolio traded per rebalance)

**Typical AAA turnover:**
- 30-60% monthly (moderate)
- Important for calculating real-world costs

### Weight Distribution Analysis

Check your weight outputs to ensure:

**✅ Healthy distribution:**
```
Asset A: 28%
Asset B: 24%
Asset C: 20%
Asset D: 18%
Asset E: 10%
```
Relatively balanced, no extreme concentration

**⚠️ Warning signs:**
```
Asset A: 92%
Asset B: 3%
Asset C: 3%
Asset D: 1%
Asset E: 1%
```
Extreme concentration - may indicate:
- Very high correlation among selected assets
- One asset has much lower volatility
- Numerical issues in optimization

**Fix:** Increase `covariance_lookback` or add weight constraints

---

## Common Questions (FAQ)

### Q1: Why isn't my favorite stock/ETF always included?

**A:** AAA only selects assets with strong momentum. If your asset has underperformed recently, it won't be selected. This is by design - the strategy avoids assets that are trending down.

### Q2: Can AAA go to cash during bear markets?

**A:** No, the standard AAA strategy stays fully invested. It will select the "best of the worst" during downturns. To add defensive positioning, consider:
- Including bond/treasury ETFs in your universe (they may get selected)
- Combining AAA with absolute momentum filters (like GEM strategy)
- Implementing a separate cash rule based on market conditions

### Q3: Why do I see warnings about "optimizer failed"?

**A:** This happens when the covariance matrix is poorly conditioned (assets too correlated or insufficient data). The strategy falls back to equal weights among selected assets. To reduce warnings:
- Increase `covariance_lookback` (more data = more stable)
- Ensure your universe has diverse, not highly correlated assets
- Check that all assets have complete price histories

### Q4: How do I handle transaction costs?

**A:** Set the `commission` parameter in the backtest configuration:

```python
backtest = BacktraderPortfolioBacktest(
    ...
    commission=0.001  # 0.1% per trade (10 basis points)
)
```

For more accurate modeling, include:
- Spread costs (bid-ask): Add ~0.05-0.2% to commission
- Slippage: Use backtrader's slippage settings
- Min trade size: Filter out tiny adjustments

### Q5: Can I set maximum/minimum weights per asset?

**A:** Yes, but requires code modification. In `Strategy_AAA.py`, modify the optimization bounds:

```python
# Current code (line ~140):
bounds = tuple((0.0, 1.0) for _ in range(num_valid_assets))

# Change to (e.g., max 30% per asset):
max_weight = 0.30
bounds = tuple((0.0, max_weight) for _ in range(num_valid_assets))
```

### Q6: How does AAA compare to equal-weight?

**Typical characteristics:**

| Metric | Equal-Weight | AAA |
|--------|--------------|-----|
| Returns | Baseline | 1-3% higher annually |
| Volatility | Higher | Lower (by design) |
| Sharpe Ratio | ~1.0 | ~1.2-1.5 |
| Max Drawdown | Larger | Smaller |
| Turnover | Low (only on rebalance) | Moderate |

AAA should outperform equal-weight on a risk-adjusted basis (Sharpe ratio).

### Q7: What if I have a small universe (5-8 assets)?

**Recommendations:**
- Set `momentum_top_n = 3` (select ~40-60% of universe)
- Increase `covariance_lookback = 44` (need more stable estimates with fewer assets)
- Consider longer `momentum_lookback = 180-252` (more reliable signals)

### Q8: Can I use AAA with international markets or FX data?

**Yes**, AAA is currency-agnostic. Just ensure:
- All prices in same currency (or use FX-adjusted returns)
- Consistent data frequency (all daily, no gaps)
- Sufficient liquidity for actual trading

---

## Advanced Tips

### 1. Combining AAA with Other Strategies

**AAA + GEM Hybrid:**
- Run AAA to get momentum-selected assets
- Apply GEM's absolute momentum filter on top
- Result: Only invest if selected assets beat treasury/cash benchmark

**AAA + Risk Parity:**
- Use AAA for asset selection
- Replace minimum variance with risk parity weighting
- Result: Equal risk contribution from selected assets

### 2. Sector/Theme-Based AAA

Instead of individual assets, use sector ETFs:

```python
data = pd.DataFrame({
    'XLF': [...],  # Financials
    'XLE': [...],  # Energy
    'XLK': [...],  # Technology
    'XLV': [...],  # Healthcare
    'XLI': [...],  # Industrials
    ...
})
```

AAA will rotate among sectors with strongest momentum.

### 3. Multi-Timeframe AAA

Run AAA with different momentum windows and combine:

```python
strategies = {
    'AAA_Short': (..., {'momentum_lookback': 66}),   # 3-month
    'AAA_Medium': (..., {'momentum_lookback': 132}), # 6-month
    'AAA_Long': (..., {'momentum_lookback': 252}),   # 12-month
}

# Blend results: 1/3 each timeframe
```

Reduces sensitivity to any single momentum period.

### 4. Add Stop-Loss Protection

Implement portfolio-level stop-loss:

```python
# Pseudo-code - add to backtest logic
if portfolio_value < 0.85 * peak_value:  # 15% drawdown
    go_to_cash_or_bonds()
    pause_AAA_for_N_days()
```

Exits positions during severe downturns.

---

## Troubleshooting

### Problem: Strategy underperforms benchmark

**Possible causes:**
1. Momentum reversal period (winners become losers)
2. Transaction costs too high
3. Rebalancing too frequent
4. Universe lacks diversity

**Solutions:**
- Lengthen `momentum_lookback` to reduce noise
- Reduce rebalancing frequency (`rebalance_every = 44` or 63)
- Analyze turnover and adjust `momentum_top_n` to reduce churn
- Add transaction cost analysis to identify impact

### Problem: Extreme weight concentration

**Symptoms:** One asset gets 70-90%+ allocation

**Causes:**
- Selected assets highly correlated → one dominates
- Short `covariance_lookback` captures temporary low volatility
- Universe has one extremely low-volatility asset

**Solutions:**
1. Increase `covariance_lookback` to 44-66 days
2. Add maximum weight constraint (see FAQ Q5)
3. Remove or adjust highly correlated assets from universe
4. Add regularization to covariance matrix (requires code change)

### Problem: Frequent optimization failures

**Symptoms:** Many "WARNING: AAA - optimizer failed" messages

**Causes:**
- Covariance matrix singular or near-singular
- Too few observations relative to assets
- Extreme correlations (ρ ≈ 1.0)

**Solutions:**
1. Increase `covariance_lookback` (ensure C > n_valid)
2. Reduce `momentum_top_n` (fewer assets = more stable)
3. Check data quality (missing values, duplicates)
4. Consider using shrinkage estimator (requires code modification)

### Problem: No trades occurring

**Check:**
1. Data loaded correctly? (`print(data.head())`)
2. Sufficient history? (need `momentum_lookback + warmup` days)
3. Rebalance frequency set? (`rebalance_every` parameter)
4. Strategy activated? (check backtest logs)

---

## Next Steps

### Learning Path

1. **Start simple**: Run backtest with default parameters on diverse universe
2. **Understand behavior**: Plot weight evolution, analyze selected assets over time
3. **Parameter sensitivity**: Test different `momentum_top_n` and lookback values
4. **Transaction costs**: Add realistic commissions and compare results
5. **Combine strategies**: Explore AAA+GEM or AAA+Risk Parity hybrids

### Recommended Reading

- **Original AAA Paper**: Butler, Philbrick & Gordillo - "Adaptive Asset Allocation" (2012)
- **Momentum Strategies**: Jegadeesh & Titman (1993) on momentum persistence
- **Portfolio Optimization**: Markowitz - "Portfolio Selection" (1952)

### Helpful Backtest Practices

1. **Walk-forward testing**: Use rolling windows, not single period
2. **Out-of-sample validation**: Reserve last 20% of data for testing
3. **Parameter stability**: Test multiple configurations to avoid overfitting
4. **Regime analysis**: Check performance in bull/bear/sideways markets separately

---

## Example: Complete Working Backtest

Here's a full example you can run immediately:

```python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_AAA import AdaptiveAssetAllocationStrategy
import pandas as pd

# Load your data (assumes CSV format with Date index)
data = pd.read_csv('universe_data.csv', index_col='Date', parse_dates=True)

# Configure AAA strategy
strategies = {
    'AAA_Balanced': (AdaptiveAssetAllocationStrategy, {
        'rebalance_every': 21,              # Monthly rebalancing
        'lookback': 150,                    # Keep extra history
        'portfolio_func_kwargs': {
            'momentum_top_n': 5,            # Top 5 assets
            'momentum_lookback': 132,       # 6-month momentum
            'covariance_lookback': 22,      # 1-month covariance
            'base_column': 'adjusted'       # Use adjusted prices
        }
    }),
    
    # Optional: Add a benchmark comparison
    'EqualWeight': (AdaptiveAssetAllocationStrategy, {
        'rebalance_every': 21,
        'lookback': 22,
        'portfolio_func_kwargs': {
            'momentum_top_n': len(data.columns),  # Select all
            'momentum_lookback': 1,               # No momentum filter
            'covariance_lookback': 22
        }
    })
}

# Create backtest instance
backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets={'universe': data},
    benchmark=['1-N'],                      # Equal-weight benchmark
    initial_cash=100000,                    # $100k starting capital
    commission=0.001,                       # 0.1% commission
    start_date='2020-01-01',                # Optional: specify start
    end_date='2024-12-31'                   # Optional: specify end
)

# Run backtest
print("Running backtest...")
results = backtest.run()

# Display results
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(results.summary())

# Plot performance
results.plot(title='AAA Strategy Performance')

# Export detailed results
results.export_trades('aaa_trades.csv')
results.export_weights('aaa_weights.csv')
```

---

**Document Version:** 1.0  
**Last Updated:** November 5, 2025  
**Target Audience:** Strategy users and backtest practitioners  
**Related:** See `AAA_Technical_Reference.md` for implementation details
