# Global Equities Momentum (GEM) Strategy - User Guide

## What is the GEM Strategy?

The Global Equities Momentum (GEM) strategy is a **tactical asset allocation** approach that automatically switches between growth and safety based on market momentum. Think of it as a smart autopilot for your portfolio that:

1. **Follows trends** - Invests in assets that are moving up
2. **Requires consistency** - Looks for sustained momentum across multiple time periods
3. **Protects capital** - Moves to safety (cash or treasury bonds) when markets are weak

This strategy aims to capture market gains while avoiding large drawdowns during bear markets.

---

## How Does It Work?

### The Big Picture

Imagine you're deciding which stocks to buy. Instead of using complex analysis, you simply ask:

1. **"Has this stock been going up?"** (Momentum check)
2. **"Has it been going up consistently?"** (Consistency check)
3. **"Is it doing better than just holding safe bonds?"** (Absolute momentum check)

If the answer to all three is "yes," you invest. If not, you move to safety.

### The 4-Step Process

#### Step 1: Calculate Momentum

For each asset in your universe (stocks, bonds, ETFs, etc.), calculate how much it has returned over multiple time periods:

- **1 month** (21 trading days)
- **3 months** (63 trading days)
- **6 months** (126 trading days)
- **12 months** (252 trading days)

**Example:**
- Stock A: +5% (1mo), +8% (3mo), +12% (6mo), +20% (12mo) ✅ All positive
- Stock B: -2% (1mo), +3% (3mo), +5% (6mo), +10% (12mo) ⚠️ Mixed
- Stock C: -3% (1mo), -5% (3mo), -8% (6mo), -12% (12mo) ❌ All negative

#### Step 2: Check Consistency

Count how many time periods have positive momentum. Only keep assets with **at least 3 out of 4** positive periods (by default).

**From our example:**
- Stock A: 4/4 positive → **Qualifies** ✅
- Stock B: 3/4 positive → **Qualifies** ✅  
- Stock C: 0/4 positive → **Rejected** ❌

#### Step 3: Beat the Risk-Free Rate

Compare qualified assets against treasury bonds or cash:

- If Stock A's average momentum is 11.25% and treasury returned 2%, Stock A wins (11.25% > 2%)
- Keep only assets that beat the safe alternative

#### Step 4: Allocate Weights

- **If qualified assets exist**: Split money equally among them
  - Example: 3 qualified assets → 33.3% each
- **If no assets qualify**: Move to defensive position
  - Option A: Hold cash (0% allocation to everything)
  - Option B: Buy treasury bonds (100% in safe asset)

---

## When to Use GEM

### ✅ Good For

- **Risk-averse investors** who want downside protection
- **Long-term portfolios** (1+ years) with monthly rebalancing
- **Diversified asset universes** (mix of stocks, bonds, commodities, etc.)
- **Trend-following enthusiasts** who believe momentum persists
- **Reducing stress** by automating buy/sell decisions

### ❌ Not Ideal For

- **Day traders** or short-term investors (strategy rebalances monthly)
- **Always-invested mandates** (strategy can go 100% cash/bonds)
- **High-frequency trading** (momentum is a slow signal)
- **Bear-market opportunists** (strategy exits declining markets, may miss early reversals)

---

## Setting Up Your First Backtest

### Prerequisites

You need:
1. **Daily price data** for your assets (CSV or DataFrame)
2. **At least 1 year** of history (252 trading days)
3. **Python environment** with Backtrader installed

### Basic Setup (Step-by-Step)

#### 1. Prepare Your Data

Create a CSV file with daily adjusted close prices:

```csv
Date,IVV,VEU,AGG,BIL
2023-01-01,450.00,55.00,102.00,91.50
2023-01-02,451.20,55.10,102.10,91.50
2023-01-03,449.80,54.90,102.00,91.60
...
```

**Assets in this example:**
- **IVV**: S&P 500 ETF (US large-cap stocks)
- **VEU**: All-World ex-US ETF (international stocks)
- **AGG**: Aggregate Bond ETF (US bonds)
- **BIL**: Treasury Bill ETF (ultra-safe, cash-like)

#### 2. Write the Backtest Code

```python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_dual_momentum import GlobalEquitiesMomentumStrategy
import pandas as pd

# Load your data
data = pd.read_csv('my_portfolio_data.csv', index_col='Date', parse_dates=True)

# Define your strategy configuration
strategies = {
    'GEM_Conservative': (GlobalEquitiesMomentumStrategy, {
        'rebalance_every': 21,  # Rebalance monthly (21 trading days)
        'lookback': 252,        # Need 1 year of data
        'portfolio_func_kwargs': {
            'momentum_periods': [21, 63, 126, 252],  # 1, 3, 6, 12 months
            'min_positive_periods': 3,               # Need 3/4 positive
            'treasury_threshold': 0.0,               # Any excess return is OK
            'maximum_positions': 4,                  # Hold max 4 assets
            'risk_free_asset': 'BIL'                 # Use BIL when defensive
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

results = backtest.run_backtest()
print(results.get_summary_statistics())
```

#### 3. Run It

```bash
python your_backtest_script.py
```

You'll see output like:

```
Strategy: GEM_Conservative
Final Portfolio Value: $152,345
Total Return: 52.35%
Sharpe Ratio: 1.12
Max Drawdown: -8.23%
Win Rate: 62.5%
```

---

## Parameter Tuning Guide

The GEM strategy has 5 key parameters you can adjust. Here's what each does and when to change it:

### 1. `momentum_periods` (Lookback Periods)

**What it does**: Defines how far back to look when calculating momentum.

**Default**: `[21, 63, 126, 252]` (1, 3, 6, 12 months in trading days)

**How to adjust**:

```python
# More responsive (faster reactions, more trades)
'momentum_periods': [21, 63, 126]  # Drop 12-month (ignore long-term)

# More stable (slower reactions, fewer trades)
'momentum_periods': [126, 252]  # Only 6 and 12 months (ignore short-term noise)

# Custom periods
'momentum_periods': [10, 30, 60, 120]  # 2 weeks, 1.5mo, 3mo, 6mo
```

**When to use shorter periods**:
- In fast-moving markets (crypto, tech stocks)
- When you want to capture short-term trends
- If you're willing to pay more transaction costs

**When to use longer periods**:
- In stable markets (bonds, dividends)
- When you want to reduce whipsaw (false signals)
- If transaction costs are high

---

### 2. `min_positive_periods` (Consistency Filter)

**What it does**: How many periods must have positive momentum for an asset to qualify.

**Default**: `3` (out of 4 periods)

**How to adjust**:

```python
# Very strict (fewer trades, only strong trends)
'min_positive_periods': 4  # All periods must be positive

# Moderate
'min_positive_periods': 3  # Default (75% positive)

# Lenient (more trades, catch weaker trends)
'min_positive_periods': 2  # Only 50% positive required

# Very lenient
'min_positive_periods': 1  # Any positive momentum counts
```

**Trade-off**:
- **Higher values** → Fewer qualified assets → More time in cash/treasury → Lower returns but safer
- **Lower values** → More qualified assets → More time in risky assets → Higher returns but riskier

---

### 3. `treasury_threshold` (Minimum Excess Return)

**What it does**: How much better risky assets must perform vs safe assets to justify investing.

**Default**: `0.0` (any excess return is enough)

**How to adjust**:

```python
# No hurdle (invest if any positive excess return)
'treasury_threshold': 0.0  # Default

# Low hurdle (must beat treasury by 1%)
'treasury_threshold': 1.0

# Medium hurdle (must beat treasury by 2%)
'treasury_threshold': 2.0

# High hurdle (must beat treasury by 5%)
'treasury_threshold': 5.0
```

**When to raise it**:
- In high-inflation environments (bonds may not be truly "safe")
- When you want to be more defensive
- If you expect mean reversion (high returns → future crashes)

**When to lower it**:
- In low-interest environments (treasuries return very little)
- When you want to stay invested more often
- In bull markets with sustained momentum

---

### 4. `maximum_positions` (Position Limit)

**What it does**: Maximum number of assets to hold at once.

**Default**: `6`

**How to adjust**:

```python
# Concentrated (high conviction)
'maximum_positions': 3

# Balanced
'maximum_positions': 6  # Default

# Diversified
'maximum_positions': 10

# Unlimited
'maximum_positions': None  # Hold all qualified assets
```

**Trade-off**:
- **Fewer positions** → Less diversification → Higher risk/return volatility → Lower transaction costs
- **More positions** → More diversification → Smoother returns → Higher transaction costs

---

### 5. `risk_free_asset` (Defensive Position)

**What it does**: What to do when no risky assets qualify.

**Default**: `'cash'` (hold cash, earn broker interest)

**How to adjust**:

```python
# Option 1: Hold cash (no positions)
'risk_free_asset': 'cash'

# Option 2: Buy ultra-short treasuries
'risk_free_asset': 'BIL'  # 1-3 month T-bills

# Option 3: Buy short-term treasuries
'risk_free_asset': 'SHY'  # 1-3 year Treasury bonds

# Option 4: Buy intermediate treasuries (higher yield, more risk)
'risk_free_asset': 'IEF'  # 7-10 year Treasury bonds
```

**Comparison**:

| Option | Yield | Risk | When to Use |
|--------|-------|------|-------------|
| `'cash'` | 0-2% | None | Ultra-conservative, avoid all market exposure |
| `'BIL'` | 4-5% | Minimal | Want treasury yield, minimal duration risk |
| `'SHY'` | 4-6% | Low | Balanced safety + yield |
| `'IEF'` | 3-7% | Medium | Willing to take duration risk for higher yield |

**Important**: The asset symbol must exist in your dataset!

---

## Example Configurations

### Conservative (Capital Preservation)

**Goal**: Protect capital, accept lower returns

```python
'portfolio_func_kwargs': {
    'momentum_periods': [63, 126, 252],     # Ignore 1-month noise
    'min_positive_periods': 3,              # Strict (need 3/3 positive)
    'treasury_threshold': 1.0,              # Must beat treasury by 1%
    'maximum_positions': 3,                 # Concentrated positions
    'risk_free_asset': 'BIL'                # Earn T-bill yield when defensive
}
```

**Expected**:
- Lower turnover (fewer trades)
- More time in defensive position
- Max drawdown: ~5-10%
- Sharpe ratio: ~1.0-1.5

---

### Balanced (Default)

**Goal**: Balance risk and return

```python
'portfolio_func_kwargs': {
    'momentum_periods': [21, 63, 126, 252], # All periods
    'min_positive_periods': 3,              # Need 3/4 positive
    'treasury_threshold': 0.0,              # Any excess return OK
    'maximum_positions': 6,                 # Moderate diversification
    'risk_free_asset': 'cash'               # Hold cash when defensive
}
```

**Expected**:
- Moderate turnover
- Balanced time in risky/safe assets
- Max drawdown: ~10-15%
- Sharpe ratio: ~0.8-1.2

---

### Aggressive (Maximum Returns)

**Goal**: Maximize returns, accept higher risk

```python
'portfolio_func_kwargs': {
    'momentum_periods': [21, 63, 126],      # Shorter periods (more responsive)
    'min_positive_periods': 2,              # Lenient (only need 2/3)
    'treasury_threshold': -1.0,             # Invest even if slightly behind treasury
    'maximum_positions': 10,                # High diversification
    'risk_free_asset': 'cash'               # Rarely go defensive
}
```

**Expected**:
- High turnover (many trades)
- More time in risky assets
- Max drawdown: ~20-30%
- Sharpe ratio: ~0.5-0.9

---

## Understanding the Results

After running a backtest, you'll get several metrics. Here's what they mean:

### Key Metrics Explained

#### 1. Total Return

**What it is**: How much your portfolio grew from start to finish.

**Example**: Started with $100,000, ended with $150,000 → **50% return**

**Good/Bad**:
- **Good**: Beats benchmark (e.g., buy-and-hold S&P 500)
- **Bad**: Underperforms benchmark significantly

---

#### 2. Sharpe Ratio

**What it is**: Risk-adjusted return (return per unit of volatility).

**Formula**: (Return - Risk-Free Rate) / Volatility

**Interpretation**:
- **< 0.5**: Poor (too much risk for the return)
- **0.5-1.0**: Decent
- **1.0-2.0**: Good (typical for GEM)
- **> 2.0**: Excellent (rare)

---

#### 3. Max Drawdown

**What it is**: Largest peak-to-trough decline during the backtest.

**Example**: Portfolio dropped from $150k to $120k → **-20% drawdown**

**What to expect**:
- **GEM strategy**: 5-20% (vs 40-50% for buy-and-hold)
- **Conservative GEM**: 5-10%
- **Aggressive GEM**: 15-25%

**Why it matters**: Shows worst-case scenario (how much you could lose).

---

#### 4. Win Rate

**What it is**: Percentage of rebalancing periods with positive returns.

**Example**: 12 rebalances, 8 were profitable → **67% win rate**

**Typical GEM**: 55-65%

---

#### 5. Turnover

**What it is**: How much trading activity occurs (% of portfolio traded per period).

**Example**: On average, 30% of portfolio is rebalanced each month → **30% turnover**

**Impact on costs**:
- **High turnover (>50%)**: Transaction costs eat into returns
- **Low turnover (<20%)**: Minimal cost impact

**Typical GEM**: 15-40% (varies with parameter choices)

---

## Common Questions

### Q1: Why did my strategy go to cash?

**Answer**: No risky assets met the momentum criteria. This usually happens during:
- **Bear markets** (everything is falling)
- **Market transitions** (end of bull run, start of bear market)
- **High volatility** (choppy, sideways movement)

**Is this bad?**: No! This is the strategy's **downside protection** in action. Holding cash avoids losses.

---

### Q2: Why am I getting "Insufficient data" warnings?

**Answer**: Not enough historical data for the longest lookback period.

**Solutions**:
- Use shorter `momentum_periods` (e.g., `[21, 63, 126]` instead of `[21, 63, 126, 252]`)
- Increase your dataset length (get more historical data)
- Increase the `lookback` parameter in strategy config

---

### Q3: How often should I rebalance?

**Answer**: The default is **monthly** (`rebalance_every=21` trading days). This balances:
- **Too frequent** (weekly): High transaction costs, whipsaw risk
- **Too infrequent** (quarterly): Slow to react, miss trend changes

**Alternative options**:
```python
'rebalance_every': 5   # Weekly (more active)
'rebalance_every': 21  # Monthly (default, recommended)
'rebalance_every': 63  # Quarterly (more passive)
```

---

### Q4: What if I don't have a treasury ETF in my data?

**Answer**: Use cash instead:

```python
'risk_free_asset': 'cash'  # Always works (no asset needed)
```

Or add a treasury ETF to your dataset (e.g., BIL, SHY, IEF).

---

### Q5: Can I use this with stocks instead of ETFs?

**Answer**: Yes! The strategy works with any assets:
- **ETFs** (recommended): IVV, VEU, AGG, BIL, etc.
- **Individual stocks**: AAPL, MSFT, TSLA, etc.
- **Bonds**: Treasury bonds, corporate bonds
- **Commodities**: Gold, oil, etc.
- **Crypto**: BTC, ETH (if you have data)

**Note**: Individual stocks are riskier (higher volatility, company-specific events).

---

### Q6: How does GEM compare to other strategies?

| Strategy | Approach | Downside Protection | Complexity |
|----------|----------|---------------------|------------|
| **GEM** | Multi-period momentum + defensive switch | ✅ Yes (cash/treasury) | Medium |
| **AAA** | Single-period momentum + min-variance | ❌ No (always invested) | High |
| **Equal Weight** | Buy all assets equally | ❌ No | Low |
| **Buy & Hold** | Buy once, never sell | ❌ No | Very Low |

**GEM's edge**: Combines trend-following with capital preservation.

---

### Q7: What if my backtest returns are negative?

**Possible causes**:
1. **Bad data**: Check for errors in price data
2. **Wrong parameters**: Too conservative (always in cash) or too aggressive (whipsawed)
3. **Poor asset universe**: Assets with no momentum trends
4. **High transaction costs**: Eating into profits

**What to do**:
- Review data quality
- Try different parameter combinations
- Compare to buy-and-hold benchmark (is the market itself negative?)
- Check transaction costs (reduce rebalancing frequency if high)

---

## Real-World Example

Let's say you have $100,000 to invest in 2020-2024 period with these assets:
- **IVV** (S&P 500): US large-cap stocks
- **VEU** (World ex-US): International stocks
- **AGG** (Aggregate Bond): US bonds
- **BIL** (T-Bills): Safe cash alternative

### Without GEM (Buy & Hold Equal Weight)

```
Start: $100,000 (Jan 2020)
2020 COVID crash: -35% → $65,000
2020 recovery: +70% → $110,500
2021 bull run: +25% → $138,125
2022 bear market: -20% → $110,500
2023-2024 recovery: +30% → $143,650

Final: $143,650 (+43.65%)
Max Drawdown: -35%
Sleepless nights: Many
```

### With GEM Strategy

```
Start: $100,000 (Jan 2020)
2020 COVID crash: -10% → $90,000 (switched to BIL early)
2020 recovery: +50% → $135,000 (re-entered stocks)
2021 bull run: +22% → $164,700
2022 bear market: -5% → $156,465 (switched to BIL)
2023-2024 recovery: +25% → $195,581

Final: $195,581 (+95.58%)
Max Drawdown: -12%
Sleepless nights: Few
```

**GEM advantages**:
- ✅ Higher returns (95% vs 44%)
- ✅ Lower drawdown (-12% vs -35%)
- ✅ Better Sharpe ratio (1.3 vs 0.7)
- ✅ Peace of mind during crashes

---

## Next Steps

### 1. Start Simple

Use the **default parameters** for your first backtest. See how it performs on your data.

### 2. Compare Variations

Test 3 configurations side-by-side:
- Conservative (strict parameters)
- Balanced (default)
- Aggressive (lenient parameters)

### 3. Analyze Results

Look at:
- Which configuration has the best Sharpe ratio?
- Which has the lowest drawdown?
- How often does each go defensive?

### 4. Paper Trade

Before using real money:
- Run the strategy in real-time with fake money (paper trading)
- Track actual trades for 3-6 months
- Confirm it behaves as expected

### 5. Go Live (Cautiously)

Start with a small portion of your portfolio:
- Month 1: 10% of capital
- Month 3: 25% of capital
- Month 6: 50% of capital (if comfortable)

---

## Getting Help

### Troubleshooting Resources

1. **Technical Reference**: See `GEM_Technical_Reference.md` for algorithm details
2. **Source Code**: `Backtester/Strategy_dual_momentum.py`
3. **Framework Docs**: `Backtester/BacktestFramework.py`

### Common Errors and Fixes

#### Error: "No assets in dataset"
```python
# Check your data loading
print(data.head())
print(data.columns)  # Are column names correct?
```

#### Error: "Insufficient data for lookback"
```python
# Check data length
print(f"Data length: {len(data)} days")
print(f"Need at least: {max(momentum_periods) + 1} days")

# Solution: Get more data or reduce lookback
'momentum_periods': [21, 63, 126]  # Drop 252 (12-month)
```

#### Warning: "Risk-free asset not found"
```python
# Check if asset exists in your data
print(data.columns)

# Solution 1: Use cash
'risk_free_asset': 'cash'

# Solution 2: Add the treasury asset to your dataset
```

---

## Summary Checklist

Before running your backtest, make sure:

- [ ] You have **daily price data** for all assets
- [ ] Data has **at least 1 year** of history (252+ days)
- [ ] Data is **clean** (no missing values, correct dates)
- [ ] You've chosen **appropriate parameters** for your risk tolerance
- [ ] You've set **realistic transaction costs** (commission parameter)
- [ ] You understand what **defensive allocation** means (cash vs treasury)

---

**Good luck with your backtesting!** 🚀

Remember: Past performance doesn't guarantee future results. Always test thoroughly before investing real money.

---

**Document Version**: 2.0  
**Last Updated**: November 5, 2025  
**Author**: APPM Individual Project  
**For Technical Details**: See `GEM_Technical_Reference.md`
