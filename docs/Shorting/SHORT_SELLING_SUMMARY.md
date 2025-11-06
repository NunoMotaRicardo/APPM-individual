# Short Selling Feature - Quick Reference

## What Was Added
Short selling capability for three portfolio strategies:
- ✅ **Markowitz** (Mean-Variance)
- ✅ **GMVP** (Global Minimum Variance)  
- ✅ **MSRP** (Maximum Sharpe Ratio)

## Default Behavior
**All strategies remain long-only by default** - backward compatible with existing code.

## How to Enable Short Selling

### Correct Method (Following DRPP/AAA Pattern)
Pass `allow_short=True` via `portfolio_func_kwargs`:

```python
bt_strategies = {
    "Markowitz_Short": (MarkowitzStrategy, {
        'portfolio_func_kwargs': {'allow_short': True}
    }),
}
```

### What Changed
1. **Portfolio functions** now accept `allow_short` parameter (default: False)
2. **Strategy classes** remain simple (no custom params at class level)
3. **Parameter passing** via `portfolio_func_kwargs` (consistent with AAA/DRPP)

## Complete Example

```python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_Core import GMVPStrategy

bt_strategies = {
    # Long-only (default)
    "GMVP_Long": (GMVPStrategy, {}),
    
    # Short-selling enabled
    "GMVP_Short": (GMVPStrategy, {
        'portfolio_func_kwargs': {'allow_short': True},
        'buy_slippage_buffer': 0.05
    }),
}

bt_backtest = BacktraderPortfolioBacktest(
    strategies=bt_strategies,
    datasets=my_dataset_list,
    rebalance_every=21,
    lookback=126,
    initial_cash=100000,
    commission=0.001,
    short_interest=0.03,  # Important! Add borrowing costs
    interest_long=False
)

results = bt_backtest.run_backtest()
```

## Key Points

1. **No class-level params** - consistent with DRPP and AAA strategy patterns
2. **Use `portfolio_func_kwargs`** - this is how custom parameters are passed
3. **Always set `short_interest`** when using short selling
4. **Backward compatible** - existing code works without changes

## Implementation Details

### Markowitz
```python
# Constraint changes:
if allow_short:
    constraints = [cp.sum(w) == 1]  # No non-negativity
else:
    constraints = [w >= 0, cp.sum(w) == 1]  # Long-only
```

### GMVP
```python
# Normalization changes:
if allow_short:
    w = w / np.sum(w)  # Allow negatives
else:
    w = np.abs(w) / np.sum(np.abs(w))  # Force positive
```

### MSRP
```python
# Variable definition changes:
if allow_short:
    w = cp.Variable(n)  # Unconstrained
else:
    w = cp.Variable(n, nonneg=True)  # Non-negative
```

## Files Modified
- `Backtester/Strategy_Core.py` - Added `allow_short` parameter to 3 strategies
- `docs/SHORT_SELLING_GUIDE.md` - Comprehensive usage guide
- `docs/SHORT_SELLING_SUMMARY.md` - This quick reference

## See Also
- `SHORT_SELLING_GUIDE.md` - Full documentation with examples
- `SHORT_INTEREST_IMPLEMENTATION.md` - Short interest cost configuration
