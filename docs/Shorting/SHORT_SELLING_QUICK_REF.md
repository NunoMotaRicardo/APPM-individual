# Short Selling - Quick Reference

## Summary

Added llow_short parameter to three strategies: **Markowitz**, **GMVP**, and **MSRP**.

## Quick Usage

### Enable short selling for a strategy:

`python
bt_strategies = {
    "MVP_Short": (MarkowitzStrategy, {
        'allow_short': True,
        'buy_slippage_buffer': 0.07,
    }),
    "GMVP_Short": (GMVPStrategy, {
        'allow_short': True,
        'buy_slippage_buffer': 0.05,
    }),
    "MSRP_Short": (MaximumSharpeRatioStrategy, {
        'allow_short': True,
        'buy_slippage_buffer': 0.05,
    }),
}
`

### Add short interest costs:

`python
bt_backtest = BacktraderPortfolioBacktest(
    strategies=bt_strategies,
    datasets=my_dataset_list,
    short_interest=0.03,  # 3% annual
    interest_long=False,
    # ... other parameters
)
`

## Key Points

- **Default**: llow_short=False (long-only, backward compatible)
- **Strategies supporting shorts**: Markowitz, GMVP, MSRP only
- **Recommended**: Use short_interest parameter when enabling shorts
- **Margin buffer**: Increase uy_slippage_buffer to 0.05-0.10 for short strategies

## Weight Interpretation

- **Long-only**: All weights >= 0, sum = 1
- **With shorts**: Positive = long, negative = short, sum = 1 (net exposure)

Example: [0.6, 0.8, -0.4] = 60% long, 80% long, 40% short (180% gross exposure)

## See Also

- docs/SHORT_SELLING_IMPLEMENTATION.md - Complete documentation
- docs/SHORT_INTEREST_IMPLEMENTATION.md - Short interest costs details
