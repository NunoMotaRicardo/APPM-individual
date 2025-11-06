# Bollinger Bands Momentum Method

## Overview

The dual momentum strategy (Strategy_dual_momentum.py) now supports two methods for calculating momentum:
1. **Simple Momentum** (default): Traditional return-based calculation
2. **Bollinger Bands Momentum** (new): Position-relative calculation using Bollinger Bands

## Bollinger Bands Method

### Concept

The Bollinger Bands momentum method evaluates momentum based on an asset's position relative to its Bollinger Bands at each lookback period. This approach captures both trend direction and volatility-adjusted strength.

### Calculation

For each lookback period and each asset:

1. **Calculate Bollinger Bands** over the specified b_period (default: 20 days):
   - **Middle Band**: Simple Moving Average (SMA) of prices
   - **Upper Band**: Middle Band + (bb_std × Standard Deviation)
   - **Lower Band**: Middle Band - (bb_std × Standard Deviation)
   - Where b_std is the standard deviation multiplier (default: 2.0)

2. **Calculate Momentum Score**:
   - **Positive Momentum**: When current price > middle band
   - **Negative Momentum**: When current price < middle band
   - **Score Formula**: ((price - middle_band) / (band_width / 2)) × 100
   - Where and_width = upper_band - lower_band

3. **Interpretation**:
   - Score > 0: Asset is in uptrend (above SMA)
   - Score < 0: Asset is in downtrend (below SMA)
   - Score magnitude: Strength of trend relative to volatility
   - Score ≈ +100: Price near upper band (strong uptrend)
   - Score ≈ -100: Price near lower band (strong downtrend)

### Advantages

- **Volatility-Adjusted**: Accounts for asset volatility in momentum measurement
- **Trend Confirmation**: Uses moving average as trend baseline
- **Range-Bound Scoring**: Normalized scores facilitate comparison across assets
- **Adaptive**: Band width adjusts to market conditions

### Comparison with Simple Momentum

| Aspect | Simple Momentum | Bollinger Momentum |
|--------|----------------|-------------------|
| **Calculation** | Raw percentage return | Position relative to bands |
| **Volatility** | Not considered | Inherently adjusted |
| **Baseline** | Past price | Moving average (SMA) |
| **Score Range** | Unbounded | Typically -200 to +200 |
| **Trend Strength** | Absolute price change | Relative to volatility |

## Parameters

### Core Parameters (both methods)
- momentum_periods: List of lookback periods in days (default: [21, 63, 126, 252])
- min_positive_periods: Minimum periods with positive momentum to qualify (default: 3)
- 	reasury_threshold: Minimum return threshold (default: 0.0)
- maximum_positions: Maximum number of positions (default: 6)
- isk_free_asset: 'cash' or symbol name (default: 'cash')

### Method Selection
- momentum_method: 'simple' or 'bollinger' (default: 'simple')

### Bollinger Bands Specific
- b_period: Bollinger Bands lookback period in days (default: 20)
- b_std: Standard deviation multiplier (default: 2.0)

## Usage Examples

### Example 1: Simple Momentum (Default)
\\\python
strategy = GlobalEquitiesMomentumStrategy()
kwargs = {
    'momentum_method': 'simple',
    'momentum_periods': [21, 63, 126, 252],
    'min_positive_periods': 3,
    'risk_free_asset': 'cash'
}
\\\

### Example 2: Bollinger Bands Momentum
\\\python
strategy = GlobalEquitiesMomentumStrategy()
kwargs = {
    'momentum_method': 'bollinger',
    'bb_period': 20,
    'bb_std': 2.0,
    'momentum_periods': [21, 63, 126, 252],
    'min_positive_periods': 3,
    'risk_free_asset': 'BIL'
}
\\\

### Example 3: Conservative Bollinger Settings
\\\python
# Tighter bands (1.5 std) for earlier signals
kwargs = {
    'momentum_method': 'bollinger',
    'bb_period': 20,
    'bb_std': 1.5,
    'min_positive_periods': 2,
    'risk_free_asset': 'cash'
}
\\\

### Example 4: Long-term Bollinger Trend
\\\python
# Longer BB period for smoother trend detection
kwargs = {
    'momentum_method': 'bollinger',
    'bb_period': 50,
    'bb_std': 2.5,
    'momentum_periods': [63, 126, 252],
    'min_positive_periods': 2,
    'maximum_positions': 4
}
\\\

## Data Requirements

### Simple Momentum
- Minimum data needed: max(momentum_periods) + 1 days

### Bollinger Bands Momentum
- Minimum data needed: max(momentum_periods) + bb_period days
- Example with defaults: max(252) + 20 = 272 days

## Implementation Notes

1. **Lookback Logic**: For Bollinger method, the bands are calculated over data ENDING at each lookback point
2. **NaN Handling**: Invalid scores (inf, -inf, NaN) are handled gracefully
3. **Error Messages**: Informative warnings when insufficient data available
4. **Asset Filtering**: Risk-free assets excluded from momentum calculation

## Strategy Behavior

Both methods follow the same GEM strategy logic:
1. Calculate momentum scores across multiple periods
2. Count positive momentum periods for each asset
3. Filter assets meeting min_positive_periods threshold
4. Compare qualified assets against treasury threshold
5. Select top maximum_positions assets by average momentum
6. Allocate equal weights or move to risk-free asset

## Performance Considerations

- **Bollinger method** may be more stable in volatile markets (volatility-adjusted)
- **Simple method** may be more responsive to strong trends (raw returns)
- Consider testing both methods with your specific universe and time period

## References

- Bollinger Bands: John Bollinger, "Bollinger on Bollinger Bands" (2001)
- Dual Momentum: Gary Antonacci, "Dual Momentum Investing" (2014)
- Original GEM strategy: references/gem.py

## Version History

- **v1.0** (2025-11-05): Initial implementation of Bollinger Bands momentum method
