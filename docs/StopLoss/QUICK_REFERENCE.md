# Stop Loss Quick Reference

## New Parameters

- use_stops: Enable stop loss (default: False)
- stop_loss: Fixed stop percentage (default: 0.05 = 5%)
- trailing_stop: Use trailing stop (default: False)
- trail_percent: Trailing % distance (default: 0.0)
- trail_amount: Trailing fixed distance (default: 0.0)

## Examples

Fixed 5% stop:
  use_stops=True, stop_loss=0.05

Trailing 3% stop:
  use_stops=True, trailing_stop=True, trail_percent=0.03

Trailing 50pt stop:
  use_stops=True, trailing_stop=True, trail_amount=50.0

## Documentation
https://www.backtrader.com/blog/posts/2017-03-22-stoptrail/stoptrail/
