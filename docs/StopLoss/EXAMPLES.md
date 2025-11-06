# Stop Loss Examples

## Example 1: Fixed Stop Loss

strategies = {
    'DRPP_with_stops': (
        DownsideRiskParityStrategy,
        {
            'rebalance_every': 21,
            'lookback': 126,
            'use_stops': True,
            'stop_loss': 0.05,  # 5% stop loss
            'trailing_stop': False,
            'verbose': True
        }
    )
}

## Example 2: Trailing Stop (Percentage)

strategies = {
    'HRPP_trailing': (
        HRPPStrategy,
        {
            'rebalance_every': 21,
            'lookback': 126,
            'use_stops': True,
            'trailing_stop': True,
            'trail_percent': 0.03,  # 3% trailing distance
            'verbose': True
        }
    )
}

## Example 3: Trailing Stop (Fixed Amount)

strategies = {
    'AAA_trailing_fixed': (
        AdaptiveAssetAllocation,
        {
            'rebalance_every': 21,
            'lookback': 126,
            'use_stops': True,
            'trailing_stop': True,
            'trail_amount': 50.0,  # 50 point distance
            'verbose': True
        }
    )
}

## Full Backtest Example

from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_DRPP import DownsideRiskParityStrategy

# Load your data
datasets = [dataset1, dataset2, dataset3]

# Configure strategy with stop loss
strategies = {
    'DRPP_stops': (
        DownsideRiskParityStrategy,
        {
            'use_stops': True,
            'trailing_stop': True,
            'trail_percent': 0.04,  # 4% trailing
        }
    )
}

# Run backtest
backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=datasets,
    initial_cash=100000,
    commission=0.001
)

results = backtest.run_backtest()

# Check order history for stop triggers
orders = results['DRPP_stops']['dataset_1']['order_history']
print(orders['summary'])
