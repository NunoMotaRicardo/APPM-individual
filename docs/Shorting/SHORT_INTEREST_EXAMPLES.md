# Example: Using Short Interest in Backtests

This example shows how to use the new short interest feature with your existing backtesting code.

## Example 1: Running a backtest with short interest

\\\python
from Backtester.BacktestFramework import BacktraderPortfolioBacktest
from Backtester.Strategy_Core import MarkowitzStrategy, GMVPStrategy
from Backtester.helpers import load_universe_data

# Load your data
universe_data = load_universe_data('data/selection3')

# Define strategies
strategies = {
    'Markowitz': MarkowitzStrategy,
    'GMVP': GMVPStrategy
}

# Create backtest with short interest
backtest = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=[universe_data],  # Your dataset(s)
    rebalance_every=21,        # Monthly rebalancing
    lookback=126,              # 6-month lookback
    warmup=252,                # 1-year warmup
    initial_cash=100000,
    commission=0.001,          # 0.1% transaction cost
    short_interest=0.03,       # 3% annual short interest (NEW!)
    interest_long=False        # Only charge shorts (NEW!)
)

# Run backtest
results = backtest.run_backtest()

# Access results as usual
for strategy_name, strategy_results in results.items():
    for dataset_name, dataset_results in strategy_results.items():
        print(f"{strategy_name} on {dataset_name}:")
        print(f"  Final Value: {dataset_results['final_value']:.2f}")
        print(f"  Sharpe Ratio: {dataset_results['performance']['Sharpe ratio']:.4f}")
\\\

## Example 2: Comparing with and without short interest

\\\python
# Run without short interest
backtest_no_interest = BacktraderPortfolioBacktest(
    strategies={'Markowitz': MarkowitzStrategy},
    datasets=[universe_data],
    short_interest=0.0  # No interest
)
results_no_interest = backtest_no_interest.run_backtest()

# Run with realistic short interest
backtest_with_interest = BacktraderPortfolioBacktest(
    strategies={'Markowitz': MarkowitzStrategy},
    datasets=[universe_data],
    short_interest=0.03  # 3% annual
)
results_with_interest = backtest_with_interest.run_backtest()

# Compare
print("Impact of Short Interest:")
print(f"Without: {results_no_interest['Markowitz']['dataset']['final_value']:.2f}")
print(f"With:    {results_with_interest['Markowitz']['dataset']['final_value']:.2f}")
\\\

## Example 3: Different rates for different scenarios

\\\python
# Easy-to-borrow stocks (large caps)
backtest_easy = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=[large_cap_data],
    short_interest=0.01  # 1% annual
)

# Hard-to-borrow stocks (small caps, high short interest)
backtest_hard = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=[small_cap_data],
    short_interest=0.15  # 15% annual
)

# Leveraged ETF (interest on both sides)
backtest_etf = BacktraderPortfolioBacktest(
    strategies=strategies,
    datasets=[etf_data],
    short_interest=0.05,   # 5% annual
    interest_long=True     # Charge on longs too!
)
\\\

## Integration with Your Existing Notebooks

### In your 2-RunBacktest.ipynb:

\\\python
# Add to your backtest configuration cell
BACKTEST_CONFIG = {
    'rebalance_every': 21*3,
    'lookback': 252//2,
    'warmup': 252,
    'initial_cash': 100000,
    'commission': 0.001,
    'short_interest': 0.03,      # ADD THIS
    'interest_long': False       # ADD THIS
}

backtest = BacktraderPortfolioBacktest(
    strategies=strategies_to_test,
    datasets=dataset_list,
    **BACKTEST_CONFIG
)
\\\

## Notes

- Default short_interest=0.0 means your existing code works unchanged
- Interest is only charged when positions are actually short (negative weights)
- For long-only strategies, short interest has no effect
- Useful for comparing long-only vs long-short strategy performance
