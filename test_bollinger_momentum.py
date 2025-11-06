import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, r'c:\my-git\DataScience-novaIMS\APPM-individual')

from Backtester.Strategy_dual_momentum import gem_portfolio_fun

# Create synthetic test data
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=300, freq='D')

# Create 5 assets with different trends
assets = ['ASSET_A', 'ASSET_B', 'ASSET_C', 'ASSET_D', 'CASH_PROXY']
data = {}

# Strong uptrend
data['ASSET_A'] = 100 * (1 + np.cumsum(np.random.normal(0.001, 0.01, 300)))

# Moderate uptrend
data['ASSET_B'] = 100 * (1 + np.cumsum(np.random.normal(0.0005, 0.01, 300)))

# Sideways
data['ASSET_C'] = 100 + np.cumsum(np.random.normal(0, 0.01, 300))

# Downtrend
data['ASSET_D'] = 100 * (1 - np.cumsum(np.random.normal(0.0005, 0.01, 300)))

# Cash proxy (stable)
data['CASH_PROXY'] = 100 + np.cumsum(np.random.normal(0.0001, 0.001, 300))

df = pd.DataFrame(data, index=dates)

print("=" * 80)
print("TESTING DUAL MOMENTUM STRATEGY - SIMPLE vs BOLLINGER METHODS")
print("=" * 80)
print(f"\nTest Data Shape: {df.shape}")
print(f"Date Range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"\nAsset Performance (Total Return %):")
for asset in assets:
    ret = ((df[asset].iloc[-1] / df[asset].iloc[0]) - 1) * 100
    print(f"  {asset:15s}: {ret:+7.2f}%")

# Test 1: Simple Momentum
print("\n" + "-" * 80)
print("TEST 1: SIMPLE MOMENTUM METHOD")
print("-" * 80)
weights_simple, log_simple = gem_portfolio_fun(
    df,
    momentum_method='simple',
    momentum_periods=[21, 63, 126, 252],
    min_positive_periods=3,
    maximum_positions=3,
    risk_free_asset='cash'
)

print("\nWeights:")
for i, asset in enumerate(assets):
    if weights_simple[i] > 0:
        print(f"  {asset:15s}: {weights_simple[i]:.4f} ({weights_simple[i]*100:.2f}%)")

if log_simple:
    print(f"\nLog Messages:\n{log_simple}")

# Test 2: Bollinger Bands Momentum
print("\n" + "-" * 80)
print("TEST 2: BOLLINGER BANDS MOMENTUM METHOD")
print("-" * 80)
weights_bollinger, log_bollinger = gem_portfolio_fun(
    df,
    momentum_method='bollinger',
    bb_period=20,
    bb_std=2.0,
    momentum_periods=[21, 63, 126, 252],
    min_positive_periods=3,
    maximum_positions=3,
    risk_free_asset='cash'
)

print("\nWeights:")
for i, asset in enumerate(assets):
    if weights_bollinger[i] > 0:
        print(f"  {asset:15s}: {weights_bollinger[i]:.4f} ({weights_bollinger[i]*100:.2f}%)")

if log_bollinger:
    print(f"\nLog Messages:\n{log_bollinger}")

# Test 3: Comparison
print("\n" + "-" * 80)
print("COMPARISON")
print("-" * 80)
print("\nWeight Differences:")
print(f"{'Asset':<15s} {'Simple':>10s} {'Bollinger':>10s} {'Difference':>10s}")
print("-" * 50)
for i, asset in enumerate(assets):
    diff = weights_bollinger[i] - weights_simple[i]
    print(f"{asset:<15s} {weights_simple[i]:>10.4f} {weights_bollinger[i]:>10.4f} {diff:>+10.4f}")

print("\n" + "=" * 80)
print("TEST COMPLETED SUCCESSFULLY")
print("=" * 80)
