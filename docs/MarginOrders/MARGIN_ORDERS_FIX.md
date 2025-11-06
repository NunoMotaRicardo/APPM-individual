# Margin Order Handling - Implementation Guide

## Problem Statement

Orders were being rejected with MARGIN status because they exceeded the available budget. This occurred due to:

1. **Commission costs not being accounted for** when calculating affordable order sizes
2. **Price slippage** between order creation (at close) and execution (at next open)
3. **Insufficient cash reserves** for handling floating point precision issues

## Solutions Implemented

### 1. Commission-Aware Order Sizing (PRIMARY SOLUTION ✅)

**Location**: BacktestFramework.py → PortfolioRebalanceStrategy.rebalance_portfolio()

**What Changed**:
- Now uses roker.getcommissioninfo(data).getoperationcost(size, price) to calculate total order cost
- Iteratively finds maximum affordable shares that fit within available cash INCLUDING commission
- Accounts for commission on sell orders when projecting future available cash

**Code Changes**:
`python
# OLD: Simple calculation without commission
max_affordable_shares = int(effective_cash // buffered_price)

# NEW: Iterative calculation including commission
max_affordable_shares = 0
test_size = int(effective_cash // buffered_price)

while test_size > 0:
    operation_cost = asset['comminfo'].getoperationcost(test_size, buffered_price)
    if operation_cost <= effective_cash:
        max_affordable_shares = test_size
        break
    test_size = int(test_size * 0.95)  # Reduce by 5% each iteration
`

**Benefits**:
- ✅ Prevents MARGIN orders by accurate cost prediction
- ✅ Works with any commission scheme
- ✅ Predictable and transparent behavior
- ✅ No broker parameter changes needed

### 2. Broker checksubmit Parameter (OPTIONAL)

**Location**: BacktraderPortfolioBacktest.__init__()

**What Changed**:
- Added checksubmit parameter (default: True)
- When False, broker can auto-adjust order sizes (NOT RECOMMENDED)

**Usage**:
`python
backtest = BacktraderPortfolioBacktest(
    portfolios={...},
    datasets=[...],
    checksubmit=True  # Keep True for predictable behavior
)
`

**Recommendation**: Keep checksubmit=True and rely on proper order sizing instead.

### 3. Enhanced Slippage Buffer (EXISTING, ENHANCED)

**Location**: Strategy parameter uy_slippage_buffer

**What It Does**:
- Adds safety margin for price gaps between order creation and execution
- Default: 0.01 (1%)
- Now works with commission calculation for better accuracy

## Backtrader Documentation References

### Key Broker Methods Used:

1. **getcommissioninfo(data)**: Returns CommissionInfo object for a data feed
   - Docs: https://www.backtrader.com/docu/broker/#getcommissioninfodata

2. **getoperationcost(size, price)**: Calculates total cost including commission
   - Docs: https://www.backtrader.com/docu/commission-schemes/commission-schemes/#getoperationcostsize-price

3. **set_checksubmit(checksubmit)**: Controls order validation before acceptance
   - Docs: https://www.backtrader.com/docu/broker/#set_checksubmitchecksubmit
   - Parameter: checksubmit (default: True) - check margin/cash before accepting orders

### Order Status Reference:

From https://www.backtrader.com/docu/order/#order-status-values

- **Order.Margin**: "the order execution would imply a margin call and the previously accepted order has been taken off the system"

## Testing Recommendations

### 1. Verify No MARGIN Orders

Check the order_history analyzer after running backtests:

`python
results = backtest.run_backtest()

for strategy_name in results:
    for dataset_name in results[strategy_name]:
        order_summary = results[strategy_name][dataset_name]['order_history']['summary']
        margin_orders = order_summary.get('margin_orders', 0)
        
        if margin_orders > 0:
            print(f"WARNING: {strategy_name} on {dataset_name} had {margin_orders} margin orders")
`

### 2. Compare Portfolio Values

Ensure portfolio values remain consistent and orders execute as expected:

`python
for strategy_name in results:
    for dataset_name in results[strategy_name]:
        final_value = results[strategy_name][dataset_name]['final_value']
        print(f"{strategy_name} on {dataset_name}: Final Value = ")
`

### 3. Monitor Cash Utilization

Check if the strategy is efficiently using available capital:

`python
# Check weights_history to see if target allocations were achieved
weights_history = results[strategy_name][dataset_name]['weights_history']
`

## Configuration Recommendations

### Default Settings (RECOMMENDED):
`python
backtest = BacktraderPortfolioBacktest(
    portfolios={'GMVP': GMVPStrategy},
    datasets=[dataset1, dataset2],
    rebalance_every=21*3,  # Quarterly
    lookback=126,          # 6 months
    warmup=252,            # 1 year warmup
    initial_cash=100000,
    commission=0.001,      # 0.1%
    checksubmit=True       # Validate orders before acceptance
)
`

### For Volatile Markets:
`python
# Increase slippage buffer in strategy params
strategies = {
    'GMVP': (GMVPStrategy, {'buy_slippage_buffer': 0.02})  # 2% buffer
}
`

### For Conservative Approach:
Add a reserve parameter (potential future enhancement):
`python
# This would require adding a reserve parameter to reduce target allocations
# e.g., perctarget = (1.0 - reserve) / n_assets
`

## Debugging Tools

### 1. Enable Verbose Logging

Set erbose=True in strategy params:
`python
strategies = {
    'GMVP': (GMVPStrategy, {'verbose': True})
}
`

### 2. Check test_log.txt

All strategy logging goes to 	est_log.txt:
`ash
tail -f test_log.txt  # Monitor in real-time
grep MARGIN test_log.txt  # Search for margin issues
`

### 3. Analyze Order History

Access detailed order information:
`python
orders = results[strategy_name][dataset_name]['order_history']['orders']
for order in orders:
    if order['status'] == 'Margin':
        print(f"Margin order: {order}")
`

## Alternative Approaches Considered

### ❌ Disable checksubmit (NOT RECOMMENDED)
- **Why not**: Unpredictable behavior, broker may silently adjust sizes
- **When to use**: Never in production backtests

### ✅ Commission-aware sizing (IMPLEMENTED)
- **Why yes**: Predictable, accurate, maintains strategy intent
- **Trade-off**: Slightly more complex calculation

### ✅ Reserve capital parameter (PARTIALLY IMPLEMENTED via slippage_buffer)
- **Why yes**: Industry standard practice
- **Trade-off**: May reduce capital efficiency slightly

## Related Resources

1. **Conservative Rebalancing Blog Post**:
   https://www.backtrader.com/blog/2019-07-19-rebalancing-conservative/
   - Shows reserve parameter usage (5% default)

2. **Broker Documentation**:
   https://www.backtrader.com/docu/broker/
   - Details on checksubmit and cash management

3. **Commission Schemes**:
   https://www.backtrader.com/docu/commission-schemes/commission-schemes/
   - Understanding getoperationcost() method

## Summary

The implemented solution uses **commission-aware order sizing** as the primary fix, which:

✅ Prevents MARGIN orders by accurately calculating total costs  
✅ Works with any commission scheme  
✅ Maintains predictable strategy behavior  
✅ Requires no broker configuration changes  

The optional checksubmit parameter provides flexibility but is NOT recommended for normal use.

**Recommendation**: Test with current settings first. Only adjust uy_slippage_buffer if issues persist in volatile market conditions.

