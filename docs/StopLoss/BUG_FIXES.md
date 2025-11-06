# STOP LOSS BUG FIXES - CRITICAL ISSUES RESOLVED

## Date: 2025-10-23

## Issues Identified from Log Analysis

### Problem 1: Same-Bar Stop Cancellation ⚠️ CRITICAL
**Symptom:**
```
2022-08-16, BUY EXECUTED: ECH, Size: 543
2022-08-16, TRAILING STOP placed for ECH: 3.0%
2022-08-16, Cancelled stop order for ECH
```

**Root Cause:**
- Stops were placed immediately in notify_order() when buy executes
- Rebalancing calls _cancel_stop_orders() in the SAME bar
- Result: Stops are placed and immediately cancelled on the same day

**Fix:**
- Queue stop placements in pending_stop_placements list
- Process the queue at START of next bar in next()
- Move _cancel_stop_orders() to BEFORE rebalancing in next()
- This ensures proper timing: Cancel → Rebalance → Place Stops (next bar)

---

### Problem 2: Position Accumulation Issues
**Symptom:**
```
2022-08-17, BUY EXECUTED: ECH, Size: 8
2022-08-17, TRAILING STOP placed for ECH: 3.0%
(previous position of 543 shares already existed)
```

**Root Cause:**
- Entry price was overwritten when accumulating positions
- Multiple stop orders created for same asset
- Stop size didn't match total position

**Fix:**
- Calculate weighted average entry price when accumulating
- Formula: (old_entry × old_shares + new_price × new_shares) / total_shares
- Always cancel existing stop before placing new one
- Use position.size (total position) for stop order size

---

### Problem 3: No Distinction Between Stop Sells and Rebalance Sells
**Symptom:**
- Stops trigger, position closed, but entry price not properly cleared
- Rebalancing might buy back immediately after stop loss

**Root Cause:**
- notify_order() treated all sells the same way
- Didn't check if position was fully closed

**Fix:**
- Check getposition(data).size after sell execution
- Only clear entry_price and stop_order when position == 0
- This allows partial sells during rebalancing while maintaining stops

---

## Code Changes Summary

See BUG_FIXES.md for complete details.

## Testing Recommendations

1. Monitor log for same-day cancel patterns
2. Verify stop sizes match positions
3. Check weighted average entry prices
4. Confirm stops don't interfere with rebalancing

