# STOP LOSS BUG - THE REAL FIX

## The Problem (Confirmed from Log)

Even after the first "fix", the log still showed:
```
2019-08-06, BUY EXECUTED: TUR
2019-08-06, TRAILING STOP placed for TUR
2019-08-06, Cancelled stop order for TUR  ← SAME DAY!
```

## Root Cause Analysis

The timing issue was more subtle than initially thought:

### Backtrader Execution Order (within a single bar):
1. **Strategy.next()** is called FIRST
2. **Broker processes orders** 
3. **Strategy.notify_order()** is called for each executed order

### What Was Happening:
- **Day 0 (2019-08-05):** Rebalance day
  - next() runs, is rebalance day, places BUY orders
  - Orders queued in broker
  
- **Day 1 (2019-08-06):** Orders execute
  - next() runs FIRST, processes pending stops from Day 0 (none yet)
  - Then checks: is_rebalance_day? NO (counter = 1)
  - Broker executes BUY orders
  - notify_order() queues stops in pending_stop_placements
  
- **Day 2 (2019-08-07):** PROBLEM!
  - next() runs FIRST, processes pending stops → PLACES them
  - Then checks: is_rebalance_day? YES (counter = 21)
  - Calls _cancel_stop_orders() → CANCELS them
  - All on the same bar!

## The Real Fix

**Don't place stops on rebalance days** - they'll just get cancelled anyway!

### New Logic in next():
```python
is_rebalance_day = (self.rebalance_counter == 0 or 
                    (self.rebalance_counter % self.rebalance_every == 0))

if is_rebalance_day:
    # Cancel stops, CLEAR pending queue
    if self.params.use_stops:
        self._cancel_stop_orders()
        self.pending_stop_placements = []  # Don't place these!
    
    # Do rebalancing...
    
else:
    # NOT rebalance day - safe to place pending stops
    if self.params.use_stops and self.pending_stop_placements:
        for data in self.pending_stop_placements:
            self._place_stop_order(data)
        self.pending_stop_placements = []
```

## Expected Behavior Now

### Rebalance Day (Day 0):
- Cancel old stops
- Clear pending queue
- Rebalance portfolio
- BUY orders submitted
- Orders execute → notify_order() queues new stops

### Day After Rebalance (Day 1):
- NOT rebalance day
- Process pending stops → PLACE them ✓
- Stops now active

### Normal Days (Day 2-20):
- Stops remain active
- Protect positions

### Next Rebalance (Day 21):
- Cancel stops
- Rebalance
- Cycle repeats

## Timeline Fix

**BEFORE (Broken):**
```
Day 21: next() places stops → then cancels stops (SAME BAR)
```

**AFTER (Fixed):**
```
Day 21: next() skips stop placement → cancels old stops → rebalances
Day 22: next() places new stops ✓
```

## Testing

Run backtest and verify:
- ✅ No "placed → cancelled" on same day
- ✅ Stops placed day after rebalance
- ✅ Stops active for 20 days between rebalances
- ✅ Stops protect positions during market moves

---

**Status:** THIS fix should finally resolve the issue!
