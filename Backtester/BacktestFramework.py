# Import backtrader
from bdb import effective
import backtrader as bt
import backtrader.analyzers as btanalyzers

import pandas as pd
import numpy as np
import math
from typing import Any
from scipy import stats

#from Backtester.Strategy_Core import EqualWeightStrategy

"""
MARGIN ORDER HANDLING IMPROVEMENTS
===================================

This framework now includes robust handling to prevent MARGIN orders (orders rejected due to 
insufficient cash). Two complementary approaches have been implemented:

1. **Commission-Aware Order Sizing** (Primary Solution - RECOMMENDED):
   - Uses broker's `getcommissioninfo()` and `getoperationcost()` methods
   - Calculates the TOTAL cost of an order including commission before submission
   - Iteratively finds the maximum affordable shares that fit within available cash
   - Accounts for commission on sell orders when projecting available cash
   
   This approach ensures orders are sized correctly from the start, preventing margin calls
   due to underestimating costs.

2. **Broker checksubmit Parameter** (Optional):
   - New parameter `checksubmit` in BacktraderPortfolioBacktest class (default: True)
   - When True: Broker validates orders before accepting (current behavior)
   - When False: Broker may auto-adjust order sizes (less predictable, NOT recommended)
   
   Keeping checksubmit=True with proper order sizing is the recommended approach.

3. **Slippage Buffer** (Existing, Enhanced):
   - `buy_slippage_buffer` parameter in strategies (default: 0.01 = 1%)
   - Accounts for price gaps between order creation and execution
   - Now works in conjunction with commission calculation for better accuracy

USAGE RECOMMENDATIONS:
----------------------
- Keep default settings: checksubmit=True, buy_slippage_buffer=0.01
- Increase slippage buffer if experiencing margin issues in volatile markets
- Monitor order_history analyzer to track margin rejections
- Consider adding reserve capital parameter for additional safety margin

REFERENCES:
-----------
- Backtrader Broker Documentation: https://www.backtrader.com/docu/broker/
- Conservative Rebalancing Approach: https://www.backtrader.com/blog/2019-07-19-rebalancing-conservative/
- Commission Schemes: https://www.backtrader.com/docu/commission-schemes/commission-schemes/
"""

# Custom Analyzer for Portfolio Weights Tracking
class PortfolioWeightsAnalyzer(bt.Analyzer):
    """
    Custom analyzer to track portfolio weights and rebalancing dates
    """
    
    def __init__(self):
        super(PortfolioWeightsAnalyzer, self).__init__()
        self.rebalance_history = []
        
    def create_analysis(self):
        """Create the structure to hold analysis results"""
        self.rets = {
            'rebalancing_dates': [],
            'portfolio_weights': [],
            'asset_names': []
        }
        
    def notify_rebalance(self, date, weights, asset_names):
        """
        Method called by strategy when rebalancing occurs
        
        Parameters:
        - date: datetime object of rebalancing date
        - weights: numpy array of portfolio weights
        - asset_names: list of asset names corresponding to weights
        """
        # Convert date to string immediately to ensure proper JSON serialization
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
            
        self.rets['rebalancing_dates'].append(date_str)
        self.rets['portfolio_weights'].append(weights.copy() if hasattr(weights, 'copy') else list(weights))
        self.rets['asset_names'].append(asset_names.copy() if hasattr(asset_names, 'copy') else list(asset_names))
        
    def get_analysis(self):
        """Return the analysis results"""
        from collections import OrderedDict
        # Convert to OrderedDict to match backtrader's expected return type
        return OrderedDict(self.rets)


# Custom Analyzer for Asset Values Tracking
class AssetValuesAnalyzer(bt.Analyzer):
    """
    Custom analyzer to track asset values at the beginning of each rebalancing period.
    Captures prices, positions, and values for all assets in the portfolio.
    """
    
    def __init__(self):
        super(AssetValuesAnalyzer, self).__init__()
        
    def create_analysis(self):
        """Create the structure to hold analysis results"""
        self.rets = {
            'rebalancing_dates': [],
            'asset_names': [],
            'asset_prices': [],      # Price of each asset at rebalancing
            'asset_positions': [],   # Number of shares held for each asset
            'asset_values': [],      # Value (price × shares) for each asset
            'portfolio_value': [],   # Total portfolio value
            'cash': []               # Cash available at rebalancing
        }
        
    def notify_rebalance(self, date, prices, positions, asset_names, portfolio_value, cash):
        """
        Method called by strategy when rebalancing occurs
        
        Parameters:
        - date: datetime object of rebalancing date
        - prices: dict or array of asset prices
        - positions: dict or array of asset positions (shares)
        - asset_names: list of asset names
        - portfolio_value: total portfolio value
        - cash: available cash
        """
        # Convert date to string for JSON serialization
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # Convert prices and positions to lists if they're numpy arrays or dicts
        if isinstance(prices, dict):
            prices_list = [prices.get(name, 0.0) for name in asset_names]
        else:
            prices_list = list(prices) if hasattr(prices, '__iter__') else [prices]
            
        if isinstance(positions, dict):
            positions_list = [positions.get(name, 0.0) for name in asset_names]
        else:
            positions_list = list(positions) if hasattr(positions, '__iter__') else [positions]
        
        # Calculate asset values (price × position)
        asset_values_list = [p * pos for p, pos in zip(prices_list, positions_list)]
        
        # Store the data
        self.rets['rebalancing_dates'].append(date_str)
        self.rets['asset_names'].append(list(asset_names))
        self.rets['asset_prices'].append(prices_list)
        self.rets['asset_positions'].append(positions_list)
        self.rets['asset_values'].append(asset_values_list)
        self.rets['portfolio_value'].append(float(portfolio_value))
        self.rets['cash'].append(float(cash))
        
    def get_analysis(self):
        """Return the analysis results"""
        from collections import OrderedDict
        return OrderedDict(self.rets)


# Custom Analyzer for Order Tracking
class OrderTrackingAnalyzer(bt.Analyzer):
    """
    Custom analyzer to track all orders (buy, sell, and other order types)
    Captures detailed information about order submission, execution, and final status
    """
    
    def __init__(self):
        super(OrderTrackingAnalyzer, self).__init__()
        self.order_counter = 0
        
    def create_analysis(self):
        """Create the structure to hold analysis results"""
        self.rets = {
            'orders': [],
            'summary': {
                'total_orders': 0,
                'buy_orders': 0,
                'sell_orders': 0,
                'completed_orders': 0,
                'cancelled_orders': 0,
                'rejected_orders': 0,
                'margin_orders': 0,
                'partial_orders': 0
            }
        }
        
    def notify_order(self, order):
        """
        Called whenever an order changes status
        
        Parameters:
        - order: backtrader order object
        """
        # Only track each order once when it reaches a final state
        if order.status in [order.Completed, order.Canceled, order.Rejected, order.Margin]:
            
            # Determine order type
            if order.isbuy():
                order_type = 'BUY'
                self.rets['summary']['buy_orders'] += 1
            elif order.issell():
                order_type = 'SELL'
                self.rets['summary']['sell_orders'] += 1
            else:
                order_type = 'OTHER'
            
            # Get order status name
            status = order.getstatusname()
            
            # Convert matplotlib date format to datetime using bt.num2date
            created_date = None
            if order.created and hasattr(order.created, 'dt') and order.created.dt:
                created_date = bt.num2date(order.created.dt).date()
            
            # Build order record
            order_record = {
                'order_id': self.order_counter,
                'asset': order.data._name if hasattr(order.data, '_name') else 'Unknown',
                'order_type': order_type,
                'status': status,
                'created_date': created_date,  
                'created_price': float(order.created.price) if order.created.price else None,
                'created_size': float(order.created.size) if order.created.size else None,
            }
            
            # Add execution details if order was completed
            if order.status == order.Completed:
                executed_date = None
                if order.executed and hasattr(order.executed, 'dt') and order.executed.dt:
                    executed_date = bt.num2date(order.executed.dt).date()
                    
                order_record.update({
                    'executed_date': executed_date,
                    'executed_price': float(order.executed.price) if order.executed.price else None,
                    'executed_size': float(order.executed.size) if order.executed.size else None,
                    'executed_value': float(order.executed.value) if order.executed.value else None,
                    'commission': float(order.executed.comm) if order.executed.comm else None,
                    'pnl': float(order.executed.pnl) if hasattr(order.executed, 'pnl') else None,
                })
                self.rets['summary']['completed_orders'] += 1
                
            elif order.status == order.Canceled:
                self.rets['summary']['cancelled_orders'] += 1
                
            elif order.status == order.Rejected:
                self.rets['summary']['rejected_orders'] += 1
                
            elif order.status == order.Margin:
                self.rets['summary']['margin_orders'] += 1
            
            # Check for partial fills
            if order.status == order.Partial:
                order_record['partial_fill'] = True
                self.rets['summary']['partial_orders'] += 1
            
            # Add to orders list
            self.rets['orders'].append(order_record)
            self.rets['summary']['total_orders'] += 1
            self.order_counter += 1
    
    def get_analysis(self):
        """Return the analysis results"""
        from collections import OrderedDict
        return OrderedDict(self.rets)

# Portfolio Rebalancing Strategy using Backtrader
class PortfolioRebalanceStrategy(bt.Strategy):
    """
    Base portfolio rebalancing strategy for backtrader
    
    Stop Loss / Trailing Stop Parameters:
    -------------------------------------
    The strategy supports protective stops to limit losses. Configure via params:
    
    - use_stops (bool): Enable stop loss functionality (default: False)
    - stop_loss (float): Fixed stop loss as percentage below entry (e.g., 0.05 = 5%)
    - trailing_stop (bool): Use trailing stop instead of fixed stop (default: False)
    - trail_percent (float): Trailing stop distance as percentage (e.g., 0.03 = 3%)
    - trail_amount (float): Trailing stop distance as fixed price (overrides trail_percent)
    
    Example configurations:
    1. Fixed 5% stop loss: use_stops=True, stop_loss=0.05
    2. 3% trailing stop: use_stops=True, trailing_stop=True, trail_percent=0.03
    3. 50 point trailing: use_stops=True, trailing_stop=True, trail_amount=50.0
    """
    params: Any = dict(
        rebalance_every=21,      # Rebalance every ~1 month (21 trading days)
        lookback=252 // 4,       # 6 months lookback
        warmup=None,             # Warmup period (defaults to lookback if None)
        portfolio_func=None,     # Portfolio optimization function
        dataset_assets=None,     # List of asset names
        buy_slippage_buffer=0.01, # Extra cushion to protect against price surprises on buys
        verbose=False,
        portfolio_func_kwargs={}, # Custom kwargs for portfolio function (e.g., method, threshold, linkage_method)
        # Stop loss parameters
        use_stops=False,         # Enable stop loss orders
        stop_loss=0.05,          # Stop loss percentage (5% default)
        trailing_stop=False,     # Use trailing stop instead of fixed stop
        trail_percent=0.0,       # Trailing stop percentage distance
        trail_amount=0.0,        # Trailing stop fixed price distance (overrides trail_percent)
        strategy_name='test',
    )

    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        with open(f'{self.strategy_name}_log.txt', 'a') as f:
            f.write(f'{dt.isoformat()}, {txt}\n')

    def __init__(self):
        # Keep track of rebalancing
        self.rebalance_counter = 0
        self.current_weights = None
        self.asset_names = self.params.dataset_assets or [d._name for d in self.datas]
        self.verbose = self.params.verbose
        self.strategy_name = self.params.strategy_name

        # Set warmup period (default to lookback if not specified)
        self.warmup_period = self.params.warmup if self.params.warmup is not None else self.params.lookback
        self.buy_slippage_buffer = self.params.buy_slippage_buffer
        #self.linkage_method = self.params.linkage_method
        self.rebalance_every = self.params.rebalance_every
        # Track orders to prevent multiple orders
        self.orders = []
        
        # Track stop orders for each data feed
        self.stop_orders = {data: None for data in self.datas}
        self.entry_prices = {data: None for data in self.datas}  # Track entry prices for stop calculation
        self.pending_stop_placements = []  # Queue stops to be placed after current bar completes
        
        init_msg = f"Initialized strategy with slippage buffer: {self.buy_slippage_buffer}, warmup: {self.warmup_period}, lookback: {self.params.lookback}"
        if self.params.use_stops:
            if self.params.trailing_stop:
                if self.params.trail_amount > 0:
                    init_msg += f", trailing stop: {self.params.trail_amount} points"
                else:
                    init_msg += f", trailing stop: {self.params.trail_percent*100}%"
            else:
                init_msg += f", fixed stop loss: {self.params.stop_loss*100}%"
        self.log(init_msg)
        
    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted, order.Partial]:
            # Order is submitted/accepted - no action needed
            return

        elif order.status in [order.Completed]:
            if order.isbuy():
                if self.verbose:
                    self.log(f'BUY EXECUTED: {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}, Total Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                # Track entry price for stop loss calculation (weighted average for accumulation)
                if self.params.use_stops:
                    current_position = self.getposition(order.data).size - order.executed.size  # Position before this buy
                    new_shares = order.executed.size
                    
                    if current_position > 0 and self.entry_prices[order.data] is not None:
                        # Accumulating: calculate weighted average entry price
                        old_entry = self.entry_prices[order.data]
                        weighted_entry = (old_entry * current_position + order.executed.price * new_shares) / (current_position + new_shares)
                        self.entry_prices[order.data] = weighted_entry
                    else:
                        # New position
                        self.entry_prices[order.data] = order.executed.price
                    
                    # Queue stop placement for next bar (after all orders on current bar complete)
                    if order.data not in self.pending_stop_placements:
                        self.pending_stop_placements.append(order.data)
                        
            elif order.issell():
                if self.verbose:
                    self.log(f'SELL EXECUTED: {order.data._name}, Size: {order.executed.size}, Price: {order.executed.price:.2f}, Total Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                # Check if this was a stop order that closed the position
                position_after = self.getposition(order.data).size
                if position_after == 0:
                    # Position fully closed - clear entry price and stop
                    if order.data in self.entry_prices:
                        self.entry_prices[order.data] = None
                    if order.data in self.stop_orders:
                        self.stop_orders[order.data] = None
            
            if order in self.orders:
                self.orders.remove(order)

        elif order.status in [order.Margin]:
            ## implement margin handling if needed       
            self.log(f'Order {order.data._name} to be cancelled, status: {str(order.getstatusname()).upper()}')
            if order in self.orders:
                self.orders.remove(order)
            return

        # Remove completed/cancelled orders
        elif order.status in [order.Rejected]:
            self.log(f'Order {order.data._name} status: {str(order.getstatusname()).upper()}')
            if order in self.orders:
                self.orders.remove(order)
            # Also remove from stop_orders if it was a stop
            if order.data in self.stop_orders and self.stop_orders[order.data] == order:
                self.stop_orders[order.data] = None

        # Remove completed/cancelled orders
        elif order.status in [order.Canceled]:
            if self.verbose:
                self.log(f'Order {order.data._name} status: {str(order.getstatusname()).upper()}')
            if order in self.orders:
                self.orders.remove(order)
            # Also remove from stop_orders if it was a stop
            if order.data in self.stop_orders and self.stop_orders[order.data] == order:
                self.stop_orders[order.data] = None

        else:
            self.log(f'Order {order.data._name} status: {str(order.getstatusname())}')
    
    def _place_stop_order(self, data):
        """Place a stop loss or trailing stop order for a position"""
        if not self.params.use_stops:
            return
        
        # Cancel any existing stop order for this data
        if data in self.stop_orders and self.stop_orders[data] is not None:
            self.cancel(self.stop_orders[data])
            self.stop_orders[data] = None
        
        position = self.getposition(data)
        if position.size <= 0:
            return  # No position to protect
        
        entry_price = self.entry_prices.get(data)
        if entry_price is None:
            return
        
        if self.params.trailing_stop:
            # Create trailing stop order for ENTIRE position
            if self.params.trail_amount > 0:
                # Fixed distance trailing stop
                stop_order = self.sell(
                    data=data,
                    size=position.size,
                    exectype=bt.Order.StopTrail,
                    trailamount=self.params.trail_amount
                )
                if self.verbose:
                    self.log(f'TRAILING STOP placed for {data._name}: size={position.size}, trail={self.params.trail_amount} points')
            else:
                # Percentage distance trailing stop
                stop_order = self.sell(
                    data=data,
                    size=position.size,
                    exectype=bt.Order.StopTrail,
                    trailpercent=self.params.trail_percent
                )
                if self.verbose:
                    self.log(f'TRAILING STOP placed for {data._name}: size={position.size}, trail={self.params.trail_percent*100}%')
        else:
            # Create fixed stop loss order for ENTIRE position
            stop_price = entry_price * (1.0 - self.params.stop_loss)
            stop_order = self.sell(
                data=data,
                size=position.size,
                exectype=bt.Order.Stop,
                price=stop_price
            )
            if self.verbose:
                self.log(f'STOP LOSS placed for {data._name}: size={position.size}, price={stop_price:.2f} ({self.params.stop_loss*100}% below entry={entry_price:.2f})')
        
        self.stop_orders[data] = stop_order
    
    def _cancel_stop_orders(self):
        """Cancel all active stop orders"""
        for data, stop_order in self.stop_orders.items():
            if stop_order is not None:
                self.cancel(stop_order)
                if self.verbose:
                    self.log(f'Cancelled stop order for {data._name}')
                self.stop_orders[data] = None

    def get_portfolio_data(self):
        """Get current portfolio data for optimization (includes dates as index)"""
        lookback_len = min(self.params.lookback, len(self.datas[0]))
        if lookback_len <= 1:
            return None

        # Build date index from the primary data feed (assumes feeds are aligned)
        index_range = range(-lookback_len, 0)
        
        dates = [self.datas[0].datetime.date(i) for i in index_range]
        #print(f"Building portfolio data for dates: {dates[0]} to {dates[-1]}")
        # Build DataFrame with close prices for all assets
        data_dict = {}
        for i, data in enumerate(self.datas):
            asset_name = self.asset_names[i]
            closes = [data.close[j] for j in index_range]
            data_dict[asset_name] = closes

        df = pd.DataFrame(data_dict, index=pd.to_datetime(dates))
        return df
    
    def calculate_target_weights(self):
        """Calculate target portfolio weights"""
        log_message = ""
        if self.params.portfolio_func is None:
            # Equal weight default
            n_assets = len(self.datas)
            return np.ones(n_assets) / n_assets, log_message

        try:
            portfolio_data = self.get_portfolio_data()
            
            # Check if we have sufficient data
            if portfolio_data is None:
                self.log('ERROR:Insufficient data for portfolio optimization, using zeros')
                n_assets = len(self.datas)
                return np.zeros(n_assets), log_message

            # Call portfolio function with custom kwargs
            portfolio_func_kwargs = self.params.portfolio_func_kwargs or {}
            weights, log_message = self.params.portfolio_func(portfolio_data, **portfolio_func_kwargs)
            if log_message != "":
                self.log(f'Portfolio function message: {log_message}')
            weights = np.asarray(weights, dtype=float)

            n_assets = len(self.datas)
            if weights.ndim != 1 or weights.size != n_assets:
                self.log(f'ERROR: Portfolio function returned {weights.size} weights, expected {n_assets}. Do nothing.')
                return np.zeros(n_assets), "ERROR: Incorrect number of weights"

            # Clean up any numerical artefacts
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.clip(weights, 0.0, None)

            total_weight = weights.sum()
            if total_weight > 1.0:
                #self.log(f'WARNING: Portfolio weights sum to {total_weight:.4f} > 1.0, renormalizing.')
                # Renormalize to avoid exceeding available capital
                weights = weights / total_weight

            return weights, log_message
        except Exception as e:
            self.log(f'ERROR: Portfolio optimization failed: {e}')
            # Fallback to equal weights
            n_assets = len(self.datas)
            return np.zeros(n_assets), "ERROR: Fallback to zeros due to exception"
    
    def rebalance_portfolio(self, target_weights):
        """Rebalance portfolio to target weights"""
        # Cancel any pending orders (not stops - those are handled in next())
        for order in self.orders[:]:
            self.log(f'Cancelling pending order for {order.data._name}')
            self.cancel(order)
        
        total_value = self.broker.getvalue()
        available_cash = self.broker.getcash()
        if total_value <= 0:
            self.log('Total portfolio value is non-positive. Skipping rebalance.')
            return

        slippage_buffer = max(0.0, float(self.buy_slippage_buffer))
        buffer_multiplier = 1.0 + slippage_buffer

        # Get commission info for cost calculations
        #comminfo = self.broker.getcommissioninfo(self.datas[0])

        portfolio_state = []
        for data, target_weight in zip(self.datas, target_weights):
            price = data.close[0]
            if price <= 0:
                self.log(f'ERROR: Skipping {data._name} due to non-positive price: {price}')
                continue

            current_position = self.getposition(data)
            current_shares = int(round(current_position.size))
            target_weight = max(0.0, float(target_weight))
            target_value = total_value * target_weight / buffer_multiplier  # Adjust target value for buy buffer
            target_shares = int(math.floor(target_value / price)) if target_value > 0 else 0

            portfolio_state.append({
                'data': data,
                'name': data._name,
                'price': price,
                'target_value': target_value,
                'current_shares': current_shares,
                'target_shares': target_shares,
                'planned_shares': current_shares,
                'comminfo': self.broker.getcommissioninfo(data)
            })
        #print(f'Portfolio state before rebalance: {(pd.DataFrame(portfolio_state)[['name', 'current_shares', 'target_shares']])}')
        expected_cash = available_cash
        cash_reserved_for_buys = 0.0

        # First, execute sells to raise cash
        for asset in portfolio_state:
            share_diff = asset['target_shares'] - asset['planned_shares']
            if share_diff >= 0:
                continue

            shares_to_sell = int(abs(share_diff))
            if shares_to_sell <= 0:
                continue

            order = self.sell(data=asset['data'], size=shares_to_sell)
            if order:
                self.orders.append(order)
                # Account for commission on sell
                sell_commission = asset['comminfo'].getcommission(shares_to_sell, asset['price'])
                expected_cash += shares_to_sell * asset['price'] - sell_commission
                asset['planned_shares'] -= shares_to_sell

        #expected_cash=expected_cash / buffer_multiplier  # Adjust expected cash for buy buffer

        # Then, execute buys subject to available (and expected) cash
        for asset in portfolio_state:
            share_diff = asset['target_shares'] - asset['planned_shares']
            if share_diff <= 0:
                continue

            buffered_price = asset['price'] * buffer_multiplier
            effective_cash = expected_cash - cash_reserved_for_buys
            if buffered_price <= 0 or effective_cash <= 0:
                self.log(f'ERROR: Not enough cash to buy {asset["name"]}.')
                break

            # Calculate maximum shares we can afford INCLUDING commission
            # Use binary search or iterative approach to find max affordable shares
            max_affordable_shares = 0
            test_size = int(effective_cash // buffered_price)
            
            # Iteratively reduce size until we find a size that fits within budget
            while test_size > 0:
                operation_cost = asset['comminfo'].getoperationcost(test_size, buffered_price)
                if operation_cost <= effective_cash:
                    max_affordable_shares = test_size
                    
                    break
                test_size = int(test_size * 0.95)  # Reduce by 5% each iteration
            
            if max_affordable_shares <= 0:
                self.log(f'WARNING: 1 Not enough cash to buy {asset["name"]} (needed {share_diff} shares).')
                continue

            shares_to_buy = min(share_diff, max_affordable_shares)
            if shares_to_buy <= 0:
                self.log(f'WARNING: 2 Not enough cash to buy {asset["name"]} (needed {share_diff} shares).')
                continue

            order = self.buy(data=asset['data'], size=shares_to_buy)
            if order:
                self.orders.append(order)
                # Reserve cash including commission
                operation_cost = asset['comminfo'].getoperationcost(shares_to_buy, buffered_price)
                cash_reserved_for_buys += operation_cost
                asset['planned_shares'] += shares_to_buy

            if shares_to_buy < share_diff:
                if self.verbose:
                    self.log(f'Cash constraint: allocated {shares_to_buy} of {share_diff} desired shares for {asset["name"]}.')
    
    def notify_weights_analyzer(self, target_weights):
        """Notify weights analyzer of rebalancing event"""
        # Check if analyzers are available and find the PortfolioWeightsAnalyzer
        try:
            if hasattr(self, 'analyzers') and hasattr(self.analyzers, 'weights'):
                # Access the weights analyzer directly by name
                weights_analyzer = self.analyzers.weights
                if weights_analyzer:
                    current_date = self.datas[0].datetime.date(0)
                    weights_analyzer.notify_rebalance(current_date, target_weights, self.asset_names)
        except Exception as e:
            self.log(f'Failed to notify weights analyzer: {e}')
            # Analyzers not yet initialized - this can happen during early strategy setup
    
    def notify_asset_values_analyzer(self):
        """Notify asset values analyzer of rebalancing event - captures asset state BEFORE rebalancing"""
        try:
            if hasattr(self, 'analyzers') and hasattr(self.analyzers, 'asset_values'):
                # Access the asset values analyzer directly by name
                asset_values_analyzer = self.analyzers.asset_values
                if asset_values_analyzer:
                    current_date = self.datas[0].datetime.date(0)
                    
                    # Collect current prices and positions for all assets
                    prices = []
                    positions = []
                    
                    for data in self.datas:
                        asset_name = data._name if hasattr(data, '_name') else 'Unknown'
                        # Get current price (closing price)
                        price = data.close[0]
                        prices.append(price)
                        
                        # Get current position (shares held)
                        position = self.getposition(data).size
                        positions.append(position)
                    
                    # Get portfolio value and cash
                    portfolio_value = self.broker.getvalue()
                    cash = self.broker.getcash()
                    
                    # Notify the analyzer
                    asset_values_analyzer.notify_rebalance(
                        date=current_date,
                        prices=prices,
                        positions=positions,
                        asset_names=self.asset_names,
                        portfolio_value=portfolio_value,
                        cash=cash
                    )
        except Exception as e:
            self.log(f'Failed to notify asset values analyzer: {e}')
            
    
    def prenext(self):
        """
        Called during the warmup period when not all data feeds have minimum period data.
        During this phase, no trading should occur to allow proper data accumulation.
        """
        # Log warmup progress
        current_len = len(self.datas[0])
        if current_len % 20 == 0:  # Log every 20 bars to avoid spam
            if self.verbose:
                self.log(f'Warmup period: {current_len}/{self.warmup_period} bars')
        
        # Do nothing during warmup - just accumulate data
        pass
    
    def next(self):
        # Ensure we have enough data before starting any operations
        current_len = len(self.datas[0])
        if current_len < self.warmup_period:
            if (current_len % 20 == 0) and self.verbose:  # Log every 20 bars to avoid spam
                self.log(f'Warmup period: {current_len}/{self.warmup_period} bars')
            return
            
        # Log when warmup completes (only on first trading day)
        elif current_len == self.warmup_period:
            self.log(f'Warmup period complete. Starting trading with {self.warmup_period} bars of data.')
            self.rebalance_counter = 0  # Reset counter after warmup
            return

        # Check if today is a rebalance day
        is_rebalance_day = (self.rebalance_counter == 0 or (self.rebalance_counter % self.rebalance_every == 0))
        
        if is_rebalance_day:
            # Cancel all stop orders BEFORE rebalancing (they'll be recreated after new positions are established)
            if self.params.use_stops:
                self._cancel_stop_orders()
                # Clear pending stops since we're rebalancing anyway - new positions will queue new stops
                self.pending_stop_placements = []
            
            # Capture asset values BEFORE rebalancing
            self.notify_asset_values_analyzer()
            
            target_weights,log_message = self.calculate_target_weights()
            if "ERROR" in log_message:
                self.log(f'Rebalance skipped due to error in weight calculation: {log_message}')
                self.rebalance_counter += 1
                return
            self.current_weights = target_weights.copy()
            self.rebalance_portfolio(target_weights)
            # Notify the weights analyzer
            self.notify_weights_analyzer(target_weights)
        else:
            # NOT a rebalance day - process any pending stop placements from previous bar's orders
            if self.params.use_stops and self.pending_stop_placements:
                for data in self.pending_stop_placements:
                    self._place_stop_order(data)
                self.pending_stop_placements = []
        
        self.rebalance_counter += 1

class BacktraderPortfolioBacktest:
    """
    Portfolio backtesting framework using Backtrader
    """
    
    def __init__(
            self,
            strategies: dict,
            datasets,
            benchmark=['1-N'], 
            rebalance_every=21*1,
            lookback=252//3,
            warmup=None,
            initial_cash=100000,
            commission=0.001,
            short_interest=0.0,
            interest_long=False,
            checksubmit=True
        ):
        """
        Initialize the backtesting framework
        
        Parameters:
        - strategies: dict of strategy classes or tuples (strategy_class, params_dict)
                     Examples:
                     {'IVP': InverseVolatilityStrategy}  # Uses default parameters
                     {'IVP': (InverseVolatilityStrategy, {'lookback': 126})}  # Custom parameters
                     {'DRPP': (DownsideRiskParityStrategy, {
                         'buy_slippage_buffer': 0.04,
                         'portfolio_func_kwargs': {'method': 'cvxpy', 'threshold': 0.0}
                     })}  # Custom parameters including portfolio function kwargs
        - datasets: list of datasets
        - benchmark: list of benchmark strategies
        - rebalance_every: rebalancing frequency (days) - default for strategies without custom params
        - lookback: lookback window for optimization (days) - default for strategies without custom params
        - warmup: warmup period before trading starts (defaults to lookback if None)
        - initial_cash: starting cash
        - commission: commission rate
        - short_interest: yearly interest rate charged for short positions (e.g., 0.03 = 3% annual)
                         Formula: days * price * abs(size) * (interest / 365)
                         Default is 0.0 (no short interest charged)
        - interest_long: if True, interest is charged on both long and short positions (e.g., for ETFs)
                        Default is False (only charge short positions)
        - checksubmit: if True, broker checks cash/margin before accepting orders (default: True)
                       if False, broker may auto-adjust order sizes (less predictable)
        """
        self.strategies = strategies
        self.datasets = datasets
        self.benchmark = benchmark
        self.rebalance_every = rebalance_every
        self.lookback = lookback
        self.warmup = warmup if warmup is not None else lookback
        self.initial_cash = initial_cash
        self.commission = commission
        self.short_interest = short_interest
        self.interest_long = interest_long
        self.checksubmit = checksubmit

        self.results = {}

    def log(self, txt, dt=None):
        """Logging function"""
        #print(f'{dt.isoformat()}, {txt}')
        with open('test_log.txt', 'a') as f:
            f.write(f'{txt}\n')
    

    def create_data_feeds(self, dataset_to_feed):
        """Create backtrader data feeds from dataset"""
        data_feeds = []
        #adjusted_prices = dataset['adjusted']
        assets_list = list(dataset_to_feed.columns.get_level_values(0).unique())

        for asset_name in assets_list:
            # Create individual asset dataframe
            asset_data = dataset_to_feed[asset_name]
            # Create backtrader data feed
            data_feed = bt.feeds.PandasData( 
                dataname=asset_data,
                name=asset_name,
                open='open',
                high='high', 
                low='low',
                close='close',
                volume='adjusted',
                openinterest=None
            )
            data_feeds.append(data_feed)
        
        return data_feeds, assets_list
    
    def run_single_backtest(self, strategy_class, dataset, dataset_name, strategy_params=None,strategy_name='log'):
        """Run backtest for a single strategy on a dataset
        
        Parameters:
        - strategy_class: The strategy class to instantiate
        - dataset: The dataset to backtest on
        - dataset_name: Name of the dataset for results tracking
        - strategy_params: Optional dict of additional parameters to pass to the strategy
        """
        
        # Create cerebro instance
        cerebro = bt.Cerebro()
        
        # Set cash and commission with short interest support
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(
            commission=self.commission,
            stocklike=True,
            interest=self.short_interest,
            interest_long=self.interest_long
        )
        
        # Set checksubmit parameter for order validation
        cerebro.broker.set_checksubmit(self.checksubmit)
        
        # Create data feeds
        data_feeds, asset_names = self.create_data_feeds(dataset)
        
        # Add data feeds to cerebro
        for data_feed in data_feeds:
            cerebro.adddata(data_feed)
        
        # Build strategy parameters - start with defaults, then override with custom params
        strat_params = {
            'rebalance_every': self.rebalance_every,
            'lookback': self.lookback,
            'warmup': self.warmup,
            'dataset_assets': asset_names,
        }
        
        # Override with strategy-specific parameters if provided
        if strategy_params:
            strat_params.update(strategy_params)
        
        # Add strategy with parameters
        cerebro.addstrategy(strategy_class, **strat_params)
        
        # Add analyzers
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.Calmar, _name='calmar')
        cerebro.addanalyzer(btanalyzers.VWR, _name='vwr')  # Variability-Weighted Return
        cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')  # System Quality Number
        cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='annual_return')  # Year-by-year returns
        cerebro.addanalyzer(btanalyzers.PeriodStats, _name='period_stats')  # Distribution statistics
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(btanalyzers.TimeReturn, _name='timereturn')
        cerebro.addanalyzer(PortfolioWeightsAnalyzer, _name='weights')
        cerebro.addanalyzer(OrderTrackingAnalyzer, _name='order_tracker')
        cerebro.addanalyzer(AssetValuesAnalyzer, _name='asset_values')  # Track asset values at rebalancing
        
        try:
            # Run backtest
            results = cerebro.run()
            strat = results[0]
            
            # Extract analyzer results
            analyzers = {}
            analyzers['Sharpe ratio'] = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            analyzers['max drawdown'] = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0) / 100
            analyzers['annual return'] = strat.analyzers.returns.get_analysis().get('rnorm100', 0) / 100
            analyzers['annual volatility'] = 0  # Not directly available, would need custom analyzer
            
            # Get Calmar ratio (Annual Return / Max Drawdown)
            calmar_analysis = strat.analyzers.calmar.get_analysis()
            # Calmar returns a dict with timeframe as key, extract the last value
            if calmar_analysis:
                analyzers['Calmar ratio'] = list(calmar_analysis.values())[-1] if calmar_analysis else 0
            else:
                analyzers['Calmar ratio'] = 0
            
            # Get VWR (Variability-Weighted Return)
            vwr_analysis = strat.analyzers.vwr.get_analysis()
            analyzers['VWR'] = vwr_analysis.get('vwr', 0) if vwr_analysis else 0
            
            # Get SQN (System Quality Number)
            sqn_analysis = strat.analyzers.sqn.get_analysis()
            analyzers['SQN'] = sqn_analysis.get('sqn', 0) if sqn_analysis else 0
            analyzers['SQN trades'] = sqn_analysis.get('trades', 0) if sqn_analysis else 0
            
            # Get Annual Returns (year-by-year)
            annual_returns_analysis = strat.analyzers.annual_return.get_analysis()
            # Store as a dictionary for later analysis
            analyzers['annual_returns_by_year'] = dict(annual_returns_analysis) if annual_returns_analysis else {}
            
            # Get Period Stats
            period_stats_analysis = strat.analyzers.period_stats.get_analysis()
            if period_stats_analysis:
                analyzers['period_avg'] = period_stats_analysis.get('average', 0)
                analyzers['period_stddev'] = period_stats_analysis.get('stddev', 0)
                analyzers['period_positive'] = period_stats_analysis.get('positive', 0)
                analyzers['period_negative'] = period_stats_analysis.get('negative', 0)
                analyzers['period_best'] = period_stats_analysis.get('best', 0)
                analyzers['period_worst'] = period_stats_analysis.get('worst', 0)
            
            # Get time returns for further analysis
            time_returns = strat.analyzers.timereturn.get_analysis()
            returns_series = pd.Series(time_returns)
            
            # Get weights analysis
            weights_analysis = strat.analyzers.weights.get_analysis()
            
            # Get order tracking analysis
            order_tracking_analysis = strat.analyzers.order_tracker.get_analysis()
            
            # Get asset values analysis
            asset_values_analysis = strat.analyzers.asset_values.get_analysis()
            
            if len(returns_series) > 0:
                analyzers['annual volatility'] = returns_series.std() * np.sqrt(252)
                analyzers['VaR (5%)'] = returns_series.quantile(0.05)
                analyzers['skewness'] = stats.skew(returns_series.dropna()) if len(returns_series) > 1 else 0
                analyzers['kurtosis'] = stats.kurtosis(returns_series.dropna()) if len(returns_series) > 1 else 0
                
                # Calculate Sortino Ratio (similar to Sharpe but using downside deviation)
                # Sortino = (Annual Return - Risk Free Rate) / Downside Deviation
                # We'll use 0 as the minimum acceptable return (MAR)
                risk_free_rate = 0.0  # Can be adjusted if needed
                negative_returns = returns_series[returns_series < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std() * np.sqrt(252)
                    if downside_deviation != 0:
                        analyzers['Sortino ratio'] = (analyzers['annual return'] - risk_free_rate) / downside_deviation
                    else:
                        analyzers['Sortino ratio'] = 0
                else:
                    # No negative returns means infinite Sortino (cap at a reasonable value)
                    analyzers['Sortino ratio'] = analyzers['Sharpe ratio'] * 2 if analyzers['Sharpe ratio'] else 0
            else:
                analyzers['Sortino ratio'] = 0
            
            return {
                'final_value': cerebro.broker.getvalue(),
                'performance': analyzers,
                'returns': returns_series.values if len(returns_series) > 0 else np.array([]),
                'return_dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) 
                               for date in returns_series.index] if len(returns_series) > 0 else [],
                'weights_history': weights_analysis,
                'order_history': order_tracking_analysis,
                'asset_values_history': asset_values_analysis,  # Add asset values tracking
                'cerebro': cerebro,
                'strategy': strat
            }
        
        except Exception as e:
            self.log(f"Error running backtest for {strategy_class.__name__} on {dataset_name}: {e}")
            return {
                'final_value': self.initial_cash,
                'performance': {
                    'Sharpe ratio': 0,
                    'Sortino ratio': 0,
                    'Calmar ratio': 0,
                    'VWR': 0,
                    'SQN': 0,
                    'SQN trades': 0,
                    'max drawdown': 0,
                    'annual return': 0,
                    'annual volatility': 0,
                    'VaR (5%)': 0,
                    'skewness': 0,
                    'kurtosis': 0,
                    'annual_returns_by_year': {},
                    'period_avg': 0,
                    'period_stddev': 0,
                    'period_positive': 0,
                    'period_negative': 0,
                    'period_best': 0,
                    'period_worst': 0
                },
                'returns': np.array([]),
                'weights_history': {'rebalancing_dates': [], 'portfolio_weights': [], 'asset_names': []},
                'order_history': {'orders': [], 'summary': {}},
                'cerebro': None,
                'strategy': None
            }
    
    def run_backtest(self):
        """Run the complete backtesting procedure"""
        self.log("Starting backtrader portfolio backtesting...")
        
        # Prepare all strategies - normalize format to (class, params_dict)
        all_strategies = {}
        for name, value in self.strategies.items():
            if isinstance(value, tuple):
                # Format: (StrategyClass, {'param': value})
                all_strategies[name] = value
            else:
                # Format: StrategyClass - convert to tuple with empty params
                all_strategies[name] = (value, {})
        
        # Add benchmark if specified
#        if '1-N' in self.benchmark:
#            all_strategies['1-N'] = (EqualWeightStrategy, {})

        # Run backtests
        for strategy_name, (strategy_class, strategy_params) in all_strategies.items():
            self.results[strategy_name] = {}
            for dataset_i, dataset_tmp in enumerate(self.datasets, start=1):
                self.log(f"Backtesting {strategy_name} on {dataset_i}...")
                result = self.run_single_backtest(
                    strategy_class, 
                    dataset_tmp, 
                    f"dataset_{dataset_i}",
                    strategy_params=strategy_params,
                    strategy_name=strategy_name
                )
                self.results[strategy_name][f"dataset_{dataset_i}"] = result

        self.log(f"Backtesting completed for {len(all_strategies)} strategies on {len(self.datasets)} datasets.")
        return self.results
    
    def get_summary_statistics(self, metric='Sharpe ratio'):
        """Get summary statistics for a specific performance metric"""
        summary = {}
        
        for strategy_name in self.results.keys():
            metric_values = []
            for dataset_name in self.results[strategy_name].keys():
                metric_values.append(self.results[strategy_name][dataset_name]['performance'][metric])
            
            summary[strategy_name] = {
                'mean': np.mean(metric_values),
                'median': np.median(metric_values),
                'std': np.std(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values)
            }
        
        return summary


