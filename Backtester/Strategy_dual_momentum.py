import numpy as np
import pandas as pd
from Backtester.BacktestFramework import PortfolioRebalanceStrategy


def gem_portfolio_fun(dataset: pd.DataFrame, base_column: str = "adjusted",
                      momentum_periods: list = [21, 63, 126, 252],  # lookback periods in days
                      min_positive_periods: int = 3,  # minimum periods with positive momentum
                      treasury_threshold: float = 0.0,  # minimum return to invest in risky assets
                      maximum_positions: int = 6,
                      risk_free_asset: str = 'cash',  # 'cash' or symbol name (e.g., 'BIL', 'SHY')
                      momentum_method: str = 'simple',  # 'simple' or 'bollinger'
                      bb_period: int = 20,  # Bollinger Bands period (only for bollinger method)
                      bb_std: float = 2.0,  # Bollinger Bands standard deviation multiplier
                      **kwargs):
    """
    Global Equities Momentum (GEM) portfolio function.
    
    Strategy Logic (based on gem.py):
    1. Calculate momentum over multiple periods (default: 1, 3, 6, 12 months = 21, 63, 126, 252 days)
    2. For each risky asset, count how many periods have positive momentum
    3. If an asset has positive momentum in >= min_positive_periods, it's a candidate
    4. Compare top candidates against risk-free asset returns
    5. If no risky assets beat treasury threshold, allocate to risk-free asset (cash or treasury)
    6. Otherwise, equal-weight the top performing risky assets
    
    Momentum Calculation Methods:
    - 'simple': Traditional return-based momentum (price change over period)
    - 'bollinger': Bollinger Bands-based momentum (position relative to bands)
    
    Parameters:
    - dataset: DataFrame with DAILY price data (indexed by date)
    - base_column: column name to extract from multi-level columns
    - momentum_periods: list of lookback periods in DAYS (default: [21, 63, 126, 252] = 1, 3, 6, 12 months)
    - min_positive_periods: minimum number of periods with positive momentum to qualify
    - treasury_threshold: minimum return threshold to prefer risky assets over treasury
    - maximum_positions: maximum number of positions to hold (default: None = no limit)
    - risk_free_asset: 'cash' (default) to hold cash, or symbol name to buy a specific treasury/bond asset
    - momentum_method: 'simple' (default) or 'bollinger' for momentum calculation method
    - bb_period: Bollinger Bands lookback period in days (default: 20)
    - bb_std: Bollinger Bands standard deviation multiplier (default: 2.0)
    
    Returns:
    - weights: numpy array of portfolio weights
    - log_message: string with warnings/errors encountered
    """
    log_message = ""
    
    # Handle MultiIndex columns (backtrader may pass multi-level columns)
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        try:
            extracted_data = dataset.xs(base_column, level=1, axis=1)
            if isinstance(extracted_data, pd.Series):
                dataset = extracted_data.to_frame()
            else:
                dataset = extracted_data
        except Exception as e:
            log_message += f"WARNING: GEM - couldn't extract level '{base_column}' from MultiIndex: {e}"
    
    asset_names = list(dataset.columns)
    num_assets = len(asset_names)
    
    if num_assets == 0:
        return np.array([]), "ERROR: GEM - No assets in dataset."
    
    # Identify risk-free asset based on parameter
    use_cash = (risk_free_asset.lower() == 'cash')
    treasury_idx = None
    
    if use_cash:
        # Using cash as risk-free asset - no treasury asset in portfolio
        #log_message += "INFO: GEM - Using cash as risk-free asset."
        pass
    else:
        # Look for the specified risk-free asset symbol in the dataset
        matching_assets = [name for name in asset_names if risk_free_asset.upper() in name.upper()]
        if not matching_assets:
            log_message += f"WARNING: GEM - Risk-free asset '{risk_free_asset}' not found in dataset; using cash instead."
            use_cash = True
        else:
            treasury_idx = asset_names.index(matching_assets[0])
            #log_message += f"INFO: GEM - Using '{matching_assets[0]}' as risk-free asset."
    
    # Calculate maximum lookback needed (in trading days)
    max_lookback_days = max(momentum_periods)
    
    # For Bollinger Bands method, we also need bb_period for calculating the bands
    if momentum_method == 'bollinger':
        max_lookback_days = max(max_lookback_days, bb_period)
    
    if len(dataset) <= max_lookback_days:
        log_message += f"WARNING: GEM - Insufficient data ({len(dataset)} days) for {max_lookback_days} day lookback."
        # Default to risk-free asset (cash or treasury)
        weights = np.zeros(num_assets)
        if not use_cash and treasury_idx is not None:
            weights[treasury_idx] = 1.0
        # If using cash, weights remain all zeros (holding cash)
        return weights, log_message
    
    # Calculate momentum for each asset over specified periods
    current_prices = dataset.iloc[-1]
    momentum_scores = pd.DataFrame(index=asset_names, columns=momentum_periods)
    
    if momentum_method == 'simple':
        # Simple momentum: percentage return over lookback period
        for lookback_days in momentum_periods:
            if len(dataset) > lookback_days:
                past_prices = dataset.iloc[-lookback_days-1]
                returns = ((current_prices - past_prices) / past_prices * 100).replace([np.inf, -np.inf], np.nan)
                momentum_scores[lookback_days] = returns
            else:
                log_message += f"WARNING: GEM - Skipping {lookback_days}-day momentum (need {lookback_days} days, have {len(dataset)})."
    
    elif momentum_method == 'bollinger':
        # Bollinger Bands momentum: compare historical prices to current Bollinger Bands
        # Calculate current Bollinger Bands using the last bb_period days
        # Then check where prices were at each lookback period relative to current bands
        # Positive momentum = historical price below current middle band (price has risen)
        # Negative momentum = historical price above current middle band (price has fallen)
        
        # First, calculate current Bollinger Bands for all assets
        bb_data = {}
        current_data = dataset.iloc[-bb_period:]
        
        for asset in asset_names:
            asset_prices = current_data[asset]
            
            # Calculate middle band (SMA of last bb_period days)
            middle_band = asset_prices.mean()
            
            # Calculate standard deviation
            std_dev = asset_prices.std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (bb_std * std_dev)
            lower_band = middle_band - (bb_std * std_dev)
            
            bb_data[asset] = {
                'middle': middle_band,
                'upper': upper_band,
                'lower': lower_band,
                'std': std_dev
            }
        
        # Now calculate momentum scores by comparing historical prices to current BB
        for lookback_days in momentum_periods:
            if len(dataset) > lookback_days:
                historical_prices = dataset.iloc[-lookback_days-1]
                bb_scores = []
                
                for asset in asset_names:
                    historical_price = historical_prices[asset]
                    bb = bb_data[asset]
                    
                    # Calculate momentum score as price change relative to band width
                    # Positive if current price > historical price (upward momentum)
                    band_width = bb['upper'] - bb['lower']
                    if band_width > 0:
                        # Score: percentage change from historical to current middle band
                        # scaled by band width
                        price_change = current_prices[asset] - historical_price
                        bb_score = (price_change / (band_width / 2)) * 100
                    else:
                        bb_score = 0
                    
                    bb_scores.append(bb_score)
                
                momentum_scores[lookback_days] = bb_scores
            else:
                log_message += f"WARNING: GEM - Skipping {lookback_days}-day Bollinger momentum (need {lookback_days} days, have {len(dataset)})."
    
    else:
        log_message += f"ERROR: GEM - Invalid momentum_method '{momentum_method}'. Use 'simple' or 'bollinger'."
        weights = np.zeros(num_assets)
        return weights, log_message
    
    # Remove risk-free asset from risky asset evaluation
    risky_assets = [name for name in asset_names if treasury_idx is None or asset_names.index(name) != treasury_idx]
    
    if len(risky_assets) == 0:
        log_message += "WARNING: GEM - No risky assets to evaluate."
        weights = np.zeros(num_assets)
        if not use_cash and treasury_idx is not None:
            weights[treasury_idx] = 1.0
        # If using cash, weights remain all zeros
        return weights, log_message
    
    # Count positive momentum periods for each risky asset
    positive_counts = {}
    avg_momentum = {}
    
    for asset in risky_assets:
        asset_momentum = momentum_scores.loc[asset].dropna()
        if len(asset_momentum) > 0:
            positive_counts[asset] = (asset_momentum > 0).sum()
            avg_momentum[asset] = asset_momentum.mean()
        else:
            positive_counts[asset] = 0
            avg_momentum[asset] = 0
    
    # Filter assets that meet minimum positive periods criterion
    qualified_assets = [asset for asset in risky_assets 
                       if positive_counts[asset] >= min_positive_periods]
    
    # If risk-free asset exists (not cash), check if qualified assets beat treasury threshold
    if not use_cash and treasury_idx is not None and len(qualified_assets) > 0:
        treasury_return = momentum_scores.loc[asset_names[treasury_idx]].mean()
        
        # Filter qualified assets that beat treasury + threshold
        qualified_assets = [asset for asset in qualified_assets 
                          if avg_momentum[asset] > treasury_return + treasury_threshold]
    
    #if maximum_positions is set, limit the number of qualified assets, selecting those with highest average momentum
    if maximum_positions is not None and len(qualified_assets) > maximum_positions:
        sorted_assets = sorted(qualified_assets, key=lambda x: avg_momentum[x], reverse=True)
        qualified_assets = sorted_assets[:maximum_positions]

    # Allocate weights
    weights = np.zeros(num_assets)
    
    if len(qualified_assets) == 0:
        # No qualified risky assets - allocate to risk-free asset (cash or treasury)
        if use_cash:
            # All weights remain zero = holding cash
            log_message += "INFO: GEM - No qualified risky assets; holding cash."
        elif treasury_idx is not None:
            weights[treasury_idx] = 1.0
            log_message += f"INFO: GEM - No qualified risky assets; allocating to {asset_names[treasury_idx]}."
        else:
            # Shouldn't reach here, but fallback to equal weight
            weights = np.ones(num_assets) / num_assets
            log_message += "INFO: GEM - No qualified assets; equal weighting all."
    else:
        # Equal weight qualified assets
        weight_per_asset = 1.0 / len(qualified_assets)
        for asset in qualified_assets:
            weights[asset_names.index(asset)] = weight_per_asset
        #log_message += f"INFO: GEM - Allocated to {len(qualified_assets)} qualified assets: {qualified_assets}"
    
    return weights, log_message


class GlobalEquitiesMomentumStrategy(PortfolioRebalanceStrategy):
    """
    Global Equities Momentum (GEM) Strategy
    
    Based on the momentum analysis from gem.py, this strategy:
    1. Evaluates momentum across multiple time periods (1, 3, 6, 12 months)
    2. Selects assets with consistently positive momentum
    3. Compares risky assets against risk-free asset returns
    4. Allocates to defensive position (cash or treasury) when risk is unfavorable
    
    This is a tactical asset allocation strategy that aims to capture momentum
    while protecting capital during market downturns.
    
    Momentum Calculation Methods:
    - 'simple': Traditional return-based momentum (percentage price change over period)
    - 'bollinger': Bollinger Bands-based momentum (position relative to bands)
        * Positive momentum when price is above the middle band (SMA)
        * Negative momentum when price is below the middle band
        * Score magnitude represents distance from middle band scaled by band width
    
    Risk-Free Asset Options:
    - 'cash' (default): Hold cash when defensive (zero weights for all assets)
    - Symbol name (e.g., 'BIL', 'SHY', 'IEF'): Buy specific treasury/bond asset when defensive
    
    Usage examples in portfolio_func_kwargs:
        # Simple momentum with cash
        'momentum_method': 'simple',
        'risk_free_asset': 'cash'
        
        # Bollinger Bands momentum with custom parameters
        'momentum_method': 'bollinger',
        'bb_period': 20,
        'bb_std': 2.0,
        'risk_free_asset': 'BIL'
    """
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = gem_portfolio_fun