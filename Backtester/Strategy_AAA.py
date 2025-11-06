import numpy as np
import pandas as pd
from scipy.optimize import minimize

from Backtester.BacktestFramework import PortfolioRebalanceStrategy


def adaptive_asset_allocation_portfolio_fun(dataset: pd.DataFrame, base_column: str = "adjusted", 
                                            momentum_top_n: int = 5, momentum_lookback: int = 6 * 22, 
                                            covariance_lookback: int = 22,
                                            momentum_method: str = 'simple',  # 'simple' or 'bollinger'
                                            bb_period: int = 20,  # Bollinger Bands period (only for bollinger method)
                                            bb_std: float = 2.0,  # Bollinger Bands standard deviation multiplier
                                            **kwargs):
    """
    Adaptive Asset Allocation (AAA) portfolio function.

    Behavior adapted from the notebook `adaptiveassetallocation.ipynb`.

    Steps:
    - Extract price series (handles MultiIndex with a base column)
    - Compute momentum using selected method:
      * 'simple': price / price.shift(momentum_lookback) (price relative)
      * 'bollinger': price change relative to Bollinger Band volatility
    - Select top `momentum_top_n` assets by momentum
    - For the selected assets, compute recent log-returns (last `covariance_lookback` periods)
    - Solve a long-only minimum variance optimization over the selected assets
      with bounds [0,1] and sum(weights)==1. If optimizer fails, fall back to equal
      weights across the selected assets.

    Parameters:
    - dataset: DataFrame with price data
    - base_column: column name to extract from multi-level columns
    - momentum_top_n: number of top momentum assets to select (default: 5)
    - momentum_lookback: momentum lookback period in days (default: 132 = 6 months)
    - covariance_lookback: lookback period for covariance estimation in days (default: 22 = 1 month)
    - momentum_method: 'simple' (default) or 'bollinger' for momentum calculation method
    - bb_period: Bollinger Bands lookback period in days (default: 20, only used with bollinger method)
    - bb_std: Bollinger Bands standard deviation multiplier (default: 2.0, only used with bollinger method)

    Returns:
    - weights: numpy array of portfolio weights
    - log_message: string with warnings/errors encountered
    
    Notes:
    - Simple method computes momentum using price relatives
    - Bollinger method computes momentum as price change scaled by BB volatility
    - Selects top momentum assets for further analysis
    - Uses minimum variance optimization on selected momentum winners
    - Falls back to equal weights if optimization fails or insufficient data
    - All weights are long-only (no shorting)
    """
    log_message = ""#f"momentum_top_n={momentum_top_n}, momentum_lookback={momentum_lookback}, covariance_lookback={covariance_lookback} | "

    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        # Try to extract specified base column (e.g., 'adjusted' / 'Adj Close')
        try:
            extracted_data = dataset.xs(base_column, level=1, axis=1)
            if isinstance(extracted_data, pd.Series):
                dataset = extracted_data.to_frame()
            else:
                dataset = extracted_data
        except Exception:
            # Fall back to keeping the dataset as-is
            log_message += f"WARNING: AAA - couldn't extract level '{base_column}' from MultiIndex; using full dataset.\n"

    # Number of assets
    asset_names = list(dataset.columns)
    num_assets = len(asset_names)

    # If no assets available, return empty array
    if num_assets == 0:
        return np.array([]), "ERROR: AAA - No assets in dataset."

    # Determine minimum data requirement based on momentum method
    if momentum_method == 'bollinger':
        min_history = max(momentum_lookback, bb_period)
    else:
        min_history = momentum_lookback
    
    # Require enough history for momentum calculation
    if len(dataset) <= min_history:
        log_message += f"WARNING: AAA - Not enough history for momentum calculation (need {min_history}, have {len(dataset)}); returning equal weights.\n"
        return np.ones(num_assets) / num_assets, log_message
    #print(num_assets)
    # Compute momentum based on selected method
    try:
        current_prices = dataset.iloc[-1]
        
        if momentum_method == 'simple':
            # Simple momentum: price relatives (current_price / price_momentum_lookback_ago)
            #print(f"Current prices: {current_prices}\n")
            past_prices = dataset.shift(momentum_lookback).iloc[-1]
            #print(f"Past prices: {past_prices}\n")
            momentum_scores = (current_prices / past_prices).replace([np.inf, -np.inf], np.nan).dropna()
        
        elif momentum_method == 'bollinger':
            # Bollinger Bands momentum: price change scaled by current BB volatility
            # Calculate current Bollinger Bands using last bb_period days
            current_data = dataset.iloc[-bb_period:]
            
            # Get historical prices from momentum_lookback ago
            if len(dataset) <= momentum_lookback:
                log_message += f"WARNING: AAA - Not enough data for {momentum_lookback}-day lookback; returning equal weights.\n"
                return np.ones(num_assets) / num_assets, log_message
            
            past_prices = dataset.iloc[-momentum_lookback-1]
            
            # Calculate BB momentum for each asset
            bb_momentum = {}
            for asset in asset_names:
                asset_current_data = current_data[asset]
                current_price = float(current_prices.loc[asset])
                past_price = float(past_prices.loc[asset])
                
                # Calculate standard deviation
                std_dev = float(asset_current_data.std())
                
                # Calculate band width
                band_width = 2 * bb_std * std_dev
                
                if band_width > 0:
                    # Momentum score: price change scaled by band width
                    price_change = current_price - past_price
                    # Score as multiple of half band width (similar to price relative concept)
                    bb_momentum[asset] = 1.0 + (price_change / (band_width / 2))
                else:
                    # No volatility - use simple price relative
                    if past_price > 0:
                        bb_momentum[asset] = current_price / past_price
                    else:
                        bb_momentum[asset] = 1.0
            
            momentum_scores = pd.Series(bb_momentum).replace([np.inf, -np.inf], np.nan).dropna()
        
        else:
            log_message += f"ERROR: AAA - Invalid momentum_method '{momentum_method}'. Use 'simple' or 'bollinger'.\n"
            return np.ones(num_assets) / num_assets, log_message
            
    except Exception as e:
        log_message += f"ERROR: AAA - momentum calculation failed: {e}; returning equal weights.\n"
        return np.ones(num_assets) / num_assets, log_message

    if momentum_scores.empty:
        log_message += "WARNING: AAA - Momentum series empty after dropna; returning equal weights.\n"
        return np.ones(num_assets) / num_assets, log_message

    # Select top momentum assets
    num_selected = min(momentum_top_n, len(momentum_scores))
    top_momentum_assets = momentum_scores.nlargest(num_selected).index.tolist()
    #print(f"Selected top momentum assets: {top_momentum_assets}\n")
    # Compute log returns and take the most recent covariance_lookback observations
    log_returns: pd.DataFrame = np.log(dataset / dataset.shift(1)).dropna()  # type: ignore
    if len(log_returns) < 2 or len(log_returns) < covariance_lookback:
        # Not enough returns history for a stable covariance; fall back
        log_message += "WARNING: AAA - Not enough return history for covariance estimation; using equal weights among selected assets.\n"
        weights = np.zeros(num_assets)
        if len(top_momentum_assets) > 0:
            for asset in top_momentum_assets:
                weights[asset_names.index(asset)] = 1.0 / len(top_momentum_assets)
        else:
            weights = np.ones(num_assets) / num_assets
        return weights, log_message

    recent_returns = log_returns.tail(covariance_lookback)[top_momentum_assets].dropna()

    # If some selected assets do not have complete history, reduce the set
    valid_assets = [asset for asset in top_momentum_assets 
                    if asset in recent_returns.columns and recent_returns[asset].notna().sum() >= 1]
    if len(valid_assets) == 0:
        log_message += "WARNING: AAA - No selected assets have sufficient recent history; returning equal weights.\n"
        return np.ones(num_assets) / num_assets, log_message

    # If fewer assets than originally selected, proceed with the available ones
    if set(valid_assets) != set(top_momentum_assets):
        log_message += f"WARNING: AAA - reduced top assets to those with data: {valid_assets}.\n"

    # Build covariance matrix
    cov_matrix = recent_returns[valid_assets].cov().values

    # Optimization: minimize weights' Σ weights subject to sum(weights)==1, 0<=weights<=1
    def portfolio_variance(weights_vec):
        return float(weights_vec.T @ cov_matrix @ weights_vec)

    num_valid_assets = len(valid_assets)
    initial_weights = np.ones(num_valid_assets) / num_valid_assets
    bounds = tuple((0.0, 1.0) for _ in range(num_valid_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights_vec: np.sum(weights_vec) - 1.0},)

    try:
        result = minimize(portfolio_variance, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        if not result.success:
            # Fallback to equal weights among selected assets
            log_message += f"WARNING: AAA - optimizer failed ({result.message}); using equal weights among selected assets.\n"
            optimal_weights = np.ones(num_valid_assets) / num_valid_assets
        else:
            optimal_weights = np.maximum(result.x, 0)
            weights_sum = optimal_weights.sum()
            if weights_sum <= 0:
                optimal_weights = np.ones(num_valid_assets) / num_valid_assets
            else:
                optimal_weights = optimal_weights / weights_sum
    except Exception as e:
        log_message += f"ERROR: AAA - optimization exception: {e}; using equal weights among selected assets.\n"
        optimal_weights = np.ones(num_valid_assets) / num_valid_assets

    # Build full weight vector aligned with dataset columns
    weights = np.zeros(num_assets)
    for asset in valid_assets:
        asset_idx = asset_names.index(asset)
        weights[asset_idx] = float(optimal_weights[valid_assets.index(asset)])

    return weights, log_message


class AdaptiveAssetAllocationStrategy(PortfolioRebalanceStrategy):
    """Adaptive Asset Allocation (AAA) strategy using momentum selection + minimum variance weighting.

    This strategy selects the top N assets by momentum and computes minimum-variance 
    weights using recent return observations.
    
    The strategy follows these steps:
    1. Calculate momentum using selected method (simple or Bollinger Bands)
    2. Select top momentum_top_n assets
    3. Estimate covariance matrix using covariance_lookback period
    4. Solve minimum variance optimization for selected assets
    
    Momentum Calculation Methods:
    - 'simple' (default): Price relative (current_price / past_price)
        * Values > 1.0 indicate positive momentum
        * Standard momentum measure used in most tactical allocation strategies
    
    - 'bollinger': Bollinger Bands-based momentum
        * Calculates price change scaled by current volatility (BB width)
        * Score = 1.0 + (price_change / half_band_width)
        * Values > 1.0 indicate upward momentum
        * Accounts for volatility - larger moves in volatile assets aren't over-weighted
    
    Parameters are controlled by the backtester configuration and passed via kwargs
    to the portfolio function:
    - momentum_top_n: number of top assets to select (default: 5)
    - momentum_lookback: lookback period in days (default: 132 = 6 months)
    - covariance_lookback: period for covariance estimation (default: 22 = 1 month)
    - momentum_method: 'simple' or 'bollinger' (default: 'simple')
    - bb_period: Bollinger Bands period in days (default: 20, only for bollinger method)
    - bb_std: BB standard deviation multiplier (default: 2.0, only for bollinger method)
    """
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = adaptive_asset_allocation_portfolio_fun
