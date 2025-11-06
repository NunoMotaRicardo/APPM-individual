import numpy as np
import cvxpy as cp
import pandas as pd

from Backtester.BacktestFramework import PortfolioRebalanceStrategy

def downside_risk_parity_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", method:str="cvxpy", 
                                       threshold:float=0.0, max_iter:int=2000, tol:float=1e-6, **kwargs):
    """
    Downside Risk Parity Portfolio (DRPP): Equal downside risk contribution from all assets
    
    Similar to Vanilla Risk Parity but uses downside semi-covariance matrix instead of 
    full covariance matrix. This focuses risk parity on downside (negative) returns only,
    which is more aligned with investor preferences (similar to Sortino vs Sharpe).
    
    Solves for weights where each asset contributes equally to portfolio downside risk:
        DRC_i = w_i * (Σ_downside * w)_i = 1/n * σ_downside_p    for all i
    
    Parameters:
    - dataset: DataFrame with price data
    - base_column: column name to extract from multi-level columns
    - method: 'cvxpy' (default, most robust) or 'iterative' (faster but less stable)
    - threshold: return threshold for defining "downside" (default: 0.0 for negative returns)
    - max_iter: maximum iterations for convergence (iterative method only)
    - tol: convergence tolerance
    
    Returns:
    - numpy array of portfolio weights
    
    Notes:
    - CVXPY method uses Sequential Convex Programming (SCP) for convergence
    - Iterative method uses cyclical coordinate descent with dampening
    - Only returns below threshold are considered in the semi-covariance calculation
    - More conservative than traditional Risk Parity - focuses on downside protection
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.xs(base_column, level=1, axis=1)
    
    # Compute log returns
    log_returns = np.log(dataset / dataset.shift(1)).dropna()
    
    # Check if we have enough data
    if len(log_returns) < 2:
        log_message = "ERROR: DRPP - Not enough data, returning equal weights."
        n = len(dataset.columns)
        return np.ones(n) / n, log_message
    
    # Compute downside semi-covariance matrix
    # Only use returns below threshold (default 0 = negative returns)
    downside_returns = log_returns.copy()
    downside_returns[downside_returns > threshold] = 0  # Zero out upside returns
    
    # Check if we have any downside observations
    if (downside_returns != 0).sum().sum() < 2:
        log_message = "WARNING: DRPP - Insufficient downside observations, using full covariance."
        # Fallback to regular risk parity if no downside data
        Sigma_downside = log_returns.cov().values
    else:
        # Compute semi-covariance matrix (only from downside returns)
        Sigma_downside = downside_returns.cov().values
    
    n = Sigma_downside.shape[0]
    
    # Add small regularization for numerical stability
    Sigma_downside = Sigma_downside + 1e-8 * np.eye(n)
    
    # Check if matrix is positive semi-definite
    eigenvalues = np.linalg.eigvals(Sigma_downside)
    if np.any(eigenvalues < -1e-6):
        log_message += "WARNING: DRPP - Semi-covariance matrix not PSD, adding regularization."
        # Add more regularization to ensure positive semi-definite
        min_eig = np.min(eigenvalues)
        Sigma_downside = Sigma_downside + (abs(min_eig) + 1e-6) * np.eye(n)
    
    if method == "cvxpy":
        result, method_log = _downside_risk_parity_cvxpy(Sigma_downside, n)
        return result, log_message + method_log
    else:
        result, method_log = _downside_risk_parity_iterative(Sigma_downside, n, max_iter, tol)
        return result, log_message + method_log

def _downside_risk_parity_cvxpy(Sigma_downside, n):
    """
    Solve downside risk parity using Sequential Convex Programming (SCP).
    
    Uses the same SCP approach as regular risk parity but with downside semi-covariance.
    """
    log_message = ""
    try:
        # Initial guess: inverse downside volatility weights
        downside_vol = np.sqrt(np.diag(Sigma_downside))
        
        # Handle zero volatility cases
        if np.any(downside_vol < 1e-8):
            log_message = "WARNING: DRPP - Near-zero downside volatility detected, using equal weights."
            return np.ones(n) / n, log_message
        
        w_current = (1.0 / downside_vol)
        w_current = w_current / np.sum(w_current)
        
        max_scp_iter = 50
        tol = 1e-6
        
        for scp_iter in range(max_scp_iter):
            # Define optimization variable
            w = cp.Variable(n, nonneg=True)
            
            # Linearize downside risk contributions around current point
            # DRC_i = w_i * (Σ_downside * w)_i
            marginal_downside_risk = Sigma_downside @ w_current
            
            # Approximate downside risk contributions (first-order Taylor expansion)
            drc_approx = cp.multiply(w, marginal_downside_risk)
            
            # Target: equal downside risk contributions
            target_drc = cp.sum(drc_approx) / n
            
            # Minimize squared deviations from equal downside risk contribution
            objective = cp.Minimize(cp.sum_squares(drc_approx - target_drc))
            
            # Constraints
            constraints = [cp.sum(w) == 1]
            
            # Solve convex subproblem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=2000)
            
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Try SCS solver
                prob.solve(solver=cp.SCS, verbose=False, max_iters=2000, eps=1e-4)
            
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or w.value is None:
                # SCP failed, fallback to iterative method
                log_message = f"WARNING: DRPP SCP failed at iteration {scp_iter}, using iterative method"
                return _downside_risk_parity_iterative(Sigma_downside, n, max_iter=2000, tol=1e-6)
            
            # Update weights
            w_new = np.asarray(w.value).reshape(-1)
            w_new = np.maximum(w_new, 0)  # Clean numerical noise
            w_new = w_new / np.sum(w_new)  # Normalize
            
            # Check convergence
            weight_change = np.max(np.abs(w_new - w_current))
            if weight_change < tol:
                return w_new, log_message
            
            w_current = w_new
        
        # SCP didn't converge, but return best solution found
        log_message = f"WARNING: DRPP SCP did not fully converge, returning best solution (change: {weight_change:.6f})"
        return w_current, log_message
    
    except Exception as e:
        log_message = f"ERROR: DRPP SCP optimization error: {e}, using iterative method\n"
        return _downside_risk_parity_iterative(Sigma_downside, n, max_iter=2000, tol=1e-6)

def _downside_risk_parity_iterative(Sigma_downside, n, max_iter, tol):
    """
    Solve downside risk parity using improved iterative Newton-Raphson method.
    
    Uses cyclical coordinate descent with backtracking for better convergence.
    """
    log_message = ""
    # Equal downside risk budgets
    b = np.ones(n) / n
    
    # Initial weights (inverse downside volatility as better starting point)
    downside_vol = np.sqrt(np.diag(Sigma_downside))
    
    # Handle zero volatility cases
    if np.any(downside_vol < 1e-8):
        log_message = "ERROR: DRPP - Near-zero downside volatility, returning equal weights."
        return np.ones(n) / n, log_message
    
    w = (1.0 / downside_vol)
    w = w / np.sum(w)
    
    # Iterative algorithm with improved update rule
    for iteration in range(max_iter):
        # Calculate portfolio downside variance
        portfolio_var = w.T @ Sigma_downside @ w
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Prevent division by zero
        if portfolio_vol < 1e-10:
            log_message = "ERROR: DRPP - Portfolio downside volatility too low, returning equal weights."
            return np.ones(n) / n, log_message
        
        # Calculate marginal downside risk contributions and downside risk contributions
        marginal_contrib = Sigma_downside @ w  # MDRC_i = (Σ_downside * w)_i
        risk_contrib = w * marginal_contrib  # DRC_i = w_i * MDRC_i
        
        # Target downside risk contribution for each asset
        target_rc = portfolio_var * b  # Target: 1/n of total portfolio downside variance
        
        # Newton-Raphson update with dampening
        # Update formula: w_new = w * sqrt(target_rc / current_rc)
        update_ratio = np.sqrt(target_rc / np.maximum(risk_contrib, 1e-10))
        
        # Apply dampening factor for stability (0.5 = slower but more stable)
        dampening = 0.5
        w_new = w * (dampening * update_ratio + (1 - dampening))
        
        # Normalize
        w_new = w_new / np.sum(w_new)
        
        # Check convergence (use relative change)
        relative_change = np.max(np.abs(w_new - w) / np.maximum(w, 1e-8))
        if relative_change < tol:
            return w_new, log_message
        
        w = w_new
    
    # If max iterations reached, check if solution is reasonable
    final_risk_contrib = w * (Sigma_downside @ w)
    rc_std = np.std(final_risk_contrib)
    rc_mean = np.mean(final_risk_contrib)
    
    if rc_std / rc_mean < 0.5:  # Downside risk contributions are reasonably equal
        log_message += f"WARNING: DRPP did not fully converge but solution is acceptable (CV: {rc_std/rc_mean:.3f})"
        return w, log_message
    else:
        log_message += f"ERROR: DRPP optimization did not converge to acceptable solution (CV: {rc_std/rc_mean:.3f})"
        return np.ones(n) / n, log_message
    
class DownsideRiskParityStrategy(PortfolioRebalanceStrategy):
    """Downside Risk Parity Portfolio (DRPP) strategy - Equal downside risk contribution"""
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = downside_risk_parity_portfolio_fun