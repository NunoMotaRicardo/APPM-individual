import numpy as np
import cvxpy as cp
import pandas as pd

from Backtester.BacktestFramework import PortfolioRebalanceStrategy

def quintile_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", **kwargs):
    """
    Quintile portfolio: invest equally in top 20% performing stocks
    
    Parameters:
    - dataset: dict with dataset from yahoo containing price data
    
    Returns:
    - numpy array of portfolio weights
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.xs(base_column, level=1, axis=1)
    #print(f"Dataset columns after extracting '{base_column}': {dataset.columns.tolist()}")

    # Compute log returns (exclude first row due to differencing)
    log_returns = np.log(dataset / dataset.shift(1)).dropna()

    N = len(dataset.columns)

    # Compute mean returns and rank them
    mean_returns = log_returns.mean()
    ranking = mean_returns.sort_values(ascending=False)

    w = np.zeros(N)
    top_quintile_size = round(N / 5)
    top_stocks = ranking.head(top_quintile_size).index

    for i, stock in enumerate(dataset.columns):
        if stock in top_stocks:
            w[i] = 1 / top_quintile_size

    return w , log_message

def gmvp_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", allow_short:bool=False, **kwargs):
    """
    Global Minimum Variance Portfolio (GMVP)
    
    Parameters:
    - dataset: dict with 'adjusted' key containing price data
    - allow_short: if True, allows short positions (negative weights); if False, long-only constraint
    
    Returns:
    - numpy array of portfolio weights
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.xs(base_column, level=1, axis=1)
    
    # Compute log returns
    log_returns = np.log(dataset / dataset.shift(1)).dropna()

    # Compute covariance matrix
    cov_matrix = log_returns.cov().values
    
    # Solve for GMVP weights: w = (Σ^-1 * 1) / (1' * Σ^-1 * 1)
    ones = np.ones(cov_matrix.shape[0])
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        w = inv_cov @ ones
        
        if allow_short:
            # Allow negative weights (short positions)
            w = w / np.sum(w)
        else:
            # Long-only constraint
            w = np.abs(w) / np.sum(np.abs(w))
    except np.linalg.LinAlgError:
        # If covariance matrix is singular, use equal weights
        #print("ERROR: Covariance matrix is singular, returning equal weights.")
        log_message="ERROR: Covariance matrix is singular, returning equal weights."
        w = np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]
    
    return w , log_message

def markowitz_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", lambda_param=0.5, allow_short:bool=False, **kwargs):
    """
    Markowitz mean-variance portfolio using CVXPY
    
    Parameters:
    - dataset: dict with 'adjusted' key containing price data
    - lambda_param: risk aversion parameter (higher = more risk averse)
    - allow_short: if True, allows short positions (negative weights); if False, long-only constraint
    
    Returns:
    - numpy array of portfolio weights
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.xs(base_column, level=1, axis=1)
    
    # Compute log returns
    log_returns = np.log(dataset / dataset.shift(1)).dropna()

    # Compute mean returns and covariance matrix
    mu = log_returns.mean().values
    Sigma = log_returns.cov().values
    n = len(mu)
    w = cp.Variable(n)
    
    # Objective: maximize expected return - lambda * variance
    objective = cp.Maximize(mu.T @ w - lambda_param * cp.quad_form(w, Sigma))
    
    # Constraints: sum to 1, and optionally long-only
    if allow_short:
        constraints = [cp.sum(w) == 1]
    else:
        constraints = [w >= 0, cp.sum(w) == 1]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
        #print(f"Optimization status: {problem.status}")
        #print(np.array(w.value).flatten())
        if w.value is not None:
            return np.array(w.value).flatten() , log_message
        else:
            # If optimization fails, return equal weights
            #print("ERROR: Optimization failed, returning equal weights.")
            log_message="ERROR: Optimization failed, returning equal weights."
            return np.ones(n) / n , log_message
    except Exception as e:
        # If optimization fails, return equal weights
        #print(f"ERROR: Optimization failed: {e}, returning equal weights.")
        log_message=f"ERROR: Optimization failed: {e}, returning equal weights."
        return np.ones(n) / n , log_message

def inverse_volatility_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", **kwargs):
    """
    Inverse Volatility Portfolio (IVP): weights inversely proportional to volatility
    
    Parameters:
    - dataset: DataFrame with price data
    - base_column: column name to extract from multi-level columns
    
    Returns:
    - numpy array of portfolio weights
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.xs(base_column, level=1, axis=1)
    
    # Compute log returns
    log_returns = np.log(dataset / dataset.shift(1)).dropna()
    
    # Check if we have enough data
    if len(log_returns) < 2:
        # Not enough data - return equal weights
        n = len(dataset.columns)
        #print("ERROR: IVP - Not enough data, returning equal weights.")
        log_message = "ERROR: IVP - Not enough data, returning equal weights."
        return np.ones(n) / n, log_message
    
    # Compute covariance matrix
    cov_matrix = log_returns.cov().values
    
    # Extract volatilities (standard deviations) from diagonal
    volatilities = np.sqrt(np.diag(cov_matrix))
    
    # Check for zero or near-zero volatility
    if np.any(volatilities < 1e-8):
        # Handle edge case: if any volatility is too small, use equal weights
        n = len(volatilities)
        #print("ERROR: Optimization failed, returning equal weights.")
        log_message = "ERROR: Optimization failed, returning equal weights."
        return np.ones(n) / n, log_message

    # Compute inverse volatility weights
    inv_vol = 1.0 / volatilities
    weights = inv_vol / np.sum(inv_vol)

    return weights, log_message

def vanilla_risk_parity_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", method:str="cvxpy", max_iter=2000, tol=1e-6, **kwargs):
    """
    Vanilla Risk Parity Portfolio (B2): Equal risk contribution from all assets
    
    Solves for weights where each asset contributes equally to portfolio risk:
        RC_i = w_i * (Σw)_i = 1/n * σ_p    for all i
    
    Parameters:
    - dataset: DataFrame with price data
    - base_column: column name to extract from multi-level columns
    - method: 'cvxpy' (default, most robust) or 'iterative' (faster but less stable)
    - max_iter: maximum iterations for convergence (iterative method only)
    - tol: convergence tolerance
    
    Returns:
    - numpy array of portfolio weights
    
    Notes:
    - CVXPY method uses convex optimization for guaranteed convergence
    - Iterative method uses Newton-Raphson with backtracking line search
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.xs(base_column, level=1, axis=1)
    
    # Compute log returns
    log_returns = np.log(dataset / dataset.shift(1)).dropna()
    
    # Check if we have enough data
    if len(log_returns) < 2:
        # Not enough data - return equal weights
        #print ("ERROR: RPP - Not enough data, returning equal weights.")
        log_message = "ERROR: RPP - Not enough data, returning equal weights."
        n = len(dataset.columns)
        return np.ones(n) / n, log_message
    
    # Compute covariance matrix (NOT correlation!)
    Sigma = log_returns.cov().values
    n = Sigma.shape[0]
    
    # Add small regularization for numerical stability
    Sigma = Sigma + 1e-8 * np.eye(n)
    
    if method == "cvxpy":
        result, log_message = _risk_parity_cvxpy(Sigma, n)
        return result, log_message
    else:
        result, log_message = _risk_parity_iterative(Sigma, n, max_iter, tol)
        return result, log_message

def _risk_parity_cvxpy(Sigma, n):
    """
    Solve risk parity using Sequential Convex Programming (SCP).
    
    Risk parity is non-convex, so we use SCP: iteratively solve convex 
    approximations around current weights until convergence.
    
    At each iteration, we linearize the non-convex risk contribution terms
    and solve a quadratic program.
    """
    log_message = ""
    try:
        # Initial guess: inverse volatility weights
        vol = np.sqrt(np.diag(Sigma))
        w_current = (1.0 / vol)
        w_current = w_current / np.sum(w_current)
        
        max_scp_iter = 50
        tol = 1e-6
        
        for scp_iter in range(max_scp_iter):
            # Define optimization variable
            w = cp.Variable(n, nonneg=True)
            
            # Linearize risk contributions around current point
            # RC_i = w_i * (Σw)_i ≈ w_i * (Σw_current)_i + (w - w_current)_i * (Σw_current)_i
            marginal_risk = Sigma @ w_current
            
            # Approximate risk contributions (first-order Taylor expansion)
            rc_approx = cp.multiply(w, marginal_risk)
            
            # Target: equal risk contributions
            target_rc = cp.sum(rc_approx) / n
            
            # Minimize squared deviations from equal risk contribution
            objective = cp.Minimize(cp.sum_squares(rc_approx - target_rc))
            
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
                #print(f"WARNING: RPP SCP failed at iteration {scp_iter}, using iterative method")
                log_message=f"WARNING: RPP SCP failed at iteration {scp_iter}, using iterative method"
                return _risk_parity_iterative(Sigma, n, max_iter=2000, tol=1e-6), log_message

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
        #print(f"WARNING: RPP SCP did not fully converge, returning best solution {weight_change}")
        log_message=f"WARNING: RPP SCP did not fully converge, returning best solution {weight_change}"
        return w_current, log_message

    except Exception as e:
        #print(f"ERROR: RPP SCP optimization error: {e}, using iterative method")
        log_message=f"ERROR: RPP SCP optimization error: {e}, using iterative method"
        return _risk_parity_iterative(Sigma, n, max_iter=2000, tol=1e-6), log_message

def _risk_parity_iterative(Sigma, n, max_iter, tol):
    """
    Solve risk parity using improved iterative Newton-Raphson method.
    
    This uses a cyclical coordinate descent with backtracking line search
    for better convergence properties than simple fixed-point iteration.
    """
    log_message = ""
    # Equal risk budgets
    b = np.ones(n) / n
    
    # Initial weights (inverse volatility as better starting point)
    vol = np.sqrt(np.diag(Sigma))
    w = (1.0 / vol)
    w = w / np.sum(w)
    
    # Iterative algorithm with improved update rule
    for iteration in range(max_iter):
        # Calculate portfolio volatility
        portfolio_var = w.T @ Sigma @ w
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Prevent division by zero
        if portfolio_vol < 1e-10:
            #print("ERROR: RPP - Portfolio volatility too low, returning equal weights.")
            log_message = "ERROR: RPP - Portfolio volatility too low, returning equal weights.\n"
            return np.ones(n) / n, log_message

        # Calculate marginal risk contributions and risk contributions
        marginal_contrib = Sigma @ w  # MRC_i = (Σw)_i
        risk_contrib = w * marginal_contrib  # RC_i = w_i * MRC_i
        
        # Target risk contribution for each asset
        target_rc = portfolio_var * b  # Target: 1/n of total portfolio variance
        
        # Newton-Raphson update with dampening
        # Update formula: w_new = w * (target_rc / current_rc)
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
    final_risk_contrib = w * (Sigma @ w)
    rc_std = np.std(final_risk_contrib)
    rc_mean = np.mean(final_risk_contrib)
    
    if rc_std / rc_mean < 0.5:  # Risk contributions are reasonably equal
        #print(f"WARNING: RPP did not fully converge but solution is acceptable (CV: {rc_std/rc_mean:.3f})")
        log_message += f"WARNING: RPP did not fully converge but solution is acceptable (CV: {rc_std/rc_mean:.3f})\n"
        return w, log_message
    else:
        #print(f"ERROR: RPP optimization did not converge to acceptable solution (CV: {rc_std/rc_mean:.3f})")
        log_message += f"ERROR: RPP optimization did not converge to acceptable solution (CV: {rc_std/rc_mean:.3f})\n"
        return np.ones(n) / n, log_message

def maximum_sharpe_ratio_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", allow_short:bool=False, **kwargs):
    """
    Maximum Sharpe Ratio Portfolio (MSRP) using the dual formulation
    
    Solves: Minimize w'Σw subject to μ'w = 1, optionally w >= 0
    Then normalizes: w_final = w / sum(w)
    
    This is equivalent to maximizing the Sharpe ratio (assuming risk-free rate = 0)
    
    Parameters:
    - dataset: DataFrame with price data
    - base_column: column name to extract from multi-level columns
    - allow_short: if True, allows short positions (negative weights); if False, long-only constraint
    
    Returns:
    - numpy array of portfolio weights
    
    Notes:
    - Converts pandas objects to numpy arrays to avoid shape/typing issues with cvxpy
    - Accepts 'optimal_inaccurate' solutions as these are often good enough for financial applications
    - Uses regularization to handle ill-conditioned covariance matrices
    - If the optimization is infeasible or fails, falls back to equal weights.
    - It usually fails when all mu are < 0
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        extracted_data = dataset.xs(base_column, level=1, axis=1)
        if isinstance(extracted_data, pd.Series):
            # Convert Series to DataFrame
            dataset = extracted_data.to_frame()
        else:
            dataset = extracted_data
    
    # Compute log returns
    log_returns_raw = np.log(dataset / dataset.shift(1))
    log_returns = log_returns_raw.dropna()
    
    # Check if we have enough data
    if len(log_returns) < 2:
        n = len(dataset.columns)
        #print("ERROR: MSRP - Not enough data, returning equal weights.")
        log_message = "ERROR: MSRP - Not enough data, returning equal weights.\n"
        return np.ones(n) / n, log_message

    # Compute expected returns and covariance matrix
    mu = log_returns.mean().values  # Convert to numpy array
    Sigma = log_returns.cov().values  # Convert to numpy array
    n = mu.shape[0]
    
    # Add small regularization to covariance matrix for numerical stability
    # This helps with ill-conditioned matrices
    regularization = 1e-8 * np.eye(n)
    Sigma_reg = Sigma + regularization
    
    # Check for non-positive expected returns
    if np.all(mu <= 0):
        # If all returns are non-positive, fallback to equal weights
        #print("ERROR: MSRP - Non-positive expected returns, returning equal weights.")
        log_message = "ERROR: MSRP - Non-positive expected returns, returning equal weights.\n"
        return np.ones(n) / n, log_message

    try:
        # Define the optimization variables
        if allow_short:
            w = cp.Variable(n)  # Allow negative weights
        else:
            w = cp.Variable(n, nonneg=True)  # Long-only
        
        # Objective: minimize portfolio variance (use regularized covariance matrix)
        objective = cp.Minimize(cp.quad_form(w, Sigma_reg))
        
        # Constraint: expected return requirement (use proper cvxpy constraint)
        constraints = [mu.T @ w == 1]
        
        # Solve the problem with ECOS first (good for QP problems)
        prob = cp.Problem(objective, constraints)
        
        # Try ECOS with relaxed tolerances (financial data doesn't need extreme precision)
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=10000, 
                   abstol=1e-5, reltol=1e-5, feastol=1e-5)
        
        # Accept both OPTIMAL and OPTIMAL_INACCURATE as valid solutions
        acceptable_statuses = [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
        
        # If ECOS didn't work well, try SCS solver
        if prob.status not in acceptable_statuses:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-4)
        
        # Validate solution and normalize
        if prob.status in acceptable_statuses and w.value is not None:
            w_unnormalized = np.asarray(w.value).reshape(-1)
            
            if not allow_short:
                # Clean up small negative values (numerical artifacts) for long-only
                w_unnormalized = np.maximum(w_unnormalized, 0)
            
            s = np.sum(w_unnormalized)
            if abs(s) > 1e-6:  # Check we have meaningful weights
                w_final = w_unnormalized / s
                return w_final, log_message
            else:
                # Fallback to equal weights
                #print("ERROR: MSRP - Infeasible solution (sum too small), returning equal weights.")
                log_message = "ERROR: MSRP - Infeasible solution (sum too small), returning equal weights.\n"
                return np.ones(n) / n, log_message
        else:
            # Optimization failed - fallback to equal weights
            #print(f"ERROR: MSRP optimization failed with status: {prob.status}")
            log_message = f"ERROR: MSRP optimization failed with status: {prob.status}\n"
            return np.ones(n) / n, log_message

    except Exception as e:
        # Any error - fallback to equal weights
        #print(f"MSRP optimization error: {e}")
        log_message = f"MSRP optimization error: {e}\n"
        return np.ones(n) / n, log_message

def most_diversified_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", **kwargs):
    """
    Most Diversified Portfolio (MDP)
    
    Maximizes the diversification ratio: DR = (w' * σ) / sqrt(w' * Σ * w)
    where σ is the vector of individual asset volatilities
    
    This is equivalent to MSRP where expected returns are replaced by individual volatilities
    
    Parameters:
    - dataset: DataFrame with price data
    - base_column: column name to extract from multi-level columns
    
    Returns:
    - numpy array of portfolio weights
    
    Notes:
    - Higher diversification ratio indicates better diversification
    - MDP tends to favor assets with lower correlations
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.xs(base_column, level=1, axis=1)
    
    # Compute log returns
    log_returns = np.log(dataset / dataset.shift(1)).dropna()
    
    # Check if we have enough data
    if len(log_returns) < 2:
        n = len(dataset.columns)
        #print("ERROR: MDP - Not enough data, returning equal weights.")
        log_message = "ERROR: MDP - Not enough data, returning equal weights."
        return np.ones(n) / n, log_message
    
    # Compute covariance matrix
    Sigma = log_returns.cov().values
    
    # Use individual volatilities as "expected returns" for MSRP formulation
    vol = np.sqrt(np.diag(Sigma))
    n = vol.shape[0]
    
    try:
        # Define the optimization variables
        w = cp.Variable(n, nonneg=True)
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        
        # Constraint: volatility-weighted sum equals 1
        constraints = [cp.sum(cp.multiply(vol, w)) == 1]
        
        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        # Validate solution and normalize
        if prob.status == cp.OPTIMAL and w.value is not None:
            w_unnormalized = np.asarray(w.value).reshape(-1)
            s = np.sum(w_unnormalized)
            if s > 0:
                w_final = w_unnormalized / s
                return w_final, log_message
            else:
                #print("ERROR: MDP - Infeasible solution, returning equal weights.")
                log_message = "ERROR: MDP - Infeasible solution, returning equal weights."
                return np.ones(n) / n, log_message
        else:
            #print("ERROR: MDP optimization failed, returning equal weights.")
            log_message = "ERROR: MDP optimization failed, returning equal weights."
            return np.ones(n) / n, log_message

    except Exception as e:
        # Any error - fallback to equal weights
        #print(f"MDP optimization error: {e}")
        log_message = f"MDP optimization error: {e}\n"
        return np.ones(n) / n, log_message

def maximum_decorrelation_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", **kwargs):
    """
    Maximum Decorrelation Portfolio (MDCP)
    
    Applies Global Minimum Variance Portfolio (GMVP) to the correlation matrix
    instead of the covariance matrix. This focuses purely on minimizing correlation
    exposure without being influenced by individual asset volatilities.
    
    Parameters:
    - dataset: DataFrame with price data
    - base_column: column name to extract from multi-level columns
    
    Returns:
    - numpy array of portfolio weights
    
    Notes:
    - MDCP minimizes the weighted average correlation in the portfolio
    - Lower portfolio correlation indicates better diversification
    """
    log_message = ""
    dataset = pd.DataFrame(dataset)
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset = dataset.xs(base_column, level=1, axis=1)
    
    # Compute log returns
    log_returns = np.log(dataset / dataset.shift(1)).dropna()
    
    # Check if we have enough data
    if len(log_returns) < 2:
        n = len(dataset.columns)
        #print("ERROR: MDCP - Not enough data, returning equal weights.")
        log_message = "ERROR: MDCP - Not enough data, returning equal weights."
        return np.ones(n) / n, log_message

    # Compute covariance matrix
    Sigma = log_returns.cov().values
    
    # Convert to correlation matrix
    # Correlation matrix C = D^(-1/2) * Sigma * D^(-1/2)
    # where D is diagonal matrix of variances
    vol = np.sqrt(np.diag(Sigma))
    
    # Check for zero volatilities
    if np.any(vol <= 0):
        n = len(vol)
        #print("ERROR: MDCP - Zero volatility detected, returning equal weights.")
        log_message = "ERROR: MDCP - Zero volatility detected, returning equal weights."
        return np.ones(n) / n, log_message

    D_inv_sqrt = np.diag(1.0 / vol)
    C = D_inv_sqrt @ Sigma @ D_inv_sqrt
    
    # Apply GMVP to correlation matrix
    n = C.shape[0]
    
    try:
        # Define the optimization variables
        w = cp.Variable(n, nonneg=True)
        
        # Objective: minimize portfolio variance on correlation matrix
        objective = cp.Minimize(cp.quad_form(w, C))
        
        # Constraints: weights sum to 1
        constraints = [cp.sum(w) == 1]
        
        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if prob.status == cp.OPTIMAL and w.value is not None:
            return np.asarray(w.value).reshape(-1), log_message
        else:
            #print(f"ERROR: MDCP optimization failed {prob.status}, returning equal weights.")
            log_message = f"ERROR: MDCP optimization failed {prob.status}, returning equal weights."
            return np.ones(n) / n, log_message

    except Exception as e:
        # Any error - fallback to equal weights
        #print(f"MDCP optimization error: {e}")
        log_message = f"MDCP optimization error: {e}\n"
        return np.ones(n) / n, log_message



class QuintileStrategy(PortfolioRebalanceStrategy):
    """Quintile momentum strategy"""
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = quintile_portfolio_fun

class GMVPStrategy(PortfolioRebalanceStrategy):
    """Global Minimum Variance Portfolio strategy
    
    By default, uses long-only constraint. To allow short selling, pass 
    allow_short=True via portfolio_func_kwargs in backtest configuration.
    """
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = gmvp_portfolio_fun

class MarkowitzStrategy(PortfolioRebalanceStrategy):
    """Markowitz mean-variance optimization strategy
    
    By default, uses long-only constraint. To allow short selling, pass 
    allow_short=True via portfolio_func_kwargs in backtest configuration.
    """
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = markowitz_portfolio_fun

class EqualWeightStrategy(PortfolioRebalanceStrategy):
    """Equal weight benchmark strategy"""
    def __init__(self):
        super().__init__()
        # No portfolio_func - will default to equal weights

class InverseVolatilityStrategy(PortfolioRebalanceStrategy):
    """Inverse Volatility Portfolio (IVP) strategy"""
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = inverse_volatility_portfolio_fun

class VanillaRiskParityStrategy(PortfolioRebalanceStrategy):
    """Vanilla Risk Parity Portfolio strategy - equal risk contribution"""
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = vanilla_risk_parity_portfolio_fun

class MaximumSharpeRatioStrategy(PortfolioRebalanceStrategy):
    """Maximum Sharpe Ratio Portfolio (MSRP) strategy
    
    By default, uses long-only constraint. To allow short selling, pass 
    allow_short=True via portfolio_func_kwargs in backtest configuration.
    """
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = maximum_sharpe_ratio_portfolio_fun

class MostDiversifiedStrategy(PortfolioRebalanceStrategy):
    """Most Diversified Portfolio (MDP) strategy"""
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = most_diversified_portfolio_fun

class MaximumDecorrelationStrategy(PortfolioRebalanceStrategy):
    """Maximum Decorrelation Portfolio (MDCP) strategy"""
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = maximum_decorrelation_portfolio_fun


