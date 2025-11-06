import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from Backtester.BacktestFramework import PortfolioRebalanceStrategy

def hierarchical_risk_parity_portfolio_fun(dataset:pd.DataFrame, base_column:str="adjusted", 
                                           linkage_method:str='single', **kwargs):
    """
    Hierarchical Risk Parity Portfolio (HRPP)
    
    Uses hierarchical clustering and recursive bisection to allocate capital.
    Based on López de Prado's approach:
    1. Cluster assets using correlation distance
    2. Quasi-diagonalize via dendrogram ordering
    3. Recursively split clusters, allocating inversely to cluster variance
    
    Parameters:
    - dataset: DataFrame with price data
    - base_column: column name to extract from multi-level columns
    - linkage_method: hierarchical clustering linkage ('single', 'complete', 'average', 'ward')
    
    Returns:
    - numpy array of portfolio weights
    
    Notes:
    - 'single' linkage is the default (López de Prado recommendation)
    - Uses inverse variance portfolio (IVP) for cluster variance calculation
    - More robust than traditional mean-variance optimization
    - Naturally diversifies across hierarchical asset clusters
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
        log_message = "ERROR: HRPP - Not enough data, returning equal weights."
        return np.ones(n) / n, log_message
    
    # Compute correlation and covariance matrices
    corr_matrix = log_returns.corr().values
    cov_matrix = log_returns.cov().values
    n = corr_matrix.shape[0]
    
    # Check for invalid matrices
    if np.any(np.isnan(corr_matrix)) or np.any(np.isnan(cov_matrix)):
        log_message = "ERROR: HRPP - Invalid correlation/covariance matrix, returning equal weights."
        return np.ones(n) / n, log_message
    
    try:
        # Apply HRPP algorithm
        weights = _hrpp_weights(cov_matrix, corr_matrix, linkage_method)
        return weights, log_message
    
    except Exception as e:
        log_message = f"ERROR: HRPP optimization error: {e}, returning equal weights."
        return np.ones(n) / n, log_message


def _corr_to_dist(corr_matrix):
    """
    Convert correlation matrix to distance matrix for HRP.
    Distance metric from López de Prado: d_ij = sqrt(0.5 * (1 - rho_ij))
    
    This distance satisfies:
    - d_ij = 0 when rho_ij = 1 (perfect correlation)
    - d_ij = 1 when rho_ij = -1 (perfect negative correlation)
    - d_ij = sqrt(0.5) ≈ 0.707 when rho_ij = 0 (no correlation)
    """
    return np.sqrt(0.5 * (1 - corr_matrix))


def _ivp_weights(cov_sub):
    """
    Inverse Variance Portfolio (IVP) weights for a sub-cluster.
    Allocates inversely proportional to individual asset variances.
    """
    inv_var = 1.0 / np.diag(cov_sub)
    weights = inv_var / np.sum(inv_var)
    return weights


def _cluster_var(cov_full, indices):
    """
    Calculate cluster variance using IVP weights.
    This represents the portfolio variance if we invest only in this cluster.
    """
    cov_sub = cov_full[np.ix_(indices, indices)]
    w = _ivp_weights(cov_sub)
    variance = w @ cov_sub @ w
    return variance


def _hrpp_weights(cov_full, corr_full, linkage_method='single'):
    """
    Hierarchical Risk Parity portfolio weights.
    
    Algorithm:
    1. Compute distance matrix from correlations
    2. Perform hierarchical clustering
    3. Get quasi-diagonal ordering from dendrogram
    4. Recursively bisect clusters and allocate inversely to cluster variance
    
    Parameters:
    - cov_full: Covariance matrix
    - corr_full: Correlation matrix
    - linkage_method: Clustering method ('single', 'complete', 'average', 'ward')
    
    Returns:
    - Portfolio weights (numpy array)
    """
    # 1) Hierarchical clustering on correlation distances
    dist_mat = _corr_to_dist(corr_full)
    dist_condensed = squareform(dist_mat, checks=False)
    linkage_matrix = linkage(dist_condensed, method=linkage_method)
    
    # 2) Get leaf order (quasi-diagonalization) from dendrogram
    dend = dendrogram(linkage_matrix, no_plot=True)
    order_leaves = dend['leaves']
    
    # 3) Reorder covariance by leaf order
    cov_ordered = cov_full[np.ix_(order_leaves, order_leaves)]
    
    n = cov_ordered.shape[0]
    weights = np.ones(n)  # Start with equal weights, will be scaled multiplicatively
    
    # 4) Recursive bisection until we reach individual leaves
    clusters = [list(range(n))]  # Start with all assets as one cluster
    
    while clusters:
        new_clusters = []
        
        for cluster in clusters:
            if len(cluster) <= 1:
                # Single asset - no further splitting needed
                continue
            
            # Split cluster into two halves (based on dendrogram ordering)
            k = len(cluster) // 2
            left = cluster[:k]
            right = cluster[k:]
            
            # Calculate variance of each sub-cluster using IVP
            v_left = _cluster_var(cov_ordered, left)
            v_right = _cluster_var(cov_ordered, right)
            
            # Split factor: allocate inversely to cluster variances
            # Lower variance cluster gets higher weight
            alpha = 1 - v_left / (v_left + v_right)  # Weight for left cluster
            
            # Update weights multiplicatively
            weights[left] *= alpha
            weights[right] *= (1 - alpha)
            
            # Add sub-clusters to process in next iteration
            new_clusters.extend([left, right])
        
        clusters = new_clusters
    
    # 5) Map weights back to original asset order
    w_full = np.zeros(n)
    w_full[order_leaves] = weights
    
    # 6) Normalize (numerical safety)
    w_full = w_full / np.sum(w_full)
    
    return w_full


class HierarchicalRiskParityStrategy(PortfolioRebalanceStrategy):
    """Hierarchical Risk Parity Portfolio (HRPP) strategy - Hierarchical clustering with recursive bisection"""
    def __init__(self):
        super().__init__()
        self.params.portfolio_func = hierarchical_risk_parity_portfolio_fun