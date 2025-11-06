# ------------------------------------------------------------------------------
# Hierarchical Risk-Based Portfolios (HRPP) Toy example
# ------------------------------------------------------------------------------

# 4-asset split factor example

rm(list=ls(all.names = TRUE))
graphics.off()
close.screen(all.screens = TRUE)
erase.screen()
windows.options(record=TRUE)
options(scipen=999)

library(pacman)
p_load(riskParityPortfolio)

# ------------------------------------------------------------------------------
# Correlation matrix (A,B) vs (C,D)
# ------------------------------------------------------------------------------

# Correlation matrix (A,B) vs (C,D)
corr <- matrix(c(
  1.0, 0.8, 0.2, 0.1,
  0.8, 1.0, 0.3, 0.2,
  0.2, 0.3, 1.0, 0.7,
  0.1, 0.2, 0.7, 1.0
), nrow = 4, byrow = TRUE)
colnames(corr) <- rownames(corr) <- c("A","B","C","D"); corr

# A & B are highly correlated (cluster 1)
# C & D are highly correlated (cluster 2)

# Vols higher risk for (A,B), lower for (C,D)
vol <- c(A=0.20, B=0.20, C=0.12, D=0.12)   # c(A=0.25, B=0.20, C=0.15, D=0.12)

# Covariance from corr and vol
cov_mat <- (vol %o% vol) * corr

# ---- Helpers ----

# Distance used in HRP (López de Prado): d_ij = sqrt(0.5*(1 - rho_ij))
corr_to_dist <- function(R) sqrt(0.5 * (1 - R))

# Inverse-variance portfolio inside a sub-cluster
ivp <- function(cov_sub) {
  inv_var <- 1 / diag(cov_sub)
  w <- inv_var / sum(inv_var)
  as.numeric(w)
}

# Cluster variance using IVP weights
cluster_var <- function(cov_full, idx) {
  S <- cov_full[idx, idx, drop = FALSE]
  w <- ivp(S)
  as.numeric(t(w) %*% S %*% w)
}


# Optional: show the top-level split factor numbers explicitly
# (split Root -> {A,B} vs {C,D})
# Get the leaf order used:
dist_mat <- corr_to_dist(corr); dist_mat
hc <- hclust(as.dist(dist_mat), method = "single")
plot(hc, col='red')

hc$merge        # merging of clusters at step i of the clustering
hc$height       # the value of the criterion associated with the clustering method for the particular agglomeration.
hc$order        # permutation of the original observations suitable for plotting
hc$dist.method  # the distance metric used

dend <- as.dendrogram(hc)
ord <- order.dendrogram(dend)

# Identify the first split in that order
left_idx  <- ord[1:floor(length(ord)/2)]
right_idx <- ord[(floor(length(ord)/2)+1):length(ord)]

v_left  <- cluster_var(cov_mat, left_idx); v_left 
# Since A and B are very correlated, their combined variance is relatively high

v_right <- cluster_var(cov_mat, right_idx); v_right
# Their correlation is high but less than A-B

alpha_left <- 1 - v_left/(v_left + v_right)  # allocation to 'left' cluster

cat("\nTop-level cluster variances:\n")
cat(" Var(left) =", round(v_left, 6), " Var(right) =", round(v_right, 6), "\n")
cat(" Split factor (left vs right):",
    paste0(round(100*c(alpha_left, 1 - alpha_left), 2), "%"), "\n")

# Within-Cluster Split
# Cluster 1 (A vs B): Since they're symmetric, each gets aprox. 12.5%
alpha_left/2

# Cluster 2 (C vs D): Each gets aprox 37.5%.
(1-alpha_left)/2


# Full algorithm
# Recursive bisection applying the split factor
hrp_weights <- function(cov_full, corr_full) {
  # 1) Hierarchical clustering on correlation distances
  dist_mat <- corr_to_dist(corr_full)
  hc <- hclust(as.dist(dist_mat), method = "single")
  dend <- as.dendrogram(hc)
  order_leaves <- order.dendrogram(dend)  # quasi-diagonalization
  
  # Reorder covariance by leaf order
  S <- cov_full[order_leaves, order_leaves]
  
  n <- nrow(S)
  w <- rep(1, n)  # start equal, will be scaled down multiplicatively
  clusters <- list(seq_len(n))
  
  # Split until leaves
  while (length(clusters) > 0) {
    new_clusters <- list()
    for (cl in clusters) {
      if (length(cl) <= 1) next
      k <- floor(length(cl) / 2)
      left  <- cl[1:k]
      right <- cl[(k+1):length(cl)]
      
      # Split factor = allocate by inverse cluster variances
      vL <- cluster_var(S, left)
      vR <- cluster_var(S, right)
      # alpha is the weight given to 'left' cluster
      alpha <- 1 - vL / (vL + vR)  # equivalent to 1 / (1 + vL/vR)
      w[left]  <- w[left]  * alpha
      w[right] <- w[right] * (1 - alpha)
      
      new_clusters <- c(new_clusters, list(left), list(right))
    }
    clusters <- new_clusters
  }
  
  # Map weights back to original asset order
  w_full <- rep(NA_real_, n)
  w_full[order_leaves] <- w
  names(w_full) <- rownames(cov_full)
  # Normalize (numerical safety)
  w_full / sum(w_full)
}

# ---- Run HRP and show results ----

w <- hrp_weights(cov_mat, corr)

print(round(w, 6))



barplotPortfolioRisk(w, cov_mat, col = "#A29BFE")

