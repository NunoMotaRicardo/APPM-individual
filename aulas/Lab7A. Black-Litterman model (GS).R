# ------------------------------------------------------------------------------
# Black Litterman allocation model - Goldman Sachs 1999 paper
# ------------------------------------------------------------------------------

rm(list=ls(all.names = TRUE))
graphics.off()
close.screen(all.screens = TRUE)
erase.screen()
windows.options(record=TRUE)
options(scipen=999)

library(pacman)
p_load(ggplot2, PerformanceAnalytics, rstudioapi, readxl)

# ------------------------------------------------------------------------------
# Load volatility and correlation data
# ------------------------------------------------------------------------------

setwd('C:/Users/Jorge Bravo/Desktop/Teaching/Asset Pricing Portfolio Theory/Handouts')

Mkt_data <- read_excel('Black_Litterman_Model_examples.xlsx', 
                       sheet="BL Goldman Sachs paper", range = "A5:D12", 
                       col_names=T)

Mkt_cor <- as.matrix(read_excel('Black_Litterman_Model_examples.xlsx', 
                       sheet="BL Goldman Sachs paper", range = "B16:H22", 
                       col_names=F))
dimnames(Mkt_cor) <- list(Mkt_data$Country, Mkt_data$Country)


# Asset volatility
vol = Mkt_data$vol/100
names(vol) = Mkt_data$Country

# Portfolio weights
w0  = Mkt_data$w/100
names(w0) = Mkt_data$Country

# CAPM expected return
mu  = Mkt_data$ER/100
names(mu) = Mkt_data$Country

C = Mkt_cor

# Create a diagonal matrix of standard deviations
diag_sd <- diag(vol)

# Calculate the variance-covariance matrix
Sigma <- diag_sd %*% C %*% diag_sd

sd_P = sqrt(t(w0) %*% Sigma %*% w0); sd_P

# Portfolio return
Rp = sum(mu * w0); Rp


# Target Sharpe Ratio
SR = 0.25

# Risk-aversion coefficient
A = SR/sd_P; A

# Risk-free rate
rf = 0.03

# Returns optimal allocation
# Implied return coherent with initial portfolio allocation w0 
mu_imp = rf + Sigma %*% w0 %*% A
mu_imp

# or the alternative
rf + SR*(Sigma %*% w0) %*% (1/sd_P)

# Risk premium
risk_premium = mu_imp-rf; risk_premium
# or
SR*(Sigma %*% w0) %*% (1/sd_P)


# Specifying the portfolio manager views

# Link Matrix (P)
P = matrix(c(0,	0, 1, 0, 0, 0, 0,
             0,	0, 0, 1, 0, 0, 0,
             0,	0, 0, 0, 0, 1, 0), nrow=3, byrow = T,
           dimnames = list(c('L1','L2','L3'),Mkt_data$Country))

Q = c(0.075, 0.085, 0.055)



omega = matrix(c(0.0^2,	0, 0 ,
                 0, 0.0^2, 0,
                 0, 0, 0.0^2), nrow=3, 
               dimnames = list(c('L1','L2','L3'),c('L1','L2','L3')))



# tau
tau = 1        
rho = tau * Sigma

# Conditional expected returns mu_bar
mu_bar = mu_imp + rho %*% t(P) %*% solve((P %*% rho %*% t(P) + omega)) %*% (Q - P %*% mu_imp)

mu_bar
mu_bar-mu_imp


# Black-Litterman portfolios
# Portfolio optimization

library(PerformanceAnalytics)
library(CVXR)
Markowitz <- function(mu, Sigma, lmd = 0.5, A, rf) {
  w <- Variable(nrow(Sigma))
  prob <- Problem(Minimize(lmd*quad_form(w, Sigma) - (1/A) * (t(mu) %*% w - rf)),
                  constraints = list(w >= 0, sum(w) == 1))
  result <- solve(prob)
  return(as.vector(result$getValue(w)))
}

w_BL = Markowitz (mu=mu_bar, Sigma=Sigma, A=A, rf=rf)
names(w_BL) = Mkt_data$Country
round(w_BL, 4)

# Tracking error volatility

# TE_vol = sqrt((x-x0)' * Sigma *(x-x0))
# x=BL weights; x0: benchmark weights
x = w_BL-w0
TE_vol = sqrt(t(x) %*% Sigma %*% x); TE_vol*100




