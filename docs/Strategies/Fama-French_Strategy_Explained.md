# Fama-French Factor Models: A Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Historical Context](#historical-context)
3. [The Three-Factor Model](#the-three-factor-model)
4. [The Five-Factor Model](#the-five-factor-model)
5. [Mathematical Formulation](#mathematical-formulation)
6. [Implementation Details](#implementation-details)
7. [Practical Applications](#practical-applications)
8. [Advantages and Limitations](#advantages-and-limitations)
9. [Code Walkthrough](#code-walkthrough)

---

## Introduction

The **Fama-French models** are statistical models designed to explain stock returns through multiple risk factors beyond the traditional market beta. Developed by Nobel laureate Eugene Fama and Kenneth French, these models have become fundamental tools in asset pricing, portfolio management, and performance evaluation.

The models address a critical limitation of the Capital Asset Pricing Model (CAPM): while CAPM explains returns using only market risk (beta), empirical evidence shows that other characteristics—particularly company size and value metrics—also systematically affect returns.

---

## Historical Context

### The CAPM Foundation

The Capital Asset Pricing Model (CAPM), developed in the 1960s, proposes that asset returns can be explained by a single factor:

```
r = Rf + β(Rm - Rf)
```

Where:
- `r` = Expected return of the asset
- `Rf` = Risk-free rate
- `β` = Market beta (systematic risk)
- `Rm` = Market return

### Empirical Anomalies

By the early 1990s, researchers had identified persistent patterns that CAPM couldn't explain:

1. **Size Effect**: Historically, small-cap stocks outperformed large-cap stocks on a risk-adjusted basis
2. **Value Premium**: Stocks with high book-to-market ratios (value stocks) outperformed growth stocks
3. **Beta Inconsistencies**: When controlling for size, beta's predictive power disappeared

### The Fama-French Response

In their seminal 1992-1993 papers, Fama and French demonstrated that adding two factors—size and value—explained over 90% of diversified portfolio returns, compared to CAPM's 70%.

---

## The Three-Factor Model

### Model Equation

The Fama-French three-factor model extends CAPM by adding size and value factors:

```
r - Rf = α + β₁(Rm - Rf) + β₂·SMB + β₃·HML + ε
```

### The Three Factors

#### 1. **Market Factor (Rm - Rf)**
- **Definition**: Excess return of the market over the risk-free rate
- **Interpretation**: Captures broad market movements
- **Proxy**: Typically the S&P 500 or total stock market index minus Treasury bills

#### 2. **SMB (Small Minus Big)**
- **Definition**: Average return on small-cap portfolios minus average return on large-cap portfolios
- **Interpretation**: Captures the size premium—the historical tendency of smaller companies to outperform larger ones
- **Construction**: 
  - Rank all stocks by market capitalization
  - Create portfolios of small and big stocks
  - SMB = Return(Small Portfolio) - Return(Big Portfolio)

#### 3. **HML (High Minus Low)**
- **Definition**: Average return on high book-to-market portfolios minus low book-to-market portfolios
- **Interpretation**: Captures the value premium—the tendency of value stocks to outperform growth stocks
- **Construction**:
  - Calculate book-to-market ratio (Book Value / Market Value) for all stocks
  - Create portfolios of high B/M (value) and low B/M (growth) stocks
  - HML = Return(High B/M Portfolio) - Return(Low B/M Portfolio)

### Interpretation of Coefficients

- **α (Alpha)**: Stock's excess return not explained by the three factors. In efficient markets, alpha should be zero.
- **β₁**: Sensitivity to market movements (similar to CAPM beta)
- **β₂**: Sensitivity to the size factor (positive = behaves like small-cap, negative = like large-cap)
- **β₃**: Sensitivity to the value factor (positive = behaves like value stock, negative = like growth stock)

---

## The Five-Factor Model

### Evolution and Motivation

In 2015, Fama and French extended their model to include two additional factors that capture patterns in stock returns related to profitability and investment:

```
r - Rf = α + β₁(Rm - Rf) + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + ε
```

### The Additional Factors

#### 4. **RMW (Robust Minus Weak)**
- **Definition**: Difference between returns of firms with robust (high) profitability and weak (low) profitability
- **Interpretation**: Captures the profitability premium—more profitable companies tend to have higher returns
- **Construction**:
  - Calculate operating profitability (revenues minus COGS, interest, and SG&A, divided by book equity)
  - Sort stocks into high and low profitability portfolios
  - RMW = Return(High Profitability) - Return(Low Profitability)

#### 5. **CMA (Conservative Minus Aggressive)**
- **Definition**: Difference between returns of firms that invest conservatively versus aggressively
- **Interpretation**: Captures the investment premium—firms that invest conservatively tend to have higher returns
- **Construction**:
  - Calculate investment rate (change in total assets divided by lagged total assets)
  - Sort stocks into conservative (low investment) and aggressive (high investment) portfolios
  - CMA = Return(Conservative Investment) - Return(Aggressive Investment)

### Key Findings

1. **Improved Explanatory Power**: The five-factor model captures more variation in stock returns than the three-factor model
2. **HML Redundancy**: In US data, HML becomes largely redundant when RMW and CMA are included (correlation of 0.7 between HML and CMA)
3. **Small Stock Anomaly**: The model still struggles to explain returns of small, unprofitable firms that invest heavily

---

## Mathematical Formulation

### Matrix Form

For multiple assets simultaneously:

```
R = α + F·β + ε
```

Where:
- `R` is an (T × N) matrix of excess returns for N assets over T periods
- `F` is a (T × K) matrix of K factor returns
- `β` is a (K × N) matrix of factor loadings
- `α` is a (1 × N) vector of intercepts (alphas)
- `ε` is a (T × N) matrix of residuals

### Estimation via OLS

The factor loadings are estimated using ordinary least squares (OLS) regression:

```
β̂ = (F'F)⁻¹F'R
```

### Covariance Structure

The Fama-French model implies a specific covariance structure:

```
Σ = β·Ω·β' + Ψ
```

Where:
- `Σ` is the asset covariance matrix
- `Ω` is the factor covariance matrix
- `Ψ` is a diagonal matrix of idiosyncratic (residual) variances
- `β'` denotes the transpose of β

This decomposition separates:
- **Systematic risk** (β·Ω·β'): Risk explained by common factors
- **Idiosyncratic risk** (Ψ): Asset-specific risk

---

## Implementation Details

### Data Sources

#### Stock Returns
- **Source**: Yahoo Finance, Bloomberg, or other financial data providers
- **Frequency**: Daily, weekly, or monthly returns
- **Calculation**: Typically log returns: `ln(Pt / Pt-1)`

#### Fama-French Factors
- **Source**: [Kenneth French's Data Library](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- **Available**: Daily, weekly, and monthly frequencies
- **Coverage**: US, developed markets, emerging markets, regional factors
- **Format**: CSV files containing factor returns in percentage points

### Regression Steps

1. **Data Preparation**
   - Download stock prices and calculate returns
   - Load Fama-French factor data
   - Align dates between stock returns and factor data
   - Convert factor returns from percentages to decimals

2. **Regression Estimation**
   - For each stock, regress excess returns on factor returns
   - Estimate coefficients using OLS
   - Extract alpha, betas, and residuals

3. **Model Evaluation**
   - Examine R-squared (proportion of variance explained)
   - Test statistical significance of alpha (should be zero in efficient markets)
   - Analyze residuals for patterns (should be white noise)

4. **Performance Analysis**
   - Compare alphas across stocks
   - Identify factor exposures (which factors drive each stock's returns)
   - Construct factor-mimicking portfolios

### Python Implementation Example

Based on the provided notebook, here's a simplified workflow:

```python
# 1. Load stock returns
stock_returns = np.log(prices / prices.shift(1)).dropna()

# 2. Load Fama-French factors
ff_factors, ff_rf = load_fama_french_daily("5F")

# 3. Align data
ff_factors = ff_factors.loc[stock_returns.index]
aligned_returns = stock_returns.loc[ff_factors.index]

# 4. Run regression (matrix form)
y = aligned_returns.to_numpy()
X = np.column_stack([np.ones(len(ff_factors)), ff_factors.to_numpy()])
coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

# 5. Extract results
alpha = coeffs[0]  # Intercepts
beta = coeffs[1:]  # Factor loadings
```

---

## Practical Applications

### 1. Performance Evaluation

**Risk-Adjusted Performance**: Alpha measures manager skill after controlling for factor exposures.

**Example**: A fund with α = 2% per year has generated 2% excess return beyond what its factor exposures would predict.

### 2. Portfolio Construction

**Factor-Based Investing**: Construct portfolios that target specific factor exposures.

**Example Strategies**:
- **Value tilt**: Overweight high-HML stocks
- **Small-cap growth**: Positive SMB, negative HML exposure
- **Quality focus**: High RMW (profitability) exposure

### 3. Risk Management

**Factor Risk Decomposition**: Understand sources of portfolio risk.

```
Portfolio Variance = Systematic Risk (factor-driven) + Idiosyncratic Risk
```

**Benefits**:
- Identify unintended factor bets
- Hedge specific factor exposures
- Improve diversification

### 4. Asset Allocation

**Expected Return Estimation**: Use factor loadings and expected factor premiums to forecast returns.

```
E[r] = Rf + β₁·E[Rm - Rf] + β₂·E[SMB] + β₃·E[HML] + ...
```

### 5. Academic Research

- Testing market efficiency
- Evaluating new trading strategies
- Understanding cross-sectional return patterns

---

## Advantages and Limitations

### Advantages

#### 1. **Superior Explanatory Power**
- Explains 90%+ of diversified portfolio returns
- Significantly better than CAPM's ~70%

#### 2. **Empirically Grounded**
- Based on decades of historical data across multiple markets
- Robust across different time periods and geographies

#### 3. **Practical and Accessible**
- Factors are publicly available (Kenneth French's website)
- Easy to implement in standard statistical software
- Widely used in industry and academia

#### 4. **Risk Factor Framework**
- Provides economic intuition: factors represent systematic risks
- Helps distinguish luck (alpha) from systematic exposures (beta)

#### 5. **Versatile Applications**
- Performance evaluation
- Portfolio construction
- Risk management
- Asset pricing research

### Limitations

#### 1. **Data Mining Concerns**
- Critics argue factors were identified by searching historical data
- Risk of overfitting to past patterns that may not persist

#### 2. **Factor Instability**
- Factor premiums vary over time
- Some periods show negative premiums (e.g., growth outperforming value in 2010s)

#### 3. **Incomplete Explanation**
- Model fails Gibbons-Ross-Shanken test (factors don't fully explain all portfolios)
- Struggles with certain anomalies (e.g., small, unprofitable, high-investment firms)

#### 4. **Missing Factors**
- Momentum factor not included (though shown to be important)
- Other potential factors: liquidity, volatility, quality

#### 5. **Regional Specificity**
- Factors are country/region-specific (US factors don't work as well internationally)
- Requires local factor data for non-US markets

#### 6. **Interpretation Debate**
- **Risk-based view**: Factors represent compensation for systematic risks
- **Behavioral view**: Factors exploit investor biases and market inefficiencies
- No consensus on why factors work

#### 7. **Implementation Challenges**
- Transaction costs in replicating factor portfolios
- Factor timing is difficult
- Crowding concerns as strategies become popular

---

## Code Walkthrough

### Notebook Structure

The provided notebook (`Lab8. Fama-French 3 Factor model - Final.ipynb`) implements both three-factor and five-factor models in Python. Here's a detailed walkthrough:

### 1. Environment Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
```

**Purpose**: Import essential libraries for data manipulation, visualization, and statistical analysis.

### 2. Data Acquisition

```python
tickers = ["AAPL", "AMZN", "NFLX", "GOOG", "META", "XOM", "GM", "T", "WMT", "SBUX"]
stocks_raw = yf.download(tickers, start="2017-01-01", end="2025-09-30")
stock_prices = stocks_raw.xs("Adj Close", level=1, axis=1)
```

**Purpose**: Download historical adjusted close prices for a diverse set of stocks from Yahoo Finance.

**Portfolio composition**:
- Tech: AAPL, AMZN, NFLX, GOOG, META
- Energy: XOM
- Industrials: GM
- Telecom: T
- Retail: WMT, SBUX

### 3. Return Calculation

```python
stock_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
market_returns = np.log(market_prices / market_prices.shift(1)).dropna()
```

**Purpose**: Calculate continuously compounded (log) returns, which have better statistical properties than simple returns.

**Why log returns?**
- Time-additive: `r_total = r_1 + r_2 + ... + r_n`
- Symmetric for gains and losses
- Approximately normally distributed

### 4. Single-Factor Model (CAPM)

```python
# Calculate beta using covariance method
cov_xf = (X_centered * f_centered).sum(axis=0) / (len(f_centered) - 1)
market_var = (f_centered ** 2).sum() / (len(f_centered) - 1)
beta_ls = cov_xf / market_var
alpha_ls = stock_returns.mean() - beta_ls * market_returns.mean()
```

**Purpose**: Estimate CAPM parameters using the closed-form solution:
- β = Cov(r_stock, r_market) / Var(r_market)
- α = E[r_stock] - β·E[r_market]

### 5. Covariance Matrix Reconstruction

```python
Sigma_1factor = market_var * np.outer(beta_ls, beta_ls) + Psi_diag
```

**Purpose**: Reconstruct the full covariance matrix from:
- **Systematic component**: `market_var * β·β'` (all stocks move together due to market)
- **Idiosyncratic component**: `Ψ` (diagonal matrix of residual variances)

**Insight**: Single-factor models assume all covariance comes from shared market exposure.

### 6. Matrix Form Solution

```python
F_matrix = np.column_stack([np.ones(len(market_returns)), market_returns])
Gamma = np.linalg.lstsq(F_matrix, stock_returns, rcond=None)[0]
```

**Purpose**: Solve for all assets simultaneously using matrix algebra:
- More efficient than looping over individual stocks
- Equivalent to running separate regressions

### 7. Fama-French Factor Loading

```python
def load_fama_french_daily(model: str = "5F"):
    # Load from local zip/csv or Kenneth French's website
    # Parse dates, convert to decimals, separate factors from risk-free rate
    return factors, risk_free
```

**Purpose**: Load pre-computed Fama-French factors from Kenneth French's data library.

**Data format**:
- Dates in YYYYMMDD format
- Factor returns in percentage points (converted to decimals)
- Separate risk-free rate column

### 8. Multi-Factor Regression

```python
def fit_factor_model(asset_returns, factor_data):
    y = asset_returns.to_numpy()
    X = np.column_stack([np.ones(len(factor_data)), factor_data.to_numpy()])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = coeffs[0]
    beta = coeffs[1:]
    return alpha, beta, residuals
```

**Purpose**: Run multi-factor regression for all stocks at once.

**Output**:
- **Alpha**: Intercept (excess return not explained by factors)
- **Beta**: Factor loadings (sensitivities to each factor)
- **Residuals**: Unexplained returns (should be random)

### 9. Visualization

```python
sns.heatmap(corr_sigma, cmap="coolwarm", square=True)
```

**Purpose**: Visualize correlation structure implied by the factor model.

**Interpretation**:
- High off-diagonal correlations → strong common factor exposure
- Diagonal of Psi matrix → pure idiosyncratic risk

### 10. Statistical Testing

```python
ff_ols = sm.OLS(aapl_returns, X).fit()
ff_ols.summary()
```

**Purpose**: Perform statistical inference using statsmodels:
- Test if alpha is significantly different from zero
- Examine R-squared (model fit)
- Check statistical significance of factor loadings

**Key metrics**:
- **t-statistic for alpha**: Tests if excess returns exist after risk adjustment
- **R-squared**: Proportion of variance explained
- **F-statistic**: Overall model significance

---

## Conclusion

The Fama-French models represent a major advancement in asset pricing theory and practice. By expanding beyond the single-factor CAPM to incorporate size, value, profitability, and investment factors, these models provide:

1. **Better explanatory power** for cross-sectional stock returns
2. **Practical frameworks** for performance evaluation and risk management
3. **Economic insights** into sources of systematic risk and return

While not without limitations—including data mining concerns, factor instability, and incomplete explanation of all anomalies—the Fama-French models remain the industry standard for:
- Evaluating fund manager performance
- Understanding portfolio risk exposures
- Constructing factor-based investment strategies
- Academic research in asset pricing

As factor investing continues to evolve, with practitioners exploring additional factors like momentum, quality, and low-volatility, the Fama-French framework provides a solid foundation for understanding how and why different stocks earn different returns.

---

## Further Resources

### Academic Papers
- Fama, E. F., & French, K. R. (1992). "The Cross-Section of Expected Stock Returns." *Journal of Finance*, 47(2), 427-465.
- Fama, E. F., & French, K. R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33, 3-56.
- Fama, E. F., & French, K. R. (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics*, 116, 1-22.

### Data Sources
- [Kenneth French's Data Library](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) - Official source for Fama-French factors
- [CRSP](https://www.crsp.org/) - Center for Research in Security Prices (academic data)
- [Bloomberg](https://www.bloomberg.com/professional/product/portfolio-and-risk-analytics/) - Professional factor data

### Software Implementations
- **Python**: `statsmodels`, `pandas`, `numpy`
- **R**: `PerformanceAnalytics`, `FactorAnalytics`
- **MATLAB**: Financial Toolbox

### Online Courses
- Coursera: "Investment Management" specialization
- CFA Institute: Factor investing resources
- Quantopian/QuantConnect: Algorithmic trading tutorials

---

*Document created: October 2025*  
*Based on: Lab8. Fama-French 3 Factor model - Final.ipynb*
