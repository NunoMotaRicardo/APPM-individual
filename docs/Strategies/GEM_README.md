# GEM Strategy Documentation - Overview

This folder contains comprehensive documentation for the Global Equities Momentum (GEM) strategy implementation.

## Document Structure

### 1. **GEM_User_Guide.md** ⭐ START HERE
**Audience**: Strategy users, backtesting practitioners, portfolio managers  
**Purpose**: Learn how to use the GEM strategy in practice

**Contains**:
- Plain-language explanation of how the strategy works
- Step-by-step setup instructions
- Parameter tuning guide with practical examples
- Real-world example comparisons
- Troubleshooting common issues
- FAQ section

**Best for**: Understanding what the strategy does and how to implement it in your backtests.

---

### 2. **GEM_Technical_Reference.md**
**Audience**: Developers, researchers, algorithm analysts  
**Purpose**: Deep dive into implementation details

**Contains**:
- Complete algorithm description with pseudocode
- Mathematical formulations (momentum scores, qualification criteria)
- Code architecture and class hierarchy
- Edge case handling
- Performance characteristics (complexity, memory, execution time)
- Unit test examples
- Optimization opportunities
- Known limitations and future enhancements

**Best for**: Understanding how the code works internally, debugging, or extending the strategy.

---

## Quick Start

### For Users (Want to run backtests)
1. Read **GEM_User_Guide.md** sections 1-3 (What is GEM, How it works, Setting up)
2. Follow the "Basic Setup" example
3. Use the "Parameter Tuning Guide" to customize
4. Check "Common Questions" if you encounter issues

### For Developers (Want to modify the code)
1. Skim **GEM_User_Guide.md** to understand the strategy concept
2. Read **GEM_Technical_Reference.md** sections on Algorithm Description and Implementation
3. Review the Code Architecture section
4. Check Edge Cases and Testing sections before making changes

### For Researchers (Want to analyze the strategy)
1. Read **GEM_User_Guide.md** section "How Does It Work?"
2. Study **GEM_Technical_Reference.md** Mathematical Formulation
3. Review Comparison with Reference Implementation
4. Check Known Limitations and Future Enhancements

---

## Key Concepts

### Momentum
The tendency of assets to continue moving in their current direction. Calculated as percentage price change over various lookback periods (1, 3, 6, 12 months).

### Consistency
Assets must show positive momentum across multiple time periods (e.g., 3 out of 4) to qualify. This filters out "false signals" from one-time price spikes.

### Absolute Momentum
Comparing risky assets not just to each other (relative momentum), but also to a safe benchmark (treasury bonds or cash). If risky assets aren't beating the safe option, the strategy goes defensive.

### Defensive Allocation
When no risky assets qualify (negative or weak momentum), the strategy either:
- Holds **cash** (zero positions, earn broker interest)
- Buys **treasury bonds** (specific ETF like BIL, SHY, IEF)

---

## Related Files

- **Source Code**: `Backtester/Strategy_dual_momentum.py`
- **Framework**: `Backtester/BacktestFramework.py`
- **Original Reference**: `references/GEM-master/gem.py`

---

## Document Updates

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | Nov 5, 2025 | Complete rewrite - split into User Guide and Technical Reference |
| 1.2 | Oct 19, 2025 | Added risk-free asset flexibility (cash vs treasury) |
| 1.1 | Oct 19, 2025 | Changed to direct day-based momentum periods |
| 1.0 | Oct 19, 2025 | Initial implementation documentation |

---

## Getting Help

- **Strategy questions**: Check GEM_User_Guide.md FAQ section
- **Implementation issues**: See GEM_Technical_Reference.md Edge Cases section
- **Parameter advice**: Read GEM_User_Guide.md Parameter Tuning Guide

---

**Happy backtesting!** 🚀
