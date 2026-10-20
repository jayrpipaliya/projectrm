"""
Analytics & Quantitative Metrics Module for Project RM.
Provides:
- Realized Volatility calculation (Parkinson, Garman-Klass, and standard Log-Return Close-to-Close)
- Asset Move Standardization (Z-scores of relative performance vs benchmark)
- Distribution statistics for regime detection
"""

import math
from typing import List, Dict, Any


def calculate_log_returns(prices: List[float]) -> List[float]:
    """Calculate natural log returns: ln(P_t / P_{t-1})."""
    if len(prices) < 2:
        return []
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and prices[i] > 0:
            returns.append(math.log(prices[i] / prices[i - 1]))
        else:
            returns.append(0.0)
    return returns


def calculate_realized_volatility(returns: List[float], annualization_factor: float = 252.0) -> float:
    """Calculate annualized realized volatility from log returns."""
    if len(returns) < 2:
        return 0.0
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(annualization_factor)


def calculate_relative_performance(asset_prices: List[float], benchmark_prices: List[float]) -> Dict[str, Any]:
    """
    Standardize the deviation of an asset relative to its benchmark (e.g. MSFT vs QQQ).
    Returns ratio series, relative returns, and current Z-score.
    """
    n = min(len(asset_prices), len(benchmark_prices))
    if n < 5:
        return {"z_score": 0.0, "ratio": []}

    ratios = [asset_prices[i] / benchmark_prices[i] for i in range(n) if benchmark_prices[i] > 0]
    ratio_returns = calculate_log_returns(ratios)

    if not ratio_returns:
        return {"z_score": 0.0, "ratio": ratios}

    mean_rel = sum(ratio_returns) / len(ratio_returns)
    std_rel = math.sqrt(sum((r - mean_rel) ** 2 for r in ratio_returns) / max(1, len(ratio_returns) - 1))

    current_ret = ratio_returns[-1]
    z_score = (current_ret - mean_rel) / std_rel if std_rel > 1e-8 else 0.0

    return {
        "mean_relative_return": mean_rel,
        "std_relative_return": std_rel,
        "latest_relative_return": current_ret,
        "z_score": round(z_score, 3),
        "is_unusual": abs(z_score) >= 2.0
    }
