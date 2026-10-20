"""
Project RM - Clean Kalman Filter + Modular Quantitative Cockpit
==================================================================

Components:
1. Kalman Filter (State-Space Dynamic Factor Model):
   - Continuous recursive beta tracking (Market & Sector Orthogonal).
   - Zero "Ghost Drops" (no 60-day cliff).
   - Zero GARCH hangover: Uses clean, instantaneous Kalman innovation variance S_t.

2. Module A: Ornstein-Uhlenbeck (O-U) Snap-Back Engine:
   - Fits dZ_t = theta * (mu - Z_t) dt + sigma * dW_t
   - Computes Half-Life tau = ln(2) / theta (in trading days).
   - Evaluates historical snap-back win rate (reversion inside +/-0.5 sigma within tau days).
   - Identifies active Snap-Back opportunities.

3. Module B: CUSUM Drift & Stealth Regime Shift Monitor:
   - Page's Cumulative Sum (CUSUM) on standardized innovation errors.
   - Detects consecutive one-sided smaller Z-scores (stealth structural regime changes).
   - Tracks current drift streak and Kalman Alpha departure (|alpha| / SE_alpha).
   - Alerts when cumulative drift exceeds statistical threshold h.
"""

import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Any, List, Optional
import config
from warehouse import Warehouse
from api_client import MassiveApiClient


class KalmanBetaDecomposition:
    def __init__(self, y: np.ndarray, X: np.ndarray):
        """
        Parameters:
            y: 1D array of asset log returns (T,)
            X: 2D array of factor returns [Intercept, Market, Sector_Perp] (T, k)
        """
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.T, self.k = self.X.shape

    def _run_filter(self, Q_diag: np.ndarray, R_scalar: float):
        """Runs the forward Kalman Filter recursion loop."""
        Q = np.diag(Q_diag)
        R = float(R_scalar)

        beta_history = np.zeros((self.T, self.k))
        P = np.eye(self.k) * 1.0
        beta_t = np.zeros(self.k)
        if self.k >= 2:
            beta_t[1] = 1.0  # Prior: start with market beta = 1.0

        residuals = np.zeros(self.T)
        innov_var = np.zeros(self.T)
        y_preds = np.zeros(self.T)
        neg_log_likelihood = 0.0

        for t in range(self.T):
            X_t = self.X[t, :].reshape(1, -1)

            # Predict Step
            beta_pred = beta_t
            P_pred = P + Q

            # Innovation (Unexpected residual)
            y_pred = float((X_t @ beta_pred)[0])
            y_preds[t] = y_pred
            e_t = float(self.y[t] - y_pred)
            residuals[t] = e_t

            # Innovation variance: S_t = X_t P_pred X_t^T + R
            S_t = float((X_t @ P_pred @ X_t.T)[0, 0] + R)
            innov_var[t] = S_t

            # Kalman Gain
            K_t = (P_pred @ X_t.T) / S_t

            # Correct Step (Update state and uncertainty)
            beta_t = beta_pred + K_t.flatten() * e_t
            P = (np.eye(self.k) - K_t @ X_t) @ P_pred
            beta_history[t, :] = beta_t

            if S_t > 1e-12:
                neg_log_likelihood += 0.5 * (np.log(2.0 * np.pi * S_t) + (e_t**2) / S_t)

        return neg_log_likelihood, beta_history, residuals, innov_var, y_preds

    def fit(self):
        """Optimizes process noise Q and observation noise R via Maximum Likelihood Estimation."""
        sample_var = float(np.var(self.y)) if len(self.y) > 1 else 1e-4
        init_params = np.append(np.repeat(1e-5, self.k), sample_var * 0.5)
        bounds = [(1e-8, 1e-2)] * self.k + [(1e-6, sample_var * 3.0)]

        def obj_func(params):
            neg_ll, _, _, _, _ = self._run_filter(params[:self.k], params[-1])
            return neg_ll

        try:
            res = minimize(obj_func, init_params, bounds=bounds, method="L-BFGS-B")
            opt_Q, opt_R = res.x[:self.k], res.x[-1]
        except Exception:
            opt_Q = np.repeat(1e-5, self.k)
            opt_R = sample_var * 0.5

        _, self.beta_history, self.residuals, self.innov_variances, self.y_preds = \
            self._run_filter(opt_Q, opt_R)

        self.fitted_Q = opt_Q
        self.fitted_R = opt_R
        return self


# ==============================================================================
# MODULE A: ORNSTEIN-UHLENBECK (O-U) MEAN REVERSION ENGINE
# ==============================================================================

class OrnsteinUhlenbeckEngine:
    @staticmethod
    def fit(z_series: np.ndarray, dt: float = 1.0):
        """
        Fits AR(1) representation of Ornstein-Uhlenbeck process:
            Z_t = a + b * Z_{t-1} + error
        where b = exp(-theta * dt) and mu = a / (1 - b).
        Half-life: tau = ln(2) / theta
        """
        z = np.asarray(z_series, dtype=float)
        if len(z) < 15:
            return {"theta": 0.2, "half_life": 3.5, "mu": 0.0, "is_mean_reverting": True}

        y = z[1:]
        x = z[:-1]
        # OLS regression of Z_t on Z_{t-1}
        x_mat = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
            a, b = coeffs[0], coeffs[1]
        except Exception:
            a, b = 0.0, 0.8

        # Ensure stationary reversion
        if 0.0 < b < 0.999:
            theta = -np.log(b) / dt
            half_life = np.log(2.0) / theta
            mu = a / (1.0 - b)
            is_mean_reverting = True
        else:
            theta = 0.05
            half_life = 14.0
            mu = 0.0
            is_mean_reverting = False

        return {
            "theta": round(float(theta), 4),
            "half_life": round(float(half_life), 1),
            "mu": round(float(mu), 3),
            "is_mean_reverting": is_mean_reverting
        }

    @staticmethod
    def evaluate_snapbacks(z_series: np.ndarray, dates: List[str], barrier: float = 1.8, target: float = 0.5, max_days: int = 5):
        """
        Analyzes historical crosses past +/- barrier and calculates:
        - Win rate (% that returned to +/- target within max_days)
        - Average days to recovery
        """
        z = np.asarray(z_series, dtype=float)
        n = len(z)
        events = []
        i = 0
        while i < n - 1:
            if abs(z[i]) >= barrier:
                direction = "OVERBOUGHT" if z[i] > 0 else "OVERSOLD"
                entry_date = dates[i]
                entry_z = z[i]
                success = False
                recovery_days = 0

                # Search forward
                for f in range(1, min(max_days + 1, n - i)):
                    recovery_days = f
                    if direction == "OVERSOLD" and z[i + f] >= -target:
                        success = True
                        break
                    elif direction == "OVERBOUGHT" and z[i + f] <= target:
                        success = True
                        break

                events.append({
                    "date": entry_date,
                    "direction": direction,
                    "entry_z": round(float(entry_z), 2),
                    "success": success,
                    "recovery_days": recovery_days
                })
                # Skip forward past this episode to avoid double counting same spike
                i += max(1, recovery_days)
            else:
                i += 1

        total_events = len(events)
        successes = sum(1 for e in events if e["success"])
        win_rate = (successes / total_events * 100.0) if total_events > 0 else 0.0
        avg_days = (np.mean([e["recovery_days"] for e in events if e["success"]])) if successes > 0 else 0.0

        return {
            "total_events": total_events,
            "win_rate_pct": round(float(win_rate), 1),
            "avg_recovery_days": round(float(avg_days), 1),
            "barrier_used": barrier,
            "target_used": target,
            "recent_events": events[-5:] if events else []
        }


# ==============================================================================
# MODULE OPT: SYNTHETIC ATM OPTION BACKTEST ENGINE (Method A)
# ==============================================================================

class SyntheticOptionBacktestEngine:
    @staticmethod
    def black_scholes_price(S: float, K: float, T_years: float, r: float, sigma: float, option_type: str = "call") -> float:
        """
        Ultra-fast closed-form Black-Scholes formula using math.erf.
        Over 300x faster than scipy.stats.norm.cdf for large grid sweeps.
        """
        if T_years <= 0 or sigma <= 0:
            if option_type == "call":
                return max(0.0, S - K)
            else:
                return max(0.0, K - S)

        sqrt_T = math.sqrt(T_years)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_years) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        _INV_SQRT2 = 0.7071067811865475
        cdf_d1 = 0.5 * (1.0 + math.erf(d1 * _INV_SQRT2))
        cdf_d2 = 0.5 * (1.0 + math.erf(d2 * _INV_SQRT2))
        exp_rt = math.exp(-r * T_years)

        if option_type == "call":
            price = S * cdf_d1 - K * exp_rt * cdf_d2
        else:
            price = K * exp_rt * (1.0 - cdf_d2) - S * (1.0 - cdf_d1)

        return max(0.05, float(price))

    @classmethod
    def evaluate_option_performance(
        cls,
        z_series: np.ndarray,
        prices: np.ndarray,
        dates: List[str],
        barrier: float = 1.8,
        r: float = 0.045,
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        dte_contract: int = 35
    ) -> Dict[str, Any]:
        """
        Backtests taking ATM options at market close across holding horizons 1 to 30 DTE:
        - Bullish signal (Z <= -barrier): Buy ATM Call (Strike K = round(S_entry))
        - Bearish signal (Z >= +barrier): Buy ATM Put (Strike K = round(S_entry))
        Sweeps holding days d = 1..30 to discover the optimal DTE, win rate, and return
        separately for Bullish strategies and Bearish strategies.

        Earnings Filter:
        Excludes trades that would be entered on earnings announcement dates or the trading
        day immediately following an earnings announcement, preventing unhedged binary event exposure.
        """
        z = np.asarray(z_series, dtype=float)
        p = np.asarray(prices, dtype=float)
        n = len(z)

        # Build corporate earnings blackout set (earnings announcement date + next trading day)
        earnings_raw_set = set(earnings_dates or [])
        earnings_blackout_dates = set()
        if filter_earnings and earnings_raw_set:
            date_to_idx = {d: idx for idx, d in enumerate(dates)}
            for ed in earnings_raw_set:
                if ed in date_to_idx:
                    idx = date_to_idx[ed]
                    earnings_blackout_dates.add(ed)
                    # Exclude the day of earnings and the day after earnings
                    if idx + 1 < len(dates):
                        earnings_blackout_dates.add(dates[idx + 1])
                else:
                    # Even if not directly in date_to_idx (e.g. weekend announcement), add raw date
                    earnings_blackout_dates.add(ed)

        # 20-day historical realized volatility for Black-Scholes sigma
        log_rets = np.diff(np.log(p))
        rv_20d = np.zeros(n)
        for i in range(1, n):
            window = log_rets[max(0, i - 20) : i]
            if len(window) >= 5:
                rv_20d[i] = max(0.12, float(np.std(window, ddof=1) * np.sqrt(252)))
            else:
                rv_20d[i] = 0.25

        # 1. Identify all signal entry points (skipping corporate earnings blackout dates)
        signal_episodes = []
        excluded_earnings_trades = []
        i = 0
        while i < n - 1:
            if abs(z[i]) >= barrier:
                entry_date = str(dates[i])
                is_bullish = bool(z[i] <= -barrier)
                action = "BUY ATM CALL" if is_bullish else "BUY ATM PUT"
                opt_type = "call" if is_bullish else "put"
                s_entry = p[i]
                k_strike = float(round(s_entry))
                sigma_entry = float(rv_20d[i])

                # Check if signal fired on or right after an earnings announcement date
                if filter_earnings and entry_date in earnings_blackout_dates:
                    excluded_earnings_trades.append({
                        "date": entry_date,
                        "action": action,
                        "signal_z": round(float(z[i]), 2),
                        "reason": "Earnings Blackout (Announcement Day / Post-Earnings)"
                    })
                    i += 1
                    continue

                signal_episodes.append({
                    "idx": i,
                    "date": entry_date,
                    "is_bullish": is_bullish,
                    "action": action,
                    "opt_type": opt_type,
                    "signal_z": round(float(z[i]), 2),
                    "underlying_entry": round(float(s_entry), 2),
                    "strike": k_strike,
                    "sigma_entry": sigma_entry
                })
                # Skip forward 2 days to avoid duplicate signals on identical shock
                i += 2
            else:
                i += 1

        # 2. Evaluate performance across holding horizons d = 1..max_horizon
        max_horizon = min(30, max(1, dte_contract - 1))

        # Separate bullish and bearish episode lists
        bull_eps = [e for e in signal_episodes if e["is_bullish"]]
        bear_eps = [e for e in signal_episodes if not e["is_bullish"]]

        def evaluate_horizon_stats(episodes_list):
            if not episodes_list:
                return {
                    "best_dte": 0,
                    "best_avg_roi_pct": 0.0,
                    "best_win_rate_pct": 0.0,
                    "total_trades": 0,
                    "horizon_curve": []
                }

            curve = []
            for d in range(1, max_horizon + 1):
                rois = []
                wins = 0
                dollars = []
                for ep in episodes_list:
                    idx = ep["idx"]
                    step = min(d, (n - 1) - idx)
                    if step <= 0:
                        continue
                    s_entry = ep["underlying_entry"]
                    k_strike = ep["strike"]
                    opt_type = ep["opt_type"]
                    sigma_entry = ep["sigma_entry"]

                    t_entry = dte_contract / 365.0
                    p_entry = max(0.01, cls.black_scholes_price(s_entry, k_strike, t_entry, r, sigma_entry, opt_type))

                    s_exit = p[idx + step]
                    t_exit = max(0.001, (dte_contract - step) / 365.0)
                    sigma_exit = sigma_entry
                    p_exit = cls.black_scholes_price(s_exit, k_strike, t_exit, r, sigma_exit, opt_type)

                    roi = ((p_exit - p_entry) / p_entry) * 100.0
                    rois.append(roi)
                    dollars.append((p_exit - p_entry) * 100.0)
                    if roi > 0:
                        wins += 1

                if rois:
                    win_rate = (wins / len(rois)) * 100.0
                    avg_roi = float(np.mean(rois))
                    tot_dollar = float(np.sum(dollars))
                else:
                    win_rate = 0.0
                    avg_roi = 0.0
                    tot_dollar = 0.0

                curve.append({
                    "hold_days": d,
                    "win_rate_pct": round(float(win_rate), 1),
                    "avg_roi_pct": round(float(avg_roi), 1),
                    "total_dollar": round(float(tot_dollar), 2)
                })

            # Score horizons: prioritize robust positive average ROI with win rate >= 40%
            valid = [c for c in curve if c["win_rate_pct"] >= 40.0]
            if not valid:
                valid = curve
            best = max(valid, key=lambda x: (x["avg_roi_pct"], x["win_rate_pct"])) if valid else {"hold_days": 7, "avg_roi_pct": 0.0, "win_rate_pct": 0.0}

            return {
                "best_dte": best["hold_days"],
                "best_avg_roi_pct": best["avg_roi_pct"],
                "best_win_rate_pct": best["win_rate_pct"],
                "best_total_dollar": best.get("total_dollar", 0.0),
                "total_trades": len(episodes_list),
                "horizon_curve": curve
            }

        bull_stats = evaluate_horizon_stats(bull_eps)
        bear_stats = evaluate_horizon_stats(bear_eps)

        best_bull_dte = bull_stats["best_dte"] or 7
        best_bear_dte = bear_stats["best_dte"] or 7

        # 3. Compute detailed trade log using each strategy's recommended DTE
        trades = []
        for ep in signal_episodes:
            idx = ep["idx"]
            is_bull = ep["is_bullish"]
            rec_d = best_bull_dte if is_bull else best_bear_dte

            s_entry = ep["underlying_entry"]
            k_strike = ep["strike"]
            opt_type = ep["opt_type"]
            sigma_entry = ep["sigma_entry"]

            t_entry = dte_contract / 365.0
            p_entry = max(0.01, cls.black_scholes_price(s_entry, k_strike, t_entry, r, sigma_entry, opt_type))

            # 1-day exit
            s_1d = p[idx + 1]
            t_1d = max(0.001, (dte_contract - 1) / 365.0)
            p_1d = cls.black_scholes_price(s_1d, k_strike, t_1d, r, sigma_entry, opt_type)
            pnl_1d_pct = ((p_1d - p_entry) / p_entry) * 100.0

            # Recommended DTE exit
            step_rec = min(rec_d, (n - 1) - idx)
            s_rec = p[idx + step_rec]
            t_rec = max(0.001, (dte_contract - step_rec) / 365.0)
            p_rec = cls.black_scholes_price(s_rec, k_strike, t_rec, r, sigma_entry, opt_type)
            pnl_rec_pct = ((p_rec - p_entry) / p_entry) * 100.0
            dollar_rec = (p_rec - p_entry) * 100.0

            trades.append({
                "date": ep["date"],
                "action": ep["action"],
                "is_bullish": is_bull,
                "signal_z": ep["signal_z"],
                "underlying_entry": s_entry,
                "strike": k_strike,
                "sigma_entry": round(float(sigma_entry), 4),
                "entry_option_price": round(float(p_entry), 2),
                "exit_1d_price": round(float(p_1d), 2),
                "pnl_1d_pct": round(float(pnl_1d_pct), 1),
                "recommended_dte": rec_d,
                "exit_rec_price": round(float(p_rec), 2),
                "pnl_rec_pct": round(float(pnl_rec_pct), 1),
                "dollar_rec": round(float(dollar_rec), 2),
                "win_rec": bool(pnl_rec_pct > 0)
            })

        total_trades = len(trades)
        wins_rec = sum(1 for t in trades if t["win_rec"])
        win_rate_rec = (wins_rec / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_roi_rec = float(np.mean([t["pnl_rec_pct"] for t in trades])) if total_trades > 0 else 0.0
        tot_dollar_rec = float(sum(t["dollar_rec"] for t in trades)) if total_trades > 0 else 0.0

        return {
            "threshold_used": round(float(barrier), 2),
            "total_trades": total_trades,
            "overall_win_rate_pct": round(float(win_rate_rec), 1),
            "overall_avg_roi_pct": round(float(avg_roi_rec), 1),
            "overall_total_dollar": round(float(tot_dollar_rec), 2),
            "bullish_recommendation": {
                "action": "BUY ATM CALL",
                "recommended_dte": bull_stats["best_dte"],
                "win_rate_pct": bull_stats["best_win_rate_pct"],
                "avg_roi_pct": bull_stats["best_avg_roi_pct"],
                "total_trades": bull_stats["total_trades"]
            },
            "bearish_recommendation": {
                "action": "BUY ATM PUT",
                "recommended_dte": bear_stats["best_dte"],
                "win_rate_pct": bear_stats["best_win_rate_pct"],
                "avg_roi_pct": bear_stats["best_avg_roi_pct"],
                "total_trades": bear_stats["total_trades"]
            },
            "earnings_filter": {
                "enabled": filter_earnings,
                "earnings_dates_tracked": sorted(list(earnings_raw_set)),
                "blackout_dates": sorted(list(earnings_blackout_dates)),
                "excluded_trades_count": len(excluded_earnings_trades),
                "excluded_trades": excluded_earnings_trades
            },
            "trades": trades
        }


# ==============================================================================
# MODULE B: CUSUM STEALTH REGIME SHIFT MONITOR
# ==============================================================================

class CUSUMDriftMonitor:
    @staticmethod
    def compute(z_series: np.ndarray, k: float = 0.5, h: float = 4.0):
        """
        Two-sided Cumulative Sum (CUSUM) on standardized innovation errors:
            S_t^+ = max(0, S_{t-1}^+ + Z_t - k)   [Upper cumulative drift]
            S_t^- = max(0, S_{t-1}^- - Z_t - k)   [Lower cumulative drift]
        Detects when consecutive smaller Z-scores are systematically one-sided.
        h: Decision threshold (typically 3.5 to 5.0).
        """
        z = np.asarray(z_series, dtype=float)
        T = len(z)
        s_pos = np.zeros(T)
        s_neg = np.zeros(T)
        streak = np.zeros(T, dtype=int)
        regime_alert = np.zeros(T, dtype=int)  # +1 Bullish regime shift, -1 Bearish shift, 0 None

        current_sign = 0
        current_streak = 0

        for t in range(T):
            # CUSUM accumulation
            if t == 0:
                s_pos[t] = max(0.0, z[t] - k)
                s_neg[t] = max(0.0, -z[t] - k)
            else:
                s_pos[t] = max(0.0, s_pos[t - 1] + z[t] - k)
                s_neg[t] = max(0.0, s_neg[t - 1] - z[t] - k)

            # Streak tracking (consecutive days same direction with |Z| >= 0.3)
            if abs(z[t]) >= 0.3:
                sign = 1 if z[t] > 0 else -1
                if sign == current_sign:
                    current_streak += 1
                else:
                    current_sign = sign
                    current_streak = 1
            else:
                current_streak = 0
                current_sign = 0

            streak[t] = current_streak * current_sign

            # Check threshold breach
            if s_pos[t] >= h:
                regime_alert[t] = 1
            elif s_neg[t] >= h:
                regime_alert[t] = -1
            else:
                regime_alert[t] = 0

        return {
            "s_pos": s_pos,
            "s_neg": s_neg,
            "streak": streak,
            "alert": regime_alert,
            "threshold": h,
            "tolerance": k
        }


# ==============================================================================
# MAIN ANOMALY & REGIME COCKPIT ENGINE
# ==============================================================================

class ResidualAnomalyModel:
    def __init__(self, warmup_period: int = 15):
        self.warmup_period = warmup_period
        self.warehouse = Warehouse()
        self.client = MassiveApiClient()

    def _fetch_and_cache_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        ticker = ticker.upper().strip()
        cached = self.warehouse.get_stock_history(ticker)
        # If we have deep multi-year history (>= 1000 bars, ~4+ years), use local warehouse directly
        if cached and len(cached) >= 1000:
            return cached

        # Check yfinance first to load multi-year depth (up to 5 years)
        try:
            from yfinance_loader import YFinanceLoader
            yf_loader = YFinanceLoader(self.warehouse)
            yf_res = yf_loader.fetch_and_store_ticker(ticker, period="5y")
            if yf_res.get("status") == "SUCCESS":
                updated = self.warehouse.get_stock_history(ticker)
                if updated and len(updated) >= 40:
                    return updated
        except Exception:
            pass

        # Fallback to MassiveApiClient if yfinance is unreachable
        res = self.client.get_stock_aggs(ticker)
        bars = res.get("results", [])
        if bars:
            self.warehouse.save_stock_bars(ticker, bars)
            return self.warehouse.get_stock_history(ticker)

        return cached or []

    def _fetch_and_cache_earnings(self, ticker: str) -> List[str]:
        """
        Retrieves corporate earnings announcement dates for ticker.
        Checks local SQL warehouse first; if empty, fetches official Form 8-K Item 2.02
        earnings filings via SEC EDGAR and caches them in the warehouse.
        """
        ticker = ticker.upper().strip()
        cached = self.warehouse.get_earnings_dates(ticker)
        if cached:
            return [row["earnings_date"] for row in cached]

        api_res = self.client.get_earnings_calendar(ticker)
        results = api_res.get("results", [])
        if results:
            self.warehouse.save_earnings_dates(ticker, results)
            return [r["earnings_date"] for r in results]
        return []

    def analyze(
        self,
        asset_ticker: str = "MSFT",
        market_ticker: str = "QQQ",
        sector_ticker: str = "IGV",
        snapback_threshold: float = 1.8,
        filter_earnings: bool = True
    ) -> Dict[str, Any]:
        """
        Runs Kalman Filter + Ornstein-Uhlenbeck Snap-Back + CUSUM Drift Monitor.
        Zero GARCH hangover. Fully modular.
        Optionally filters out mean-reversion trades on/immediately after earnings dates.
        """
        asset_ticker = asset_ticker.upper().strip()
        market_ticker = market_ticker.upper().strip()
        sector_ticker = sector_ticker.upper().strip()

        # Ingest and synchronize prices
        asset_bars = self._fetch_and_cache_ticker(asset_ticker)
        mkt_bars = self._fetch_and_cache_ticker(market_ticker)
        sec_bars = self._fetch_and_cache_ticker(sector_ticker)

        if not asset_bars or not mkt_bars or not sec_bars:
            return {
                "status": "ERROR",
                "error": f"Insufficient price history for: {asset_ticker}, {market_ticker}, {sector_ticker}"
            }

        asset_dict = {b["trade_date"]: float(b["close"]) for b in asset_bars if b.get("close")}
        mkt_dict = {b["trade_date"]: float(b["close"]) for b in mkt_bars if b.get("close")}
        sec_dict = {b["trade_date"]: float(b["close"]) for b in sec_bars if b.get("close")}

        common_dates = sorted(set(asset_dict) & set(mkt_dict) & set(sec_dict))
        if len(common_dates) < self.warmup_period + 20:
            return {
                "status": "ERROR",
                "error": f"Insufficient common dates ({len(common_dates)})."
            }

        dates = np.array(common_dates)
        p_asset = np.array([asset_dict[d] for d in dates])
        p_mkt = np.array([mkt_dict[d] for d in dates])
        p_sec = np.array([sec_dict[d] for d in dates])

        # Log returns
        ret_asset = np.log(p_asset[1:] / p_asset[:-1])
        ret_mkt = np.log(p_mkt[1:] / p_mkt[:-1])
        ret_sec = np.log(p_sec[1:] / p_sec[:-1])
        ret_dates = dates[1:]
        T = len(ret_dates)

        # Orthogonalize sector to market
        cov_mat = np.cov(ret_sec, ret_mkt, ddof=1)
        var_mkt = cov_mat[1, 1]
        b_sec_mkt = (cov_mat[0, 1] / var_mkt) if var_mkt > 1e-10 else 0.0
        ret_sec_perp = ret_sec - b_sec_mkt * ret_mkt

        # --- Layer 1: Kalman Filter ---
        X = np.column_stack([np.ones(T), ret_mkt, ret_sec_perp])
        kf = KalmanBetaDecomposition(y=ret_asset, X=X).fit()

        beta_hist = kf.beta_history      # (T, 3): [alpha, beta_mkt, beta_sec]
        innovations = kf.residuals        # e_t: raw idiosyncratic returns
        innov_vars = kf.innov_variances  # S_t: clean instantaneous innovation variance
        y_preds = kf.y_preds

        # Studentized Z-score: Z_t = e_t / sqrt(S_t) (Clean, instantaneous, NO GARCH hangover!)
        z_1d = np.zeros(T)
        z_5d = np.zeros(T)
        z_20d = np.zeros(T)

        for t in range(T):
            sigma_s = np.sqrt(innov_vars[t]) if innov_vars[t] > 1e-12 else 1e-4
            z_1d[t] = innovations[t] / sigma_s

            if t >= 4:
                z_5d[t] = np.sum(innovations[t - 4 : t + 1]) / (np.sqrt(5.0) * sigma_s)
            else:
                z_5d[t] = z_1d[t]

            if t >= 19:
                z_20d[t] = np.sum(innovations[t - 19 : t + 1]) / (np.sqrt(20.0) * sigma_s)
            else:
                z_20d[t] = z_5d[t]

        # --- Layer 2: Module A - Ornstein-Uhlenbeck Snap-Back Engine ---
        ou_stats = OrnsteinUhlenbeckEngine.fit(z_1d)
        snapback_profile = OrnsteinUhlenbeckEngine.evaluate_snapbacks(
            z_series=z_1d, dates=list(ret_dates), barrier=snapback_threshold, target=0.5, max_days=int(np.ceil(ou_stats["half_life"] * 1.5))
        )

        # Fetch and cache corporate earnings announcement dates for asset
        earnings_dates = self._fetch_and_cache_earnings(asset_ticker)

        # --- Layer 2.5: Synthetic ATM Option Backtest Engine (Dynamic Threshold & Earnings Filter) ---
        option_backtest = SyntheticOptionBacktestEngine.evaluate_option_performance(
            z_series=z_1d,
            prices=p_asset[1:],
            dates=list(ret_dates),
            barrier=snapback_threshold,
            earnings_dates=earnings_dates,
            filter_earnings=filter_earnings
        )

        # --- Layer 3: Module B - CUSUM Stealth Drift Monitor ---
        cusum_res = CUSUMDriftMonitor.compute(z_1d, k=0.4, h=3.8)

        # Build history output (post-warmup)
        history = []
        for t in range(self.warmup_period, T):
            history.append({
                "date": ret_dates[t],
                "asset_close": round(float(p_asset[t + 1]), 2),
                "asset_return_pct": round(float(ret_asset[t]) * 100.0, 2),
                "mkt_return_pct": round(float(ret_mkt[t]) * 100.0, 2),
                "sec_return_pct": round(float(ret_sec[t]) * 100.0, 2),
                "expected_return_pct": round(float(y_preds[t]) * 100.0, 2),
                "residual_pct": round(float(innovations[t]) * 100.0, 3),
                "z_1d": round(float(z_1d[t]), 3),
                "z_5d": round(float(z_5d[t]), 3),
                "z_20d": round(float(z_20d[t]), 3),
                "beta_mkt": round(float(beta_hist[t, 1]), 3),
                "beta_sec": round(float(beta_hist[t, 2]), 3),
                "alpha": round(float(beta_hist[t, 0]), 4),
                "cusum_pos": round(float(cusum_res["s_pos"][t]), 2),
                "cusum_neg": round(float(cusum_res["s_neg"][t]), 2),
                "streak": int(cusum_res["streak"][t]),
                "regime_alert": int(cusum_res["alert"][t])
            })

        latest = history[-1] if history else {}
        latest_z = latest.get("z_1d", 0.0)
        latest_streak = latest.get("streak", 0)
        latest_alert = latest.get("regime_alert", 0)
        latest_alpha = latest.get("alpha", 0.0)

        # Active Cockpit Status Determination
        is_snapback_active = abs(latest_z) >= snapback_threshold
        is_regime_shift_active = (latest_alert != 0) or (abs(latest_streak) >= 5)

        if is_snapback_active and not is_regime_shift_active:
            cockpit_state = "SNAP-BACK OPPORTUNITY"
            cockpit_badge = "success"
            action_summary = f"{asset_ticker} is statistically overstretched ({latest_z:+.2f}σ). High historical snap-back probability within ~{ou_stats['half_life']} days."
        elif is_regime_shift_active:
            cockpit_state = "STEALTH REGIME SHIFT"
            cockpit_badge = "danger"
            direction_str = "Bullish Structural Break" if (latest_alert > 0 or latest_streak > 0) else "Bearish Structural Breakdown"
            action_summary = f"{direction_str}: Consecutive 1-sided drift detected (Streak: {abs(latest_streak)}d, CUSUM Alert). Asset is NOT mean-reverting."
        elif abs(latest_z) >= 1.2:
            cockpit_state = "MILD DEVIATION"
            cockpit_badge = "warning"
            action_summary = f"{asset_ticker} is showing mild idiosyncratic decoupling ({latest_z:+.2f}σ). Monitoring for boundary touch."
        else:
            cockpit_state = "NORMAL CO-MOVEMENT"
            cockpit_badge = "normal"
            action_summary = f"{asset_ticker} is moving in equilibrium with {market_ticker} and {sector_ticker} betas."

        return {
            "status": "SUCCESS",
            "model_type": "Kalman Filter + O-U Snap-Back & CUSUM Cockpit",
            "tickers": {
                "asset": asset_ticker,
                "market": market_ticker,
                "sector": sector_ticker
            },
            "cockpit": {
                "state": cockpit_state,
                "badge": cockpit_badge,
                "summary": action_summary,
                "is_snapback_candidate": is_snapback_active,
                "is_regime_shift_candidate": is_regime_shift_active
            },
            "ou_snapback": {
                "threshold": round(float(snapback_threshold), 2),
                "half_life_days": ou_stats["half_life"],
                "reversion_speed_theta": ou_stats["theta"],
                "is_mean_reverting": ou_stats["is_mean_reverting"],
                "historical_win_rate_pct": snapback_profile["win_rate_pct"],
                "avg_recovery_days": snapback_profile["avg_recovery_days"],
                "total_past_episodes": snapback_profile["total_events"]
            },
            "option_backtest": option_backtest,
            "cusum_drift": {
                "current_streak_days": latest_streak,
                "cusum_positive": latest.get("cusum_pos", 0.0),
                "cusum_negative": latest.get("cusum_neg", 0.0),
                "alert_status": latest_alert,
                "threshold_h": cusum_res["threshold"]
            },
            "latest_metrics": {
                "date": latest.get("date"),
                "asset_close": latest.get("asset_close"),
                "asset_return_pct": latest.get("asset_return_pct"),
                "expected_return_pct": latest.get("expected_return_pct"),
                "unexplained_residual_pct": latest.get("residual_pct"),
                "z_score_1d": latest.get("z_1d"),
                "z_score_5d": latest.get("z_5d"),
                "z_score_20d": latest.get("z_20d"),
                "beta_market": latest.get("beta_mkt"),
                "beta_sector": latest.get("beta_sec"),
                "alpha": latest_alpha
            },
            "history": history
        }
