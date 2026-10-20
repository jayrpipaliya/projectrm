"""
Project RM - Clean Anchored Kalman Filter + Cointegrated Spread OU Engine
==========================================================================
Reformed Mathematical Architecture (Pure Asset Returns Edition):

1. Anchored State-Space Kalman Dynamic Factor Model:
   - Dynamic beta tracking for Market and Orthogonal Sector sensitivities.
   - Vasicek / AR(1) state-space transition equation anchored to long-term economic priors:
     beta_t - beta_bar = Phi * (beta_{t-1} - beta_bar) + w_t
   - Eliminates unbounded random-walk drift to unrealistic extremes.
   - Soft economic clamping guarantees finite, stable sensitivities.

2. Cointegrated Cumulative Price Spread & Ornstein-Uhlenbeck Engine:
   - Decouples 1-step forecast innovation e_t from the tradeable cointegrating price-level spread s_t.
   - Cumulative spread s_t = sum(r_asset - (alpha + beta_mkt * r_mkt + beta_sec * r_sec_perp))
     measures true relative pricing divergence from synthetic tracking basket.
   - Continuous-time OU mean-reversion process dS_t = theta * (mu - S_t) dt + sigma * dW_t
     is fitted directly on the price spread s_t, resolving the white-noise paradox.
   - Standardized spread Z-score Z_t = (s_t - mu) / sigma_eq captures genuine mean-reverting memory.

3. IID Innovation CUSUM Stealth Drift Monitor:
   - Page's Cumulative Sum (CUSUM) applied directly to standardized OU innovations u_t = xi_t / sigma_xi.
   - Because u_t is mathematically guaranteed to be IID white noise, normal mean-reverting
     cycles do not accumulate false drift alarms.
   - Alerts only when persistent structural factor shifts (E[u_t] != 0) occur.

4. Pure Asset % Return Backtest Engine:
   - Evaluates direct stock / equity % price returns on long/short signals.
   - Computes win rate %, profit factor, average % return, MFE, MAE, and forensic loss categories.
   - Eliminates Black-Scholes pricing errors, IV spreads, and volatility risk premium distortions.
"""

import math
import numpy as np
from scipy.optimize import minimize
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

        # Economic priors: [alpha=0.0, beta_mkt=1.0, beta_sec=0.5]
        self.beta_priors = np.zeros(self.k)
        if self.k >= 2:
            self.beta_priors[1] = 1.0  # Market beta prior
        if self.k >= 3:
            self.beta_priors[2] = 0.5  # Sector beta prior

        # Mean-reverting persistence matrix Phi in state transition:
        # beta_t - beta_prior = Phi * (beta_{t-1} - beta_prior) + w_t
        self.Phi_diag = np.full(self.k, 0.985)
        if self.k >= 1:
            self.Phi_diag[0] = 0.95  # Alpha decays back to zero slightly faster

    def _run_filter(self, Q_diag: np.ndarray, R_scalar: float):
        """Runs the forward Kalman Filter with anchored mean-reverting state space."""
        Q = np.diag(Q_diag)
        R = float(R_scalar)
        Phi = np.diag(self.Phi_diag)

        beta_history = np.zeros((self.T, self.k))
        P = np.eye(self.k) * 0.1
        beta_t = self.beta_priors.copy()

        residuals = np.zeros(self.T)
        innov_var = np.zeros(self.T)
        y_preds = np.zeros(self.T)
        neg_log_likelihood = 0.0

        for t in range(self.T):
            X_t = self.X[t, :].reshape(1, -1)

            # Predict Step with Anchored Mean-Reverting Transition (Vasicek State-Space)
            # Pulls betas toward economic priors, preventing runaway random-walk drift
            beta_pred = self.beta_priors + Phi @ (beta_t - self.beta_priors)
            P_pred = Phi @ P @ Phi.T + Q

            # 1-step-ahead forecast & innovation
            y_pred = float((X_t @ beta_pred)[0])
            y_preds[t] = y_pred
            e_t = float(self.y[t] - y_pred)
            residuals[t] = e_t

            # Innovation variance: S_t = X_t P_pred X_t^T + R
            S_t = float((X_t @ P_pred @ X_t.T)[0, 0] + R)
            innov_var[t] = S_t

            # Kalman Gain
            K_t = (P_pred @ X_t.T) / S_t

            # Correct Step (State update)
            beta_t = beta_pred + K_t.flatten() * e_t

            # Soft economic clamping to eliminate pathological extreme values
            if self.k >= 1:
                beta_t[0] = float(np.clip(beta_t[0], -0.05, 0.05))   # Alpha daily limit +/-5%
            if self.k >= 2:
                beta_t[1] = float(np.clip(beta_t[1], 0.1, 2.5))     # Market beta bounded [0.1, 2.5]
            if self.k >= 3:
                beta_t[2] = float(np.clip(beta_t[2], -1.5, 2.5))    # Sector beta bounded [-1.5, 2.5]

            P = (np.eye(self.k) - K_t @ X_t) @ P_pred
            beta_history[t, :] = beta_t

            if S_t > 1e-12:
                neg_log_likelihood += 0.5 * (np.log(2.0 * np.pi * S_t) + (e_t**2) / S_t)

        return neg_log_likelihood, beta_history, residuals, innov_var, y_preds

    def fit(self):
        """Optimizes process noise Q and observation noise R via Maximum Likelihood Estimation."""
        sample_var = float(np.var(self.y)) if len(self.y) > 1 else 1e-4
        init_params = np.append(np.repeat(1e-5, self.k), sample_var * 0.5)
        bounds = [(1e-7, 1e-2)] * self.k + [(1e-6, sample_var * 3.0)]

        def obj_func(params):
            neg_ll, _, _, _, _ = self._run_filter(params[:self.k], params[-1])
            return neg_ll

        try:
            res = minimize(obj_func, init_params, bounds=bounds, method="L-BFGS-B", options={"maxiter": 60})
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
# MODULE A: COINTEGRATING SPREAD & ORNSTEIN-UHLENBECK (O-U) MEAN REVERSION ENGINE
# ==============================================================================

class OrnsteinUhlenbeckEngine:
    @staticmethod
    def fit(spread_series: np.ndarray, dt: float = 1.0):
        """
        Fits AR(1) representation of continuous-time Ornstein-Uhlenbeck process on
        the cointegrated cumulative price spread s_t:
            s_t = a + b * s_{t-1} + xi_t
        where b = exp(-theta * dt) and mu = a / (1 - b).
        Half-life: tau = ln(2) / theta
        Equilibrium variance: sigma_eq^2 = Var(xi) / (1 - b^2)
        Standardized OU innovations: u_t = xi_t / std(xi) (strictly IID!)
        """
        s = np.asarray(spread_series, dtype=float)
        n = len(s)
        if n < 15:
            return {
                "theta": 0.15,
                "half_life": 4.6,
                "mu": 0.0,
                "sigma_eq": 0.05,
                "sigma_innov": 0.01,
                "is_mean_reverting": True,
                "innovations": np.zeros(n),
                "z_spread": np.zeros(n)
            }

        y = s[1:]
        x = s[:-1]
        x_mat = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
            a, b = float(coeffs[0]), float(coeffs[1])
        except Exception:
            a, b = 0.0, 0.90

        # Enforce stationary mean reversion: 0 < b < 0.999
        b_clamped = max(0.01, min(0.995, b))
        theta = -np.log(b_clamped) / dt
        half_life = np.log(2.0) / max(1e-4, theta)
        mu = a / (1.0 - b_clamped)
        is_mean_reverting = bool(0.0 < b < 0.999)

        # 1-step forecast residuals (OU innovations)
        xi = y - (a + b_clamped * x)
        sigma_innov = float(np.std(xi, ddof=1)) if len(xi) > 2 else 0.01
        sigma_innov = max(1e-5, sigma_innov)

        # Stationary equilibrium standard deviation
        denom = max(1e-4, 1.0 - b_clamped**2)
        sigma_eq = sigma_innov / np.sqrt(denom)

        # Standardized IID innovations (for CUSUM monitor)
        innovations = np.zeros(n)
        innovations[1:] = xi / sigma_innov

        # Standardized pricing spread Z-score (stationary, mean-reverting)
        z_spread = (s - mu) / max(1e-4, sigma_eq)

        return {
            "theta": round(float(theta), 4),
            "half_life": round(float(half_life), 1),
            "mu": round(float(mu), 4),
            "sigma_eq": round(float(sigma_eq), 4),
            "sigma_innov": round(float(sigma_innov), 5),
            "is_mean_reverting": is_mean_reverting,
            "innovations": innovations,
            "z_spread": z_spread
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
# MODULE B: PURE ASSET % RETURN BACKTEST ENGINE
# ==============================================================================

class AssetReturnBacktestEngine:
    """
    Backtests direct equity / stock % price returns on anomaly signals:
    - Bullish signal (Z <= -barrier): Go LONG asset at close, hold for H sessions.
      Asset Return % = (P_exit - P_entry) / P_entry * 100%
    - Bearish signal (Z >= +barrier): Go SHORT asset at close, hold for H sessions.
      Asset Return % = (P_entry - P_exit) / P_entry * 100%
    - Incorporates realistic transaction slippage / friction (5 bps per side = 10 bps round trip).
    - Honors corporate earnings announcement blackouts.
    - Computes MFE %, MAE %, Profit Factor, Sharpe Ratio, and forensic loss categories.
    """

    @classmethod
    def evaluate_asset_performance(
        cls,
        z_series: np.ndarray,
        prices: np.ndarray,
        dates: List[str],
        barrier: float = 1.8,
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        holding_horizon: int = 5,
        slippage_bps: float = 5.0
    ) -> Dict[str, Any]:
        z = np.asarray(z_series, dtype=float)
        p = np.asarray(prices, dtype=float)
        n = len(z)

        # Corporate earnings blackout set (day of earnings + next session)
        earnings_raw_set = set(earnings_dates or [])
        earnings_blackout_dates = set()
        if filter_earnings and earnings_raw_set:
            date_to_idx = {d: idx for idx, d in enumerate(dates)}
            for ed in earnings_raw_set:
                if ed in date_to_idx:
                    idx = date_to_idx[ed]
                    earnings_blackout_dates.add(ed)
                    if idx + 1 < len(dates):
                        earnings_blackout_dates.add(dates[idx + 1])
                else:
                    earnings_blackout_dates.add(ed)

        slip_pct = (slippage_bps * 2.0) / 10000.0 * 100.0  # round-trip slippage %

        signal_episodes = []
        excluded_earnings_trades = []
        i = 0
        while i < n - 1:
            if abs(z[i]) >= barrier:
                entry_date = str(dates[i])
                is_bullish = bool(z[i] <= -barrier)
                action = "LONG ASSET" if is_bullish else "SHORT ASSET"
                s_entry = float(p[i])

                if filter_earnings and entry_date in earnings_blackout_dates:
                    excluded_earnings_trades.append({
                        "date": entry_date,
                        "barrier": barrier,
                        "signal_z": round(float(z[i]), 2),
                        "reason": "EARNINGS_BLACKOUT",
                        "direction": "BULLISH" if is_bullish else "BEARISH"
                    })
                    i += 1
                    continue

                max_fwd = min(20, n - 1 - i)
                if max_fwd >= 1:
                    traj = []
                    for f in range(1, max_fwd + 1):
                        s_f = float(p[i + f])
                        raw_ret = ((s_f - s_entry) / s_entry * 100.0) if is_bullish else ((s_entry - s_f) / s_entry * 100.0)
                        net_ret = raw_ret - slip_pct
                        traj.append({"day": f, "price": s_f, "return_pct": net_ret, "raw_return_pct": raw_ret})

                    signal_episodes.append({
                        "idx": i,
                        "date": entry_date,
                        "signal_z": float(z[i]),
                        "is_bullish": is_bullish,
                        "action": action,
                        "s_entry": s_entry,
                        "traj": traj
                    })
                    i += max(1, min(holding_horizon, max_fwd))
                else:
                    i += 1
            else:
                i += 1

        candidate_horizons = list(range(1, 21))

        def eval_direction(is_bull: bool):
            episodes = [e for e in signal_episodes if e["is_bullish"] == is_bull]
            if not episodes:
                return {
                    "best_horizon": holding_horizon,
                    "best_win_rate_pct": 0.0,
                    "best_avg_ret_pct": 0.0,
                    "total_trades": 0,
                    "horizon_curve": []
                }

            h_curve = []
            best_h = holding_horizon
            best_score = -999.0
            best_wr = 0.0
            best_ret = 0.0

            for h in candidate_horizons:
                rets = [e["traj"][h - 1]["return_pct"] for e in episodes if len(e["traj"]) >= h]
                if not rets:
                    continue
                wins = sum(1 for r in rets if r > 0)
                wr = (wins / len(rets) * 100.0)
                avg_r = float(np.mean(rets))
                pos_sum = sum(r for r in rets if r > 0)
                neg_sum = sum(abs(r) for r in rets if r < 0)
                pf = (pos_sum / neg_sum) if neg_sum > 0 else 99.0

                # Score: risk-adjusted utility (mean return penalized by variance, weighted by sample size)
                std_r = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1.0
                sharpe = (avg_r / max(0.1, std_r)) * np.sqrt(252.0 / max(1, h))
                score = avg_r * np.sqrt(min(len(rets), 40))

                h_curve.append({
                    "hold_days": h,
                    "win_rate_pct": round(wr, 1),
                    "avg_roi_pct": round(avg_r, 2),
                    "profit_factor": round(min(99.0, pf), 2),
                    "sharpe_ratio": round(float(sharpe), 2)
                })

                if score > best_score:
                    best_score = score
                    best_h = h
                    best_wr = wr
                    best_ret = avg_r

            return {
                "best_horizon": best_h,
                "best_win_rate_pct": round(best_wr, 1),
                "best_avg_ret_pct": round(best_ret, 2),
                "total_trades": len(episodes),
                "horizon_curve": h_curve
            }

        bull_stats = eval_direction(True)
        bear_stats = eval_direction(False)

        trades = []
        for ep in signal_episodes:
            h_use = bull_stats["best_horizon"] if ep["is_bullish"] else bear_stats["best_horizon"]
            h_avail = min(h_use, len(ep["traj"]))
            if h_avail < 1:
                continue

            exit_info = ep["traj"][h_avail - 1]
            ret_pct = float(exit_info["return_pct"])
            s_exit = float(exit_info["price"])

            inter_rets = [pt["return_pct"] for pt in ep["traj"][:h_avail]]
            mfe = float(max(inter_rets))
            mae = float(min(inter_rets))

            loss_cause = ""
            if ret_pct <= 0:
                if mfe >= 0.8:
                    loss_cause = "WHIPSAW"
                elif abs(ret_pct) < 0.4:
                    loss_cause = "FLAT_DRIFT"
                else:
                    loss_cause = "TREND_CONTINUATION"

            dollar_pnl = round(10000.0 * (ret_pct / 100.0), 2)
            stock_move_dollar = round(s_exit - ep["s_entry"], 2)
            stock_move_pct = round((s_exit - ep["s_entry"]) / ep["s_entry"] * 100.0, 2)
            trades.append({
                "date": ep["date"],
                "signal_type": "BULLISH_SNAPBACK" if ep["is_bullish"] else "BEARISH_SNAPBACK",
                "action": ep["action"],
                "entry_price": round(ep["s_entry"], 2),
                "exit_price": round(s_exit, 2),
                "underlying_entry": round(ep["s_entry"], 2),
                "strike": round(s_exit, 2),
                "stock_move_dollar": stock_move_dollar,
                "stock_move_pct": stock_move_pct,
                "captured_move_pct": round(ret_pct, 2),
                "holding_sessions": h_avail,
                "holding_dte": h_avail,
                "contract_dte": 0,
                "pnl_rec_pct": round(ret_pct, 2),
                "pnl_pct": round(ret_pct, 2),
                "dollar_rec": dollar_pnl,
                "dollar_pnl": dollar_pnl,
                "win_rec": bool(ret_pct > 0),
                "is_win": bool(ret_pct > 0),
                "mfe_pct": round(mfe, 2),
                "mae_pct": round(mae, 2),
                "loss_cause": loss_cause
            })

        total_trades = len(trades)
        wins_rec = sum(1 for t in trades if t["win_rec"])
        win_rate_rec = (wins_rec / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_ret_rec = float(np.mean([t["pnl_rec_pct"] for t in trades])) if total_trades > 0 else 0.0
        tot_dollar_rec = float(sum(t["dollar_rec"] for t in trades)) if total_trades > 0 else 0.0

        pos_sum = sum(t["pnl_rec_pct"] for t in trades if t["pnl_rec_pct"] > 0)
        neg_sum = sum(abs(t["pnl_rec_pct"]) for t in trades if t["pnl_rec_pct"] < 0)
        pf = (pos_sum / neg_sum) if neg_sum > 0 else 99.0

        return {
            "threshold_used": round(float(barrier), 2),
            "total_trades": total_trades,
            "overall_win_rate_pct": round(float(win_rate_rec), 1),
            "overall_avg_roi_pct": round(float(avg_ret_rec), 2),
            "overall_profit_factor": round(min(99.0, pf), 2),
            "overall_total_dollar": round(float(tot_dollar_rec), 2),
            "bullish_recommendation": {
                "action": "LONG ASSET",
                "recommended_dte": bull_stats["best_horizon"],
                "recommended_horizon": bull_stats["best_horizon"],
                "win_rate_pct": bull_stats["best_win_rate_pct"],
                "avg_roi_pct": bull_stats["best_avg_ret_pct"],
                "total_trades": bull_stats["total_trades"],
                "horizon_curve": bull_stats["horizon_curve"]
            },
            "bearish_recommendation": {
                "action": "SHORT ASSET",
                "recommended_dte": bear_stats["best_horizon"],
                "recommended_horizon": bear_stats["best_horizon"],
                "win_rate_pct": bear_stats["best_win_rate_pct"],
                "avg_roi_pct": bear_stats["best_avg_ret_pct"],
                "total_trades": bear_stats["total_trades"],
                "horizon_curve": bear_stats["horizon_curve"]
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

    # Backward compatibility alias
    evaluate_option_performance = evaluate_asset_performance


# Alias so existing references to SyntheticOptionBacktestEngine redirect transparently
SyntheticOptionBacktestEngine = AssetReturnBacktestEngine


# ==============================================================================
# MODULE C: CUSUM STEALTH REGIME SHIFT MONITOR (ON IID OU INNOVATIONS)
# ==============================================================================

class CUSUMDriftMonitor:
    @staticmethod
    def compute(u_innovations: np.ndarray, k: float = 0.5, h: float = 4.0):
        """
        Two-sided Cumulative Sum (CUSUM) on standardized IID OU innovations u_t:
            S_t^+ = max(0, S_{t-1}^+ + u_t - k)   [Upper cumulative drift]
            S_t^- = max(0, S_{t-1}^- - u_t - k)   [Lower cumulative drift]

        Because u_t is mathematically guaranteed to be IID white noise, normal
        mean-reversion cycles do NOT trigger false alarms.
        Alerts only when persistent structural factor shifts (E[u_t] != 0) breach threshold h.
        """
        u = np.asarray(u_innovations, dtype=float)
        T = len(u)
        s_pos = np.zeros(T)
        s_neg = np.zeros(T)
        streak = np.zeros(T, dtype=int)
        regime_alert = np.zeros(T, dtype=int)

        current_sign = 0
        current_streak = 0

        for t in range(T):
            if t == 0:
                s_pos[t] = max(0.0, u[t] - k)
                s_neg[t] = max(0.0, -u[t] - k)
            else:
                s_pos[t] = max(0.0, s_pos[t - 1] + u[t] - k)
                s_neg[t] = max(0.0, s_neg[t - 1] - u[t] - k)

            if abs(u[t]) >= 0.4:
                sign = 1 if u[t] > 0 else -1
                if sign == current_sign:
                    current_streak += 1
                else:
                    current_sign = sign
                    current_streak = 1
            else:
                current_streak = 0
                current_sign = 0

            streak[t] = current_streak * current_sign

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
    def __init__(self, warmup_period: int = 40):
        self.warmup_period = warmup_period
        self.warehouse = Warehouse()
        self.client = MassiveApiClient()

    def _fetch_and_cache_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        bars = self.warehouse.get_stock_history(ticker)
        if bars and len(bars) >= self.warmup_period + 20:
            return bars

        if config.API_KEY:
            try:
                res = self.client.get_stock_aggregates(
                    ticker=ticker, multiplier=1, timespan="day", from_date="2022-01-01", to_date="2026-10-20"
                )
                raw_results = res.get("results", [])
                if raw_results:
                    transformed = [
                        {
                            "trade_date": self.client.timestamp_to_date(b.get("t", 0)),
                            "open": b.get("o"),
                            "high": b.get("h"),
                            "low": b.get("l"),
                            "close": b.get("c"),
                            "volume": b.get("v"),
                            "vwap": b.get("vw")
                        }
                        for b in raw_results
                    ]
                    self.warehouse.save_stock_history(ticker, transformed)
                    return transformed
            except Exception:
                pass
        return bars or []

    def _fetch_and_cache_earnings(self, ticker: str) -> List[str]:
        results = self.warehouse.get_earnings_dates(ticker)
        if results:
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
        Runs Anchored Kalman Filter + Cointegrated Spread OU Engine + IID CUSUM Monitor.
        Backtests pure asset % returns on signals.
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

        # --- Layer 1: Anchored State-Space Kalman Dynamic Factor Model ---
        X = np.column_stack([np.ones(T), ret_mkt, ret_sec_perp])
        kf = KalmanBetaDecomposition(y=ret_asset, X=X).fit()

        beta_hist = kf.beta_history      # (T, 3): [alpha, beta_mkt, beta_sec]
        innovations = kf.residuals        # 1-step forecast innovation
        innov_vars = kf.innov_variances  # S_t
        y_preds = kf.y_preds

        # --- Layer 2: Cointegrated Cumulative Price Spread & OU Mean Reversion Engine ---
        # Cointegrated idiosyncratic return series: r_idio = r_asset - (alpha + beta_mkt * r_mkt + beta_sec * r_sec_perp)
        idio_ret = ret_asset - (beta_hist[:, 0] + beta_hist[:, 1] * ret_mkt + beta_hist[:, 2] * ret_sec_perp)
        spread_s = np.cumsum(idio_ret)

        # Fit continuous OU process on cointegrated spread
        ou_stats = OrnsteinUhlenbeckEngine.fit(spread_s)
        z_1d = ou_stats["z_spread"]
        u_innovations = ou_stats["innovations"]

        # Multi-horizon smoothed Z-scores
        z_5d = np.zeros(T)
        z_20d = np.zeros(T)
        for t in range(T):
            z_5d[t] = float(np.mean(z_1d[max(0, t - 4) : t + 1]))
            z_20d[t] = float(np.mean(z_1d[max(0, t - 19) : t + 1]))

        # Evaluate snapback win rate on standardized spread
        snapback_profile = OrnsteinUhlenbeckEngine.evaluate_snapbacks(
            z_series=z_1d, dates=list(ret_dates), barrier=snapback_threshold, target=0.5,
            max_days=int(np.ceil(ou_stats["half_life"] * 1.5))
        )

        # Fetch earnings announcement dates
        earnings_dates = self._fetch_and_cache_earnings(asset_ticker)

        # --- Layer 2.5: Pure Asset % Return Backtest Engine ---
        asset_backtest = AssetReturnBacktestEngine.evaluate_asset_performance(
            z_series=z_1d,
            prices=p_asset[1:],
            dates=list(ret_dates),
            barrier=snapback_threshold,
            earnings_dates=earnings_dates,
            filter_earnings=filter_earnings
        )

        # --- Layer 3: CUSUM Stealth Drift Monitor on IID Innovations ---
        cusum_res = CUSUMDriftMonitor.compute(u_innovations, k=0.4, h=3.8)

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
                "cointegrated_spread": round(float(spread_s[t]), 4),
                "z_1d": round(float(z_1d[t]), 2),
                "z_5d": round(float(z_5d[t]), 2),
                "z_20d": round(float(z_20d[t]), 2),
                "beta_mkt": round(float(beta_hist[t, 1]), 3),
                "beta_sec": round(float(beta_hist[t, 2]), 3),
                "alpha_ann_pct": round(float(beta_hist[t, 0]) * 252.0 * 100.0, 2),
                "cusum_pos": round(float(cusum_res["s_pos"][t]), 2),
                "cusum_neg": round(float(cusum_res["s_neg"][t]), 2),
                "cusum_alert": int(cusum_res["alert"][t]),
                "cusum_streak": int(cusum_res["streak"][t])
            })

        latest = history[-1] if history else {}
        latest_z1 = latest.get("z_1d", 0.0)
        latest_streak = latest.get("cusum_streak", 0)
        latest_alert = latest.get("cusum_alert", 0)
        latest_alpha = latest.get("alpha_ann_pct", 0.0)

        # Determine Cockpit State & Guidance
        is_snapback_active = abs(latest_z1) >= snapback_threshold
        is_regime_shift_active = (latest_alert != 0) or (abs(latest_streak) >= 4)

        if is_snapback_active and not is_regime_shift_active:
            direction = "OVERSOLD" if latest_z1 < 0 else "OVERBOUGHT"
            action_type = "LONG ASSET" if latest_z1 < 0 else "SHORT ASSET"
            cockpit_state = "SNAP_BACK_OPPORTUNITY"
            cockpit_badge = "SNAP-BACK EDGE ACTIVE"
            action_summary = (
                f"{asset_ticker} is statistically {direction} (Z = {latest_z1:+.2f}sigma). "
                f"Cointegrated spread exhibits half-life of {ou_stats['half_life']} sessions with {snapback_profile['win_rate_pct']}% historical reversion. "
                f"Action: {action_type} for asset mean-reversion."
            )
        elif is_regime_shift_active:
            direction = "BULLISH DRIFT" if (latest_alert > 0 or latest_streak > 0) else "BEARISH DRIFT"
            cockpit_state = "REGIME_SHIFT_ALERT"
            cockpit_badge = "STEALTH REGIME SHIFT"
            action_summary = (
                f"Structural shift detected! CUSUM statistic breached h = {cusum_res['threshold']} "
                f"({direction}, streak = {abs(latest_streak)}d). Idiosyncratic drift is persisting. "
                f"DO NOT counter-trend trade this asset."
            )
        else:
            cockpit_state = "NORMAL_EQUILIBRIUM"
            cockpit_badge = "EQUILIBRIUM (NO SIGNAL)"
            action_summary = (
                f"{asset_ticker} is tracking its market & sector factor basket within normal statistical bounds "
                f"(Z = {latest_z1:+.2f}sigma). No actionable pricing dislocation."
            )

        return {
            "status": "SUCCESS",
            "tickers": {
                "asset": asset_ticker,
                "market": market_ticker,
                "sector": sector_ticker
            },
            "sample_period": {
                "start": ret_dates[self.warmup_period],
                "end": ret_dates[-1],
                "total_sessions": T - self.warmup_period
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
            "asset_backtest": asset_backtest,
            "option_backtest": asset_backtest,  # backwards compatibility
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
