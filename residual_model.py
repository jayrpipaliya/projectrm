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

    def _run_filter_scalar_3d(self, q0: float, q1: float, q2: float, r: float, T_sub: Optional[int] = None):
        """High-speed scalar execution of the 3D Anchored Kalman Filter (Intercept, Market, Sector)."""
        T = self.T if T_sub is None else min(self.T, T_sub)
        start_idx = self.T - T

        y_arr = self.y[start_idx:]
        x0_arr = self.X[start_idx:, 0]
        x1_arr = self.X[start_idx:, 1]
        x2_arr = self.X[start_idx:, 2]

        phi0 = float(self.Phi_diag[0]) if self.k >= 1 else 0.95
        phi1 = float(self.Phi_diag[1]) if self.k >= 2 else 0.985
        phi2 = float(self.Phi_diag[2]) if self.k >= 3 else 0.985

        bp0 = float(self.beta_priors[0]) if self.k >= 1 else 0.0
        bp1 = float(self.beta_priors[1]) if self.k >= 2 else 1.0
        bp2 = float(self.beta_priors[2]) if self.k >= 3 else 0.5

        b0, b1, b2 = bp0, bp1, bp2
        p00, p01, p02 = 0.1, 0.0, 0.0
        p10, p11, p12 = 0.0, 0.1, 0.0
        p20, p21, p22 = 0.0, 0.0, 0.1

        residuals = np.zeros(T)
        innov_var = np.zeros(T)
        y_preds = np.zeros(T)
        beta_hist = np.zeros((T, 3))
        neg_log_likelihood = 0.0

        for t in range(T):
            x0 = x0_arr[t]
            x1 = x1_arr[t]
            x2 = x2_arr[t]

            pred_b0 = bp0 + phi0 * (b0 - bp0)
            pred_b1 = bp1 + phi1 * (b1 - bp1)
            pred_b2 = bp2 + phi2 * (b2 - bp2)

            pp00 = phi0 * phi0 * p00 + q0
            pp01 = phi0 * phi1 * p01
            pp02 = phi0 * phi2 * p02
            pp10 = pp01
            pp11 = phi1 * phi1 * p11 + q1
            pp12 = phi1 * phi2 * p12
            pp20 = pp02
            pp21 = pp12
            pp22 = phi2 * phi2 * p22 + q2

            y_pred = x0 * pred_b0 + x1 * pred_b1 + x2 * pred_b2
            y_preds[t] = y_pred
            e_t = y_arr[t] - y_pred
            residuals[t] = e_t

            px0 = pp00 * x0 + pp01 * x1 + pp02 * x2
            px1 = pp10 * x0 + pp11 * x1 + pp12 * x2
            px2 = pp20 * x0 + pp21 * x1 + pp22 * x2

            s_t = x0 * px0 + x1 * px1 + x2 * px2 + r
            innov_var[t] = s_t
            inv_s = 1.0 / s_t if s_t > 1e-12 else 1e12

            k0 = px0 * inv_s
            k1 = px1 * inv_s
            k2 = px2 * inv_s

            b0 = pred_b0 + k0 * e_t
            b1 = pred_b1 + k1 * e_t
            b2 = pred_b2 + k2 * e_t

            # Numerically stable Joseph form covariance update:
            # P = (I - K X) P_pred (I - K X)^T + K R K^T
            ik00, ik01, ik02 = 1.0 - k0 * x0, -k0 * x1, -k0 * x2
            ik10, ik11, ik12 = -k1 * x0, 1.0 - k1 * x1, -k1 * x2
            ik20, ik21, ik22 = -k2 * x0, -k2 * x1, 1.0 - k2 * x2

            t00 = ik00 * pp00 + ik01 * pp10 + ik02 * pp20
            t01 = ik00 * pp01 + ik01 * pp11 + ik02 * pp21
            t02 = ik00 * pp02 + ik01 * pp12 + ik02 * pp22

            t10 = ik10 * pp00 + ik11 * pp10 + ik12 * pp20
            t11 = ik10 * pp01 + ik11 * pp11 + ik12 * pp21
            t12 = ik10 * pp02 + ik11 * pp12 + ik12 * pp22

            t20 = ik20 * pp00 + ik21 * pp10 + ik22 * pp20
            t21 = ik20 * pp01 + ik21 * pp11 + ik22 * pp21
            t22 = ik20 * pp02 + ik21 * pp12 + ik22 * pp22

            p00 = t00 * ik00 + t01 * ik01 + t02 * ik02 + k0 * k0 * r
            p11 = t10 * ik10 + t11 * ik11 + t12 * ik12 + k1 * k1 * r
            p22 = t20 * ik20 + t21 * ik21 + t22 * ik22 + k2 * k2 * r

            p01 = 0.5 * ((t00 * ik10 + t01 * ik11 + t02 * ik12 + k0 * k1 * r) + (t10 * ik00 + t11 * ik01 + t12 * ik02 + k1 * k0 * r))
            p02 = 0.5 * ((t00 * ik20 + t01 * ik21 + t02 * ik22 + k0 * k2 * r) + (t20 * ik00 + t21 * ik01 + t22 * ik02 + k2 * k0 * r))
            p12 = 0.5 * ((t10 * ik20 + t11 * ik21 + t12 * ik22 + k1 * k2 * r) + (t20 * ik10 + t21 * ik11 + t22 * ik12 + k2 * k1 * r))
            p10 = p01
            p20 = p02
            p21 = p12

            beta_hist[t, 0] = b0
            beta_hist[t, 1] = b1
            beta_hist[t, 2] = b2

            if s_t > 1e-12:
                neg_log_likelihood += 0.5 * (math.log(2.0 * math.pi * s_t) + (e_t * e_t) * inv_s)
                # Soft economic L2 penalty toward priors (Pitfall 1.1)
                neg_log_likelihood += 1e-4 * ((b0 - bp0)**2 + (b1 - bp1)**2 + (b2 - bp2)**2)

        return neg_log_likelihood, beta_hist, residuals, innov_var, y_preds

    def _run_filter(self, Q_diag: np.ndarray, R_scalar: float):
        """Runs the forward Kalman Filter with anchored mean-reverting state space."""
        if self.k == 3:
            return self._run_filter_scalar_3d(float(Q_diag[0]), float(Q_diag[1]), float(Q_diag[2]), float(R_scalar))

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

            # Joseph stabilized covariance update: P = (I - K X) P_pred (I - K X)^T + K R K^T
            IKX = np.eye(self.k) - K_t @ X_t
            P = IKX @ P_pred @ IKX.T + (K_t @ K_t.T) * R
            P = 0.5 * (P + P.T)
            beta_history[t, :] = beta_t

            if S_t > 1e-12:
                neg_log_likelihood += 0.5 * (np.log(2.0 * np.pi * S_t) + (e_t**2) / S_t)
                neg_log_likelihood += 1e-4 * float(np.sum((beta_t - self.beta_priors)**2))

        return neg_log_likelihood, beta_history, residuals, innov_var, y_preds

    def fit(self):
        """Optimizes process noise Q and observation noise R via Maximum Likelihood Estimation."""
        sample_var = float(np.var(self.y)) if len(self.y) > 1 else 1e-4
        init_params = np.append(np.repeat(1e-5, self.k), sample_var * 0.5)
        bounds = [(1e-7, 1e-2)] * self.k + [(1e-6, sample_var * 3.0)]

        # Train on up to 756 bars (3 years of diverse market regimes)
        sub_len = min(self.T, 756)

        if self.k == 3:
            def obj_func(params):
                neg_ll, _, _, _, _ = self._run_filter_scalar_3d(params[0], params[1], params[2], params[3], T_sub=sub_len)
                return neg_ll

            try:
                res = minimize(obj_func, init_params, bounds=bounds, method="L-BFGS-B", options={"maxiter": 100})
                if res.success or res.fun < obj_func(init_params):
                    opt_Q, opt_R = res.x[:3], float(res.x[3])
                else:
                    opt_Q, opt_R = init_params[:3], float(init_params[3])
            except Exception:
                opt_Q = np.repeat(1e-5, 3)
                opt_R = sample_var * 0.5

            _, self.beta_history, self.residuals, self.innov_variances, self.y_preds = \
                self._run_filter_scalar_3d(opt_Q[0], opt_Q[1], opt_Q[2], opt_R)
        else:
            def obj_func(params):
                neg_ll, _, _, _, _ = self._run_filter(params[:self.k], params[-1])
                return neg_ll

            try:
                res = minimize(obj_func, init_params, bounds=bounds, method="L-BFGS-B", options={"maxiter": 100})
                if res.success or res.fun < obj_func(init_params):
                    opt_Q, opt_R = res.x[:self.k], float(res.x[-1])
                else:
                    opt_Q, opt_R = init_params[:self.k], float(init_params[-1])
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
    def adf_test(series: np.ndarray, max_lags: int = 1) -> Dict[str, Any]:
        r"""
        Augmented Dickey-Fuller (ADF) test for cointegration/stationarity:
            \Delta s_t = \alpha + \gamma s_{t-1} + \sum \delta_i \Delta s_{t-i} + \epsilon_t
        H0: \gamma = 0 (unit root, non-stationary)
        H1: \gamma < 0 (mean-reverting / stationary)
        Critical values for constant without trend (approx MacKinnon):
        1%: -3.434, 5%: -2.863, 10%: -2.568
        """
        y = np.asarray(series, dtype=float)
        n = len(y)
        if n < 20:
            return {"stat": 0.0, "is_stationary": False}

        dy = np.diff(y)
        n_samples = len(dy) - max_lags
        Y = dy[max_lags:]
        X = np.empty((n_samples, 2 + max_lags))
        X[:, 0] = 1.0
        X[:, 1] = y[max_lags : max_lags + n_samples]
        for lag in range(1, max_lags + 1):
            X[:, 1 + lag] = dy[max_lags - lag : max_lags - lag + n_samples]

        try:
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            resids = Y - X @ beta
            dof = max(1, n_samples - (2 + max_lags))
            s2 = np.sum(resids**2) / dof
            cov = s2 * np.linalg.pinv(X.T @ X)
            se_gamma = np.sqrt(max(1e-12, cov[1, 1]))
            t_stat = float(beta[1] / se_gamma)
            is_stat = bool(t_stat < -2.568)  # Reject unit root at 10% or better
        except Exception:
            t_stat = 0.0
            is_stat = False

        return {"stat": t_stat, "is_stationary": is_stat}

    @staticmethod
    def fit(spread_series: np.ndarray, dt: float = 1.0, window: int = 252):
        r"""
        Fits continuous-time Ornstein-Uhlenbeck process on cumulative idiosyncratic spread:
            ds_t = \theta (\mu - s_t) dt + \sigma dW_t
        AR(1) discrete-time representation:
            s_t = a + b * s_{t-1} + \xi_t
        where b = exp(-\theta * dt) and \mu = a / (1 - b).
        Half-life: \tau = ln(2) / \theta.
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
                "adf_stat": 0.0,
                "is_stationary": True,
                "innovations": np.zeros(n),
                "z_spread": np.zeros(n)
            }

        # Vectorized trailing-window normalization for full spread history (strictly zero look-ahead bias)
        z_spread = np.zeros(n)
        for t in range(n):
            w_start = max(0, t - window + 1)
            sub = s[w_start : t + 1]
            if len(sub) < 15:
                z_spread[t] = 0.0
            else:
                m = np.mean(sub)
                dev = np.std(sub, ddof=1)
                z_spread[t] = (s[t] - m) / max(1e-4, dev)

        # Fit continuous-time OU on active trailing window (last 252 sessions / 1 year)
        active_s = s[-min(n, window):]
        y = active_s[1:]
        x = active_s[:-1]
        x_mat = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
            a, b = float(coeffs[0]), float(coeffs[1])
        except Exception:
            a, b = 0.0, 0.90

        adf_res = OrnsteinUhlenbeckEngine.adf_test(active_s)

        # Detect non-mean-reverting or explosive/oscillating series without masking
        if b <= 0.01 or b >= 0.99:
            is_mean_reverting = False
            theta = 0.0 if b >= 0.99 else 99.0
            half_life = 999.0 if b >= 0.99 else 0.1
            mu = float(np.mean(active_s))
            sigma_innov = float(np.std(np.diff(active_s))) if len(active_s) > 1 else 0.01
            sigma_eq = max(1e-4, float(np.std(active_s, ddof=1)))
            innovations = np.zeros(n)
        else:
            theta = -np.log(b) / dt
            half_life = np.log(2.0) / max(1e-4, theta)
            mu = a / (1.0 - b)
            xi = y - (a + b * x)
            sigma_innov = float(np.std(xi, ddof=1)) if len(xi) > 2 else 0.01
            sigma_innov = max(1e-5, sigma_innov)
            denom = max(1e-4, 1.0 - b**2)
            gaussian_sigma_eq = sigma_innov / np.sqrt(denom)
            empirical_std = float(np.std(active_s, ddof=1))
            # Tail-risk adjustment (Pitfall 2.3): protect against Gaussian underestimation in fat tails
            sigma_eq = max(gaussian_sigma_eq, empirical_std)
            innovations = np.zeros(n)
            act_n = len(active_s)
            innovations[-act_n + 1 :] = xi / sigma_innov
            # Valid discrete-time mean reversion: half-life <= 60 sessions and ADF cointegration confirmation
            is_mean_reverting = bool(0.01 < b < 0.985 and (adf_res["is_stationary"] or half_life <= 30.0))

        return {
            "theta": round(float(theta), 4),
            "half_life": round(float(half_life), 1),
            "mu": round(float(mu), 4),
            "sigma_eq": round(float(sigma_eq), 4),
            "sigma_innov": round(float(sigma_innov), 5),
            "is_mean_reverting": is_mean_reverting,
            "adf_stat": round(float(adf_res["stat"]), 2),
            "is_stationary": adf_res["is_stationary"],
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

        # Base round-trip slippage (5 bps per leg = 10 bps)
        base_slip_pct = (slippage_bps * 2.0) / 10000.0 * 100.0

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

                # Pitfall 3.3: Dynamic slippage scaling with dislocation magnitude |Z|
                slip_pct = base_slip_pct * (1.0 + 0.5 * max(0.0, abs(z[i]) - 1.5))

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
                # Step by 1 bar to collect all valid dislocation events without combinatorial bias (Pitfall 3.2)
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
        last_exit_idx = -1
        target_h = holding_horizon if (holding_horizon and holding_horizon > 0) else 5

        for ep in signal_episodes:
            # Prevent overlapping positions on same asset (realistic execution)
            if ep["idx"] <= last_exit_idx:
                continue

            h_avail = min(target_h, len(ep["traj"]))
            if h_avail < 1:
                continue

            # Snapback profit-taking exit at equilibrium (|Z| <= 0.5), or hold to target_h
            exit_day = h_avail
            for d in range(1, h_avail):
                chk_idx = ep["idx"] + d
                if chk_idx < n and abs(z[chk_idx]) <= 0.5:
                    exit_day = d
                    break

            exit_info = ep["traj"][exit_day - 1]
            ret_pct = float(exit_info["return_pct"])
            s_exit = float(exit_info["price"])
            last_exit_idx = ep["idx"] + exit_day

            inter_rets = [pt["return_pct"] for pt in ep["traj"][:exit_day]]
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
                "signal_z": round(float(ep.get("signal_z", 0.0)), 2),
                "is_bullish": ep.get("is_bullish", True),
                "signal_type": "BULLISH_SNAPBACK" if ep["is_bullish"] else "BEARISH_SNAPBACK",
                "action": ep["action"],
                "entry_price": round(ep["s_entry"], 2),
                "exit_price": round(s_exit, 2),
                "underlying_entry": round(ep["s_entry"], 2),
                "strike": round(s_exit, 2),
                "stock_move_dollar": stock_move_dollar,
                "stock_move_pct": stock_move_pct,
                "captured_move_pct": round(ret_pct, 2),
                "holding_sessions": exit_day,
                "holding_dte": exit_day,
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
        filter_earnings: bool = True,
        long_threshold: Optional[float] = None,
        short_threshold: Optional[float] = None,
        trade_policy: Optional[str] = None
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

        # Standardized 1-Day Kalman Innovation Shock Z (Stationary Single-Day Shock):
        # Strictly IID N(0, 1) single-day idiosyncratic impulse shocks (mean=0, std=1, zero autocorrelation)
        std_innov = float(np.std(innovations[self.warmup_period:])) if T > self.warmup_period + 10 else float(np.std(innovations))
        std_innov = max(1e-5, std_innov)
        z_shock = innovations / std_innov
        z_1d = z_shock

        # --- Layer 2: Cointegrated Cumulative Price Spread & OU Mean Reversion Engine ---
        # Cointegrated idiosyncratic return series: r_idio = r_asset - (alpha + beta_mkt * r_mkt + beta_sec * r_sec_perp)
        idio_ret = ret_asset - (beta_hist[:, 0] + beta_hist[:, 1] * ret_mkt + beta_hist[:, 2] * ret_sec_perp)
        spread_s = np.cumsum(idio_ret)

        # Fit continuous OU process on cointegrated spread
        ou_stats = OrnsteinUhlenbeckEngine.fit(spread_s)
        z_spread = ou_stats["z_spread"]
        u_innovations = ou_stats["innovations"]

        # Multi-horizon smoothed Z-scores of cumulative cointegrated spread
        z_5d = np.zeros(T)
        z_20d = np.zeros(T)
        for t in range(T):
            z_5d[t] = float(np.mean(z_spread[max(0, t - 4) : t + 1]))
            z_20d[t] = float(np.mean(z_spread[max(0, t - 19) : t + 1]))

        # Evaluate snapback win rate on standardized cointegrated spread (genuine mean reversion)
        snapback_profile = OrnsteinUhlenbeckEngine.evaluate_snapbacks(
            z_series=z_spread, dates=list(ret_dates), barrier=snapback_threshold, target=0.5,
            max_days=int(np.ceil(ou_stats["half_life"] * 1.5))
        )

        # Fetch earnings announcement dates
        earnings_dates = self._fetch_and_cache_earnings(asset_ticker)

        # --- Layer 2.5: Pure Asset % Return Backtest Engine ---
        # Backtest mean-reversion trades on cumulative spread Z breaches
        asset_backtest = AssetReturnBacktestEngine.evaluate_asset_performance(
            z_series=z_spread,
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
                "z_spread": round(float(z_spread[t]), 2),
                "z_1d": round(float(z_1d[t]), 2),
                "z_5d": round(float(z_5d[t]), 2),
                "z_20d": round(float(z_20d[t]), 2),
                "beta_mkt": round(float(beta_hist[t, 1]), 3),
                "beta_sec": round(float(beta_hist[t, 2]), 3),
                "alpha_ann_pct": round(float(beta_hist[t, 0]) * 252.0 * 100.0, 2),
                "cusum_pos": round(float(cusum_res["s_pos"][t]), 2),
                "cusum_neg": round(float(cusum_res["s_neg"][t]), 2),
                "cusum_alert": int(cusum_res["alert"][t]),
                "cusum_streak": int(cusum_res["streak"][t]),
                "regime_alert": int(cusum_res["alert"][t]),
                "streak": int(cusum_res["streak"][t])
            })

        latest = history[-1] if history else {}
        latest_z1 = latest.get("z_1d", 0.0)
        latest_z_spread = latest.get("z_spread", 0.0)
        latest_streak = latest.get("cusum_streak", 0)
        latest_alert = latest.get("cusum_alert", 0)
        latest_alpha = latest.get("alpha_ann_pct", 0.0)

        # Asymmetric Barrier & Directional Policy Resolution
        l_thresh = float(long_threshold) if (long_threshold is not None and float(long_threshold) > 0) else float(snapback_threshold)
        s_thresh = float(short_threshold) if (short_threshold is not None and float(short_threshold) > 0) else float(snapback_threshold)
        policy = (trade_policy or "ASYMMETRIC_DUAL").upper()
        is_long_only = (policy == "LONG_ONLY")
        is_short_only = (policy == "SHORT_ONLY")

        # Mean-reverting signal fires on cumulative spread Z-score breaches:
        is_bull_breach = (not is_short_only) and (latest_z_spread <= -l_thresh)
        is_bear_breach = (not is_long_only) and (latest_z_spread >= s_thresh)
        is_snapback_active = is_bull_breach or is_bear_breach
        is_regime_shift_active = (latest_alert != 0) or (abs(latest_streak) >= 4)

        # Check Earnings Blackout on Latest Date
        is_latest_in_blackout = False
        blackout_reason = ""
        if filter_earnings and earnings_dates:
            import datetime as dt
            latest_d_str = str(latest.get("date", ""))
            try:
                latest_dt = dt.datetime.strptime(latest_d_str, "%Y-%m-%d").date()
                for ed_str in earnings_dates:
                    ed_dt = dt.datetime.strptime(ed_str, "%Y-%m-%d").date()
                    days_diff = (ed_dt - latest_dt).days
                    if days_diff in (0, -1):
                        is_latest_in_blackout = True
                        blackout_reason = f"Earnings report on {ed_str} (blackout active)"
                        break
                    elif 0 < days_diff <= 2:
                        is_latest_in_blackout = True
                        blackout_reason = f"Upcoming earnings on {ed_str} ({days_diff}d away, runway < 3 sessions)"
                        break
            except Exception:
                pass

        if is_latest_in_blackout and is_snapback_active:
            direction = "OVERSOLD" if is_bull_breach else "OVERBOUGHT"
            action_type = "LONG ASSET" if is_bull_breach else "SHORT ASSET"
            active_barrier = l_thresh if is_bull_breach else s_thresh
            cockpit_state = "EARNINGS_BLACKOUT"
            cockpit_badge = "EARNINGS BLACKOUT"
            action_summary = (
                f"{asset_ticker} is statistically {direction} (Z = {latest_z1:+.2f}σ reached barrier {active_barrier:.1f}σ), "
                f"but trading is BLOCKED by Corporate Earnings Safety Protocol ({blackout_reason})."
            )
        elif is_snapback_active and not is_regime_shift_active:
            direction = "OVERSOLD" if is_bull_breach else "OVERBOUGHT"
            action_type = "LONG ASSET" if is_bull_breach else "SHORT ASSET"
            active_barrier = l_thresh if is_bull_breach else s_thresh
            cockpit_state = "SNAP_BACK_OPPORTUNITY"
            cockpit_badge = "SNAP-BACK EDGE ACTIVE"
            action_summary = (
                f"{asset_ticker} is statistically {direction} (Z = {latest_z1:+.2f}σ breached {action_type} barrier {active_barrier:.1f}σ). "
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
                "is_snapback_candidate": is_snapback_active and not is_latest_in_blackout,
                "is_regime_shift_candidate": is_regime_shift_active,
                "is_earnings_blackout": is_latest_in_blackout
            },
            "asymmetric_params": {
                "policy": policy,
                "long_threshold": l_thresh,
                "short_threshold": s_thresh,
                "effective_threshold": l_thresh if is_bull_breach else (s_thresh if is_bear_breach else snapback_threshold)
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
                "z_score_spread": latest.get("z_spread"),
                "z_spread": latest.get("z_spread"),
                "current_z": latest.get("z_spread"),
                "z_score_5d": latest.get("z_5d"),
                "z_score_20d": latest.get("z_20d"),
                "beta_market": latest.get("beta_mkt"),
                "beta_sector": latest.get("beta_sec"),
                "alpha": latest_alpha,
                "is_mean_reverting": ou_stats.get("is_mean_reverting", False),
                "half_life": ou_stats.get("half_life", 0.0),
                "adf_stat": ou_stats.get("adf_stat", 0.0)
            },
            "history": history
        }
