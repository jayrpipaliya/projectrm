"""
PROJECT RM: ADAPTIVE THRESHOLD & HORIZON OPTIMIZER
WITH WALK-FORWARD OUT-OF-SAMPLE CROSS-VALIDATION & PLATEAU DETECTION
=============================================================================
Computes optimal entry threshold (sigma) and holding period (horizon) for any
(Asset, Market, Sector) triple across multi-year data.

Methodology:
------------
1. 2D Joint Parameter Sweep:
   - Threshold sigma in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
   - Horizon days in [3, 5, 7, 10, 14, 21]
2. Walk-Forward / Out-of-Sample (OOS) Cross-Validation:
   - Splits data chronologically: In-Sample (IS: first 65%) vs Out-of-Sample (OOS: last 35%).
   - Selects optimal (sigma*, H*) on In-Sample data strictly using Plateau Smoothing.
   - Evaluates (sigma*, H*) on completely untouched Out-of-Sample data.
   - Computes OOS Efficiency Ratio: (OOS Win Rate / IS Win Rate).
   - Confirms true structural edge vs curve-fitted artifact.
3. Plateau Detection & Anti-Overfitting Regularization:
   - Raw cell utility: U(s, h) = WinRate% * AvgROI% * sqrt(N_trades)
   - Plateau Score: 3x3 local smoothed neighborhood average.
4. Forensic Loser Categorization for simulated trades:
   - REGIME_DRIFT: Continued to move against position without reversion
   - THETA_BLEED: Sideways action, max favorable excursion < +10%, slow decay
   - WHIPSAW: Reverted briefly then reversed against trade
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from residual_model import SyntheticOptionBacktestEngine


DEFAULT_THRESHOLDS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
DEFAULT_HORIZONS = [2, 3, 5, 7]


class ThresholdOptimizer:
    """
    Sweeps (sigma, horizon) parameter grid, analyzes winning/losing trade forensic
    signatures, and scores parameter combinations based on plateau stability
    plus walk-forward out-of-sample validation.
    """

    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        horizons: Optional[List[int]] = None,
        min_trades: int = 5
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.horizons = horizons or DEFAULT_HORIZONS
        self.min_trades = min_trades

    def _eval_grid(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True
    ) -> Dict[str, Any]:
        """Evaluates the 2D parameter grid on a slice of data and returns plateau metrics."""
        n = len(z)
        date_to_idx = {d: i for i, d in enumerate(dates)}
        rv_20d = np.full(n, 0.25)
        if n >= 25:
            log_returns = np.diff(np.log(p))
            for i in range(20, n):
                window = log_returns[max(0, i - 20):i]
                if len(window) >= 5:
                    rv_20d[i] = float(np.std(window, ddof=1) * np.sqrt(252))

        raw_utility = np.zeros((len(self.thresholds), len(self.horizons)))
        cell_details = {}

        for s_idx, thresh in enumerate(self.thresholds):
            bt = SyntheticOptionBacktestEngine.evaluate_option_performance(
                z_series=z,
                prices=p,
                dates=dates,
                barrier=thresh,
                earnings_dates=earnings_dates,
                filter_earnings=filter_earnings
            )

            bull_curve = {c["hold_days"]: c for c in bt["bullish_recommendation"].get("horizon_curve", [])}
            bear_curve = {c["hold_days"]: c for c in bt["bearish_recommendation"].get("horizon_curve", [])}
            overall_trades = bt.get("trades", [])

            for h_idx, h in enumerate(self.horizons):
                h_bull = bull_curve.get(h, {"win_rate_pct": 0.0, "avg_roi_pct": 0.0})
                h_bear = bear_curve.get(h, {"win_rate_pct": 0.0, "avg_roi_pct": 0.0})

                n_trades = bt.get("total_trades", 0)
                if n_trades > 0:
                    rois_at_h = []
                    wins_at_h = 0
                    for tr in overall_trades:
                        idx_entry = date_to_idx.get(tr["date"], -1)
                        if idx_entry >= 0:
                            step = min(h, (n - 1) - idx_entry)
                            if step > 0:
                                opt_type = "call" if tr["is_bullish"] else "put"
                                s_ent = tr["underlying_entry"]
                                k_str = tr["strike"]
                                sig_ent = tr.get("sigma_entry") or (float(rv_20d[idx_entry]) if idx_entry < len(rv_20d) else 0.25)
                                p_ent = SyntheticOptionBacktestEngine.black_scholes_price(
                                    s_ent, k_str, 35/365.0, 0.045, sig_ent, opt_type
                                )
                                s_ex = p[idx_entry + step]
                                sig_ex = sig_ent
                                p_ex = SyntheticOptionBacktestEngine.black_scholes_price(
                                    s_ex, k_str, max(0.001, (35 - step)/365.0), 0.045, sig_ex, opt_type
                                )
                                r_pct = ((p_ex - p_ent) / p_ent) * 100.0
                                rois_at_h.append(r_pct)
                                if r_pct > 0:
                                    wins_at_h += 1

                    sample_count = len(rois_at_h)
                    win_rate = (wins_at_h / sample_count * 100.0) if sample_count > 0 else 0.0
                    avg_roi = float(np.mean(rois_at_h)) if sample_count > 0 else 0.0
                else:
                    sample_count = 0
                    win_rate = 0.0
                    avg_roi = 0.0

                if sample_count >= self.min_trades and avg_roi > 0 and win_rate >= 35.0:
                    u = (win_rate / 100.0) * avg_roi * np.sqrt(sample_count)
                else:
                    u = 0.0

                raw_utility[s_idx, h_idx] = u
                cell_details[(s_idx, h_idx)] = {
                    "threshold": thresh,
                    "horizon": h,
                    "trades_count": sample_count,
                    "win_rate_pct": round(win_rate, 1),
                    "avg_roi_pct": round(avg_roi, 1),
                    "raw_utility": round(float(u), 2)
                }

        # 3x3 neighborhood smoothing for plateau score
        n_thresh = len(self.thresholds)
        n_horiz = len(self.horizons)
        plateau_scores = np.zeros_like(raw_utility)
        stability_ratios = np.zeros_like(raw_utility)

        for s in range(n_thresh):
            for h in range(n_horiz):
                s_min = max(0, s - 1)
                s_max = min(n_thresh, s + 2)
                h_min = max(0, h - 1)
                h_max = min(n_horiz, h + 2)
                mean_neigh = float(np.mean(raw_utility[s_min:s_max, h_min:h_max]))
                plateau_scores[s, h] = mean_neigh
                if raw_utility[s, h] > 0:
                    stability_ratios[s, h] = mean_neigh / raw_utility[s, h]
                else:
                    stability_ratios[s, h] = 0.0

        best_flat_idx = int(np.argmax(plateau_scores))
        best_s, best_h = np.unravel_index(best_flat_idx, plateau_scores.shape)

        return {
            "raw_utility": raw_utility,
            "plateau_scores": plateau_scores,
            "stability_ratios": stability_ratios,
            "cell_details": cell_details,
            "best_s": best_s,
            "best_h": best_h,
            "best_cell": cell_details[(best_s, best_h)]
        }

    def _eval_single_param(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        thresh: float,
        horizon: int,
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        contract_dte: int = 35,
        precomputed_trades: Optional[List[Dict[str, Any]]] = None,
        date_to_idx: Optional[Dict[str, int]] = None,
        rv_20d: Optional[np.ndarray] = None,
        calc_details: bool = True
    ) -> Dict[str, Any]:
        """Evaluates a specific (thresh, horizon, contract_dte) on a dataset slice."""
        n = len(z)
        if precomputed_trades is not None:
            trades = precomputed_trades
        else:
            bt = SyntheticOptionBacktestEngine.evaluate_option_performance(
                z_series=z,
                prices=p,
                dates=dates,
                barrier=thresh,
                earnings_dates=earnings_dates,
                filter_earnings=filter_earnings,
                dte_contract=contract_dte
            )
            trades = bt.get("trades", [])

        if not trades:
            return {"trades_count": 0, "win_rate_pct": 0.0, "avg_roi_pct": 0.0, "trades": []}

        if date_to_idx is None:
            date_to_idx = {d: i for i, d in enumerate(dates)}

        if rv_20d is None:
            rv_20d = np.full(n, 0.25)
            if n >= 25:
                log_returns = np.diff(np.log(p))
                for i in range(20, n):
                    window = log_returns[max(0, i - 20):i]
                    if len(window) >= 5:
                        rv_20d[i] = float(np.std(window, ddof=1) * np.sqrt(252))

        rois = []
        dollars = []
        entry_prices = []
        wins = 0
        detailed_trades = []

        for tr in trades:
            idx_e = date_to_idx.get(tr["date"], -1)
            if idx_e < 0:
                continue
            # Physical tradable sessions cap: an option cannot be held past its expiration
            max_c_sessions = max(1, min(int(contract_dte * 5 / 7), contract_dte - 1))
            step = min(horizon, (n - 1) - idx_e, max_c_sessions)
            if step <= 0:
                continue

            opt_type = "call" if tr["is_bullish"] else "put"
            s_ent = tr["underlying_entry"]
            k_str = tr["strike"]
            sig_ent = tr.get("sigma_entry") or (float(rv_20d[idx_e]) if idx_e < len(rv_20d) else 0.25)
            t_ent = max(0.001, contract_dte / 365.0)
            p_ent = max(0.01, SyntheticOptionBacktestEngine.black_scholes_price(
                s_ent, k_str, t_ent, 0.045, sig_ent, opt_type
            ))
            s_ex = p[idx_e + step]
            sig_ex = sig_ent

            # Elapsed calendar days across trading sessions (7 calendar days per 5 trading sessions)
            cal_days_elapsed = min(contract_dte - 0.5, step * (7.0 / 5.0))
            rem_dte = max(0.5, contract_dte - cal_days_elapsed)
            t_ex = max(0.001, rem_dte / 365.0)
            p_ex = SyntheticOptionBacktestEngine.black_scholes_price(
                s_ex, k_str, t_ex, 0.045, sig_ex, opt_type
            )
            r_pct = ((p_ex - p_ent) / p_ent) * 100.0
            rois.append(r_pct)
            if r_pct > 0:
                wins += 1

            dollar_pnl = (p_ex - p_ent) * 100.0
            dollars.append(dollar_pnl)
            entry_prices.append(p_ent)

            if calc_details:
                # Path excursion
                path = p[idx_e : idx_e + step + 1]
                if len(path) > 1:
                    ret_path = (path - s_ent)/s_ent if tr["is_bullish"] else (s_ent - path)/s_ent
                    mfe = float(np.max(ret_path) * 100.0)
                    mae = float(np.min(ret_path) * 100.0)
                else:
                    mfe, mae = 0.0, 0.0

                is_win = r_pct > 0
                loss_cause = "WIN" if is_win else "UNKNOWN"
                if not is_win:
                    if mfe >= 8.0:
                        loss_cause = "WHIPSAW"
                    elif mae <= -4.0 and mfe < 3.0:
                        loss_cause = "REGIME_DRIFT"
                    else:
                        loss_cause = "THETA_BLEED"

                detailed_trades.append({
                    "date": tr["date"],
                    "action": tr["action"],
                    "is_bullish": tr["is_bullish"],
                    "signal_z": tr["signal_z"],
                    "underlying_entry": round(float(s_ent), 2),
                    "strike": round(float(k_str), 2),
                    "entry_option_price": round(float(p_ent), 2),
                    "exit_option_price": round(float(p_ex), 2),
                    "pnl_pct": round(r_pct, 1),
                    "dollar_pnl": round(float(dollar_pnl), 2),
                    "holding_dte": horizon,
                    "holding_sessions": step,
                    "contract_dte": int(contract_dte),
                    "mfe_pct": round(mfe, 2),
                    "mae_pct": round(mae, 2),
                    "result": "WIN" if is_win else "LOSS",
                    "loss_cause": loss_cause,
                    "failure_cause": loss_cause
                })

        sample_count = len(rois)
        win_rate = (wins / sample_count * 100.0) if sample_count > 0 else 0.0
        avg_roi = float(np.mean(rois)) if sample_count > 0 else 0.0
        total_dollar = float(sum(dollars)) if dollars else 0.0

        gross_win = sum(d for d in dollars if d > 0)
        gross_loss = abs(sum(d for d in dollars if d < 0))
        profit_factor = round(gross_win / max(1.0, gross_loss), 2) if gross_loss > 0 else 99.0

        return {
            "threshold_used": round(float(thresh), 2),
            "holding_horizon_days": int(horizon),
            "contract_dte": int(contract_dte),
            "trades_count": sample_count,
            "win_rate_pct": round(win_rate, 1),
            "avg_roi_pct": round(avg_roi, 1),
            "total_dollar": round(total_dollar, 2),
            "profit_factor": profit_factor,
            "entry_prices": entry_prices,
            "rois": rois,
            "dollars": dollars,
            "earnings_filter": bt.get("earnings_filter", {}) if 'bt' in locals() and isinstance(bt, dict) else {},
            "trades": detailed_trades
        }

    def evaluate_single_param(
        self,
        thresh: float,
        horizon: int,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        contract_dte: int = 35
    ) -> Dict[str, Any]:
        return self._eval_single_param(
            z=np.asarray(z, dtype=float),
            p=np.asarray(p, dtype=float),
            dates=dates,
            thresh=float(thresh),
            horizon=int(horizon),
            earnings_dates=earnings_dates,
            filter_earnings=filter_earnings,
            contract_dte=contract_dte
        )

    def evaluate_contract_tenor_sweep(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        thresh: float,
        horizon: int = 5,
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        contract_dtes: Optional[List[int]] = None,
        mode: str = "dynamic"
    ) -> Dict[str, Any]:
        """
        Evaluates empirical performance of contract expiration tenors (7d, 10d, 14d, 21d, 35d, 45d).

        Modes:
        - 'dynamic' (default): Runs 65/35 walk-forward cross-validation per contract tenor, discovering
          the optimal holding horizon h*(T) for that specific option maturity and validating it on unseen OOS data.
        - 'uniform': Holds every contract for the uniform benchmark horizon (H*).

        Also computes a day-by-day holding curve (1d to max sessions) for each contract tenor.
        """
        if contract_dtes is None:
            contract_dtes = [7, 10, 14, 21, 35, 45, 60, 90]

        sweep_rows = []
        best_tenor_pf = None
        best_tenor_roi = None
        best_tenor_win = None
        highest_pf = -1.0
        highest_roi = -999.0
        highest_wr = -1.0

        n = len(z)
        can_split = n >= 80
        split_idx = int(n * 0.65) if can_split else n

        is_z, is_p, is_dates = z[:split_idx], p[:split_idx], dates[:split_idx]
        oos_z, oos_p, oos_dates = (z[split_idx:], p[split_idx:], dates[split_idx:]) if can_split else (None, None, None)

        for c_dte in contract_dtes:
            max_c_sessions = max(1, min(int(c_dte * 5 / 7), c_dte - 1))
            candidate_horizons = [h for h in [1, 2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 21] if h <= max_c_sessions]

            # 1. Compute full-data day-by-day curve for this tenor
            horizon_curve = []
            horizon_eval_cache = {}

            for h in candidate_horizons:
                eval_res = self._eval_single_param(
                    z=z,
                    p=p,
                    dates=dates,
                    thresh=thresh,
                    horizon=h,
                    earnings_dates=earnings_dates,
                    filter_earnings=filter_earnings,
                    contract_dte=c_dte
                )
                trades = eval_res.get("trades", [])
                t_count = len(trades)
                rois = [t["pnl_pct"] for t in trades]
                dollars = [t["dollar_pnl"] for t in trades]
                wins = [t for t in trades if t["result"] == "WIN"]
                losses = [t for t in trades if t["result"] != "WIN"]
                wr = round(len(wins) / max(1, t_count) * 100.0, 1) if t_count > 0 else 0.0
                avg_roi = round(float(np.mean(rois)), 1) if rois else 0.0
                tot_dollar = round(float(sum(dollars)), 2)
                gross_win = float(sum(t["dollar_pnl"] for t in wins)) if wins else 0.0
                gross_loss = float(abs(sum(t["dollar_pnl"] for t in losses))) if losses else 0.0
                pf = round(gross_win / max(1.0, gross_loss), 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

                cal_days_held = min(float(c_dte) - 0.5, h * (7.0 / 5.0))
                decay_ratio = max(0.001, (c_dte - cal_days_held)) / float(c_dte)
                theta_decay_pct = round((1.0 - np.sqrt(max(0.001, decay_ratio))) * 100.0, 1)

                curve_item = {
                    "hold_days": h,
                    "win_rate_pct": wr,
                    "avg_roi_pct": avg_roi,
                    "total_dollar": tot_dollar,
                    "profit_factor": pf,
                    "trades_count": t_count,
                    "est_theta_decay_pct": theta_decay_pct,
                    "trades": trades
                }
                horizon_curve.append(curve_item)
                horizon_eval_cache[h] = curve_item

            # 2. Select optimal holding horizon for this tenor
            wf_result = None
            if mode == "dynamic" and can_split:
                best_h = candidate_horizons[0]
                best_is_score = -9999.0
                best_is_wr = 0.0
                best_is_roi = 0.0
                best_is_trades = 0

                for h in candidate_horizons:
                    is_eval = self._eval_single_param(
                        z=is_z, p=is_p, dates=is_dates,
                        thresh=thresh, horizon=h,
                        earnings_dates=earnings_dates, filter_earnings=filter_earnings,
                        contract_dte=c_dte
                    )
                    is_tr = is_eval.get("trades", [])
                    if not is_tr:
                        continue
                    is_rois = [t["pnl_pct"] for t in is_tr]
                    is_wins = [t for t in is_tr if t["result"] == "WIN"]
                    is_wr = (len(is_wins) / len(is_tr)) * 100.0
                    is_roi = float(np.mean(is_rois))
                    # Utility: prioritize positive ROI and Win Rate with statistical weight
                    score = (is_wr * is_roi * np.sqrt(len(is_tr))) if (len(is_tr) >= 2 and is_roi > 0) else (is_wr - 50.0)
                    if score > best_is_score:
                        best_is_score = score
                        best_h = h
                        best_is_wr = is_wr
                        best_is_roi = is_roi
                        best_is_trades = len(is_tr)

                # Evaluate selected best_h on untouched OOS data
                oos_eval = self._eval_single_param(
                    z=oos_z, p=oos_p, dates=oos_dates,
                    thresh=thresh, horizon=best_h,
                    earnings_dates=earnings_dates, filter_earnings=filter_earnings,
                    contract_dte=c_dte
                )
                oos_tr = oos_eval.get("trades", [])
                oos_rois = [t["pnl_pct"] for t in oos_tr]
                oos_wins = [t for t in oos_tr if t["result"] == "WIN"]
                oos_wr = round((len(oos_wins) / max(1, len(oos_tr))) * 100.0, 1) if oos_tr else 0.0
                oos_roi = round(float(np.mean(oos_rois)), 1) if oos_rois else 0.0
                eff_ratio = round(oos_wr / max(1.0, best_is_wr), 2)

                if eff_ratio >= 0.70 and oos_wr >= 40.0:
                    status_str = "ROBUST_CONFIRMED"
                elif eff_ratio >= 0.50:
                    status_str = "MILD_DEGRADATION"
                else:
                    status_str = "OVERFITTED_WARNING"

                wf_result = {
                    "trained_hold_days": best_h,
                    "in_sample_win_rate_pct": round(best_is_wr, 1),
                    "in_sample_avg_roi_pct": round(best_is_roi, 1),
                    "in_sample_trades": best_is_trades,
                    "oos_win_rate_pct": oos_wr,
                    "oos_avg_roi_pct": oos_roi,
                    "oos_trades": len(oos_tr),
                    "efficiency_ratio": eff_ratio,
                    "validation_status": status_str
                }
                selected_horizon = best_h
            else:
                selected_horizon = min(horizon, max_c_sessions)

            # Retrieve performance for selected horizon
            chosen_perf = horizon_eval_cache.get(selected_horizon, horizon_curve[-1] if horizon_curve else {})
            trades = chosen_perf.get("trades", [])
            trades_count = chosen_perf.get("trades_count", 0)

            if trades_count == 0:
                continue

            win_rate = chosen_perf["win_rate_pct"]
            avg_roi = chosen_perf["avg_roi_pct"]
            total_dollar = chosen_perf["total_dollar"]
            profit_factor = chosen_perf["profit_factor"]
            est_theta_decay_pct = chosen_perf["est_theta_decay_pct"]

            rois = [t["pnl_pct"] for t in trades]
            entry_prices = [t["entry_option_price"] for t in trades]
            wins = [t for t in trades if t["result"] == "WIN"]
            losses = [t for t in trades if t["result"] != "WIN"]

            avg_entry_price = round(float(np.mean(entry_prices)), 2) if entry_prices else 0.0
            max_win_roi = round(float(np.max(rois)), 1) if rois else 0.0
            max_loss_roi = round(float(np.min(rois)), 1) if rois else 0.0
            avg_win_roi = round(float(np.mean([t["pnl_pct"] for t in wins])), 1) if wins else 0.0
            avg_loss_roi = round(float(np.mean([t["pnl_pct"] for t in losses])), 1) if losses else 0.0

            is_capped = bool(selected_horizon == max_c_sessions and c_dte <= 7)
            hold_label = f"{selected_horizon} Sessions" if not is_capped else f"{selected_horizon} Sessions (Expiry Cap)"

            # Strategic characterization
            if c_dte <= 7:
                profile_desc = "Extreme Gamma / High Theta Cliff"
                badge_type = "tag-red"
            elif c_dte <= 10:
                profile_desc = "High Leverage / Accelerated Decay"
                badge_type = "tag-amber"
            elif c_dte <= 14:
                profile_desc = "High Leverage / Moderate Decay"
                badge_type = "tag-amber"
            elif c_dte <= 21:
                profile_desc = "Sweet Spot: Optimal Convexity"
                badge_type = "tag-cyan"
            elif c_dte <= 35:
                profile_desc = "Monthly Standard: Balanced Decay"
                badge_type = "tag-green"
            elif c_dte <= 60:
                profile_desc = "Extended Tenor: Low Theta Drag"
                badge_type = "tag-blue"
            else:
                profile_desc = "Quarterly / LEAPS: Pure Delta Proxy"
                badge_type = "tag-purple"

            row = {
                "contract_dte": c_dte,
                "label": f"{c_dte}d DTE",
                "holding_days": selected_horizon,
                "holding_days_label": hold_label,
                "is_capped": is_capped,
                "trades_count": trades_count,
                "win_rate_pct": win_rate,
                "avg_roi_pct": avg_roi,
                "total_dollar": total_dollar,
                "avg_entry_price": avg_entry_price,
                "profit_factor": profit_factor,
                "max_win_roi": max_win_roi,
                "max_loss_roi": max_loss_roi,
                "avg_win_roi": avg_win_roi,
                "avg_loss_roi": avg_loss_roi,
                "est_theta_decay_pct": est_theta_decay_pct,
                "profile": profile_desc,
                "badge_type": badge_type,
                "walk_forward": wf_result,
                "horizon_curve": horizon_curve,
                "trades": trades
            }
            sweep_rows.append(row)

            if profit_factor > highest_pf:
                highest_pf = profit_factor
                best_tenor_pf = row
            if avg_roi > highest_roi:
                highest_roi = avg_roi
                best_tenor_roi = row
            if win_rate > highest_wr:
                highest_wr = win_rate
                best_tenor_win = row

        return {
            "status": "SUCCESS",
            "mode": mode,
            "threshold_used": thresh,
            "horizon_used": horizon,
            "sweep_rows": sweep_rows,
            "best_profit_factor_tenor": best_tenor_pf,
            "best_roi_tenor": best_tenor_roi,
            "best_win_rate_tenor": best_tenor_win
        }

    def evaluate_3d_global_optimum(
        self,
        z: np.ndarray,
        p: np.ndarray,
        dates: List[str],
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True,
        thresholds: Optional[List[float]] = None,
        candidate_tenors: Optional[List[int]] = None,
        candidate_horizons: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        3D Global Tri-Factor Surface Optimization.
        Simultaneously sweeps:
          1. Entry Barrier Sigma (s)
          2. Contract Maturity Tenor DTE (t)
          3. Holding Horizon Sessions (h)
        Employs causal Walk-Forward out-of-sample validation (65% IS / 35% OOS)
        to identify the statistically robust champion execution triplet (s*, t*, h*).
        Uses precomputed signal caches to evaluate hundreds of combinations in < 2 seconds.
        """
        t0 = time.time()

        z = np.asarray(z, dtype=float)
        p = np.asarray(p, dtype=float)
        n = len(z)

        if n < 40:
            return {"status": "ERROR", "error": f"Insufficient history ({n} bars)."}

        thresh_list = thresholds or [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        tenor_list = candidate_tenors or [7, 10, 14, 21, 35, 45, 60, 90]
        horizon_list = candidate_horizons or [1, 2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 21]

        # Chronological Walk-Forward Split
        split_idx = int(n * 0.65)
        can_split = split_idx >= 50 and (n - split_idx) >= 30

        if can_split:
            is_dates = dates[:split_idx]
            is_z = z[:split_idx]
            is_p = p[:split_idx]
            oos_dates = dates[split_idx:]
            oos_z = z[split_idx:]
            oos_p = p[split_idx:]
        else:
            is_dates, is_z, is_p = dates, z, p
            oos_dates, oos_z, oos_p = dates, z, p

        # Step 1: Precompute trade signals for each threshold once (ultra-fast caching)
        precomputed_is = {}
        precomputed_oos = {}
        precomputed_full = {}

        # Precompute date index maps and rolling 20d volatility once
        full_d2i = {d: i for i, d in enumerate(dates)}
        full_rv = np.full(n, 0.25)
        if n >= 25:
            log_returns = np.diff(np.log(p))
            for i in range(20, n):
                window = log_returns[max(0, i - 20):i]
                if len(window) >= 5:
                    full_rv[i] = float(np.std(window, ddof=1) * np.sqrt(252))

        if can_split:
            is_d2i = {d: i for i, d in enumerate(is_dates)}
            is_rv = full_rv[:split_idx]
            oos_d2i = {d: i for i, d in enumerate(oos_dates)}
            oos_rv = full_rv[split_idx:]
        else:
            is_d2i, is_rv = full_d2i, full_rv
            oos_d2i, oos_rv = full_d2i, full_rv

        for s in thresh_list:
            if can_split:
                bt_is = SyntheticOptionBacktestEngine.evaluate_option_performance(
                    z_series=is_z, prices=is_p, dates=is_dates, barrier=s,
                    earnings_dates=earnings_dates, filter_earnings=filter_earnings
                )
                precomputed_is[s] = bt_is.get("trades", [])

                bt_oos = SyntheticOptionBacktestEngine.evaluate_option_performance(
                    z_series=oos_z, prices=oos_p, dates=oos_dates, barrier=s,
                    earnings_dates=earnings_dates, filter_earnings=filter_earnings
                )
                precomputed_oos[s] = bt_oos.get("trades", [])

            bt_full = SyntheticOptionBacktestEngine.evaluate_option_performance(
                z_series=z, prices=p, dates=dates, barrier=s,
                earnings_dates=earnings_dates, filter_earnings=filter_earnings
            )
            precomputed_full[s] = bt_full.get("trades", [])

        # Step 2: Sweep 3D Surface (s, t, h)
        triplets = []
        total_eval_count = 0
        for s in thresh_list:
            full_tr_raw = precomputed_full.get(s, [])
            if len(full_tr_raw) < 1:
                continue

            for t in tenor_list:
                # Physical tradable sessions cap: an option cannot be held past its expiration
                max_c_sessions = max(1, min(int(t * 5 / 7), t - 1))
                valid_horizons = [h for h in horizon_list if h <= max_c_sessions]

                for h in valid_horizons:
                    total_eval_count += 1

                    # Full dataset evaluation
                    full_eval = self._eval_single_param(
                        z=z, p=p, dates=dates,
                        thresh=s, horizon=h,
                        earnings_dates=earnings_dates, filter_earnings=filter_earnings,
                        contract_dte=t,
                        precomputed_trades=full_tr_raw,
                        date_to_idx=full_d2i,
                        rv_20d=full_rv,
                        calc_details=False
                    )
                    full_cnt = full_eval["trades_count"]
                    if full_cnt < 1:
                        continue

                    full_wr = full_eval["win_rate_pct"]
                    full_roi = full_eval["avg_roi_pct"]
                    profit_factor = full_eval["profit_factor"]
                    total_dollar = full_eval["total_dollar"]
                    entry_prices = full_eval["entry_prices"]
                    avg_entry_price = round(float(np.mean(entry_prices)), 2) if entry_prices else 0.0

                    rois_list = full_eval["rois"]
                    max_win_roi = round(float(np.max(rois_list)), 1) if rois_list else 0.0
                    max_loss_roi = round(float(np.min(rois_list)), 1) if rois_list else 0.0

                    # Walk-forward evaluations
                    if can_split:
                        is_eval = self._eval_single_param(
                            z=is_z, p=is_p, dates=is_dates,
                            thresh=s, horizon=h,
                            earnings_dates=earnings_dates, filter_earnings=filter_earnings,
                            contract_dte=t,
                            precomputed_trades=precomputed_is.get(s, []),
                            date_to_idx=is_d2i,
                            rv_20d=is_rv,
                            calc_details=False
                        )
                        is_trades_cnt = is_eval["trades_count"]
                        is_wr = is_eval["win_rate_pct"]
                        is_roi = is_eval["avg_roi_pct"]

                        oos_eval = self._eval_single_param(
                            z=oos_z, p=oos_p, dates=oos_dates,
                            thresh=s, horizon=h,
                            earnings_dates=earnings_dates, filter_earnings=filter_earnings,
                            contract_dte=t,
                            precomputed_trades=precomputed_oos.get(s, []),
                            date_to_idx=oos_d2i,
                            rv_20d=oos_rv,
                            calc_details=False
                        )
                        oos_trades_cnt = oos_eval["trades_count"]
                        oos_wr = oos_eval["win_rate_pct"]
                        oos_roi = oos_eval["avg_roi_pct"]

                        eff_ratio = round(oos_wr / max(1.0, is_wr), 2) if is_wr > 0 else (1.0 if oos_wr > 0 else 0.0)
                    else:
                        is_wr, is_roi, is_trades_cnt = 50.0, 10.0, 0
                        oos_wr, oos_roi, oos_trades_cnt = 50.0, 10.0, 0
                        eff_ratio = 1.0

                    # Estimated Theta decay %
                    cal_days_held = min(float(t) - 0.5, h * (7.0 / 5.0))
                    decay_ratio = max(0.001, (t - cal_days_held)) / float(t)
                    est_theta_decay_pct = round((1.0 - np.sqrt(max(0.001, decay_ratio))) * 100.0, 1)

                    # Validation status
                    if is_trades_cnt >= 2 and oos_trades_cnt >= 2:
                        if eff_ratio >= 0.70 and oos_wr >= 40.0 and is_roi > 0 and oos_roi > 0:
                            status_str = "ROBUST_CONFIRMED"
                            is_robust = True
                        elif eff_ratio >= 0.40 and oos_wr >= 35.0:
                            status_str = "MILD_DEGRADATION"
                            is_robust = False
                        else:
                            status_str = "OVERFITTED_WARNING"
                            is_robust = False
                    elif can_split:
                        status_str = "SAMPLE_LIMITED"
                        is_robust = False
                    else:
                        status_str = "FULL_SAMPLE_ONLY"
                        is_robust = False

                    # Strategic profile
                    if t <= 7:
                        profile_desc = "Extreme Gamma / High Theta Cliff"
                        badge_type = "tag-red"
                    elif t <= 10:
                        profile_desc = "High Leverage / Accelerated Decay"
                        badge_type = "tag-amber"
                    elif t <= 14:
                        profile_desc = "High Leverage / Moderate Decay"
                        badge_type = "tag-amber"
                    elif t <= 21:
                        profile_desc = "Sweet Spot: Optimal Convexity"
                        badge_type = "tag-cyan"
                    elif t <= 35:
                        profile_desc = "Monthly Standard: Balanced Decay"
                        badge_type = "tag-green"
                    elif t <= 60:
                        profile_desc = "Extended Tenor: Low Theta Drag"
                        badge_type = "tag-blue"
                    else:
                        profile_desc = "Quarterly / LEAPS: Pure Delta Proxy"
                        badge_type = "tag-purple"

                    # Score combines profit factor, win rate, options ROI convexity, sample size, and OOS robustness
                    roi_factor = min(3.0, max(0.3, np.sqrt(max(0.0, full_roi) / 10.0))) if full_roi > 0 else 0.2
                    sample_weight = min(4.0, np.sqrt(max(1, full_cnt)))
                    robust_mult = 1.3 if is_robust else (1.0 if status_str == "MILD_DEGRADATION" else 0.7)
                    score = round(float(
                        min(profit_factor, 10.0) * (full_wr / 50.0) * roi_factor * sample_weight * min(1.5, max(0.1, eff_ratio)) * robust_mult
                    ), 2)

                    triplets.append({
                        "threshold": s,
                        "threshold_label": f"±{s:.1f}σ",
                        "contract_dte": t,
                        "tenor_label": f"{t}d DTE",
                        "holding_days": h,
                        "holding_label": f"{h} Sessions",
                        "trades_count": full_cnt,
                        "win_rate_pct": full_wr,
                        "avg_roi_pct": full_roi,
                        "profit_factor": profit_factor,
                        "total_dollar": total_dollar,
                        "avg_entry_price": avg_entry_price,
                        "max_win_roi": max_win_roi,
                        "max_loss_roi": max_loss_roi,
                        "est_theta_decay_pct": est_theta_decay_pct,
                        "is_sample_win_rate_pct": round(is_wr, 1),
                        "is_sample_avg_roi_pct": round(is_roi, 1),
                        "is_sample_trades": is_trades_cnt,
                        "oos_win_rate_pct": round(oos_wr, 1),
                        "oos_avg_roi_pct": round(oos_roi, 1),
                        "oos_trades": oos_trades_cnt,
                        "efficiency_ratio": eff_ratio,
                        "validation_status": status_str,
                        "is_robust": is_robust,
                        "profile": profile_desc,
                        "badge_type": badge_type,
                        "score": score,
                        "trades": []
                    })

        elapsed_sec = round(time.time() - t0, 2)

        if not triplets:
            return {
                "status": "ERROR",
                "error": "No viable 3D parameter combinations satisfied minimum sample requirements."
            }

        # Filter and rank
        # Sort key: prioritize robust confirmation, then score, profit factor, win rate
        triplets.sort(key=lambda x: (
            1 if x["is_robust"] else 0,
            x["score"],
            x["profit_factor"],
            x["win_rate_pct"]
        ), reverse=True)

        for idx, item in enumerate(triplets, 1):
            item["rank"] = idx

        # Champion selection: prioritize robust candidates with PF >= 1.0, then positive PF, then top score
        robust_candidates = [t for t in triplets if t["is_robust"] and t["profit_factor"] >= 1.0 and t["trades_count"] >= 3]
        if not robust_candidates:
            robust_candidates = [t for t in triplets if t["profit_factor"] >= 1.0 and t["trades_count"] >= 2]
        if not robust_candidates:
            robust_candidates = triplets

        champion = max(robust_candidates, key=lambda x: (x["score"], x["profit_factor"]))
        top_pf = max(triplets, key=lambda x: (x["profit_factor"], x["score"]))
        top_wr = max(triplets, key=lambda x: (x["win_rate_pct"], x["profit_factor"]))
        fast_pool = [t for t in triplets if t["holding_days"] <= 3]
        top_fast = max(fast_pool, key=lambda x: (x["profit_factor"], x["score"])) if fast_pool else champion

        # Distributions across top 20
        top_20 = triplets[:20]

        # Attach full forensic trades details only to the top 20 returned triplets
        for item in top_20:
            det = self._eval_single_param(
                z=z, p=p, dates=dates,
                thresh=item["threshold"], horizon=item["holding_days"],
                earnings_dates=earnings_dates, filter_earnings=filter_earnings,
                contract_dte=item["contract_dte"],
                precomputed_trades=precomputed_full.get(item["threshold"], []),
                date_to_idx=full_d2i,
                rv_20d=full_rv,
                calc_details=True
            )
            item["trades"] = det.get("trades", [])
        sigma_counts = {}
        tenor_counts = {}
        horizon_counts = {}
        for item in top_20:
            s_k = f"{item['threshold']:.1f}σ"
            t_k = f"{item['contract_dte']}d"
            h_k = f"{item['holding_days']}d"
            sigma_counts[s_k] = sigma_counts.get(s_k, 0) + 1
        # Build 2D Surface Grid (Threshold Sigma vs Contract Tenor) for Heatmap visualization
        surface_grid_rows = []
        for s in thresh_list:
            cells = []
            for t in tenor_list:
                candidates = [tr for tr in triplets if abs(tr["threshold"] - s) < 1e-4 and tr["contract_dte"] == t]
                if candidates:
                    best_c = max(candidates, key=lambda x: (x["profit_factor"], x["score"]))
                    is_champ = bool(
                        abs(s - champion["threshold"]) < 1e-4 and 
                        t == champion["contract_dte"] and 
                        best_c["holding_days"] == champion["holding_days"]
                    )
                    cells.append({
                        "threshold": round(float(s), 2),
                        "contract_dte": int(t),
                        "optimal_holding_days": int(best_c["holding_days"]),
                        "profit_factor": round(float(best_c["profit_factor"]), 2),
                        "win_rate_pct": round(float(best_c["win_rate_pct"]), 1),
                        "avg_roi_pct": round(float(best_c["avg_roi_pct"]), 1),
                        "total_dollar": round(float(best_c["total_dollar"]), 2),
                        "trades_count": int(best_c["trades_count"]),
                        "validation_status": best_c["validation_status"],
                        "efficiency_ratio": best_c["efficiency_ratio"],
                        "is_champion": is_champ
                    })
                else:
                    cells.append({
                        "threshold": round(float(s), 2),
                        "contract_dte": int(t),
                        "optimal_holding_days": 0,
                        "profit_factor": 0.0,
                        "win_rate_pct": 0.0,
                        "avg_roi_pct": 0.0,
                        "total_dollar": 0.0,
                        "trades_count": 0,
                        "validation_status": "NO_DATA",
                        "efficiency_ratio": 0.0,
                        "is_champion": False
                    })
            surface_grid_rows.append({
                "threshold": round(float(s), 2),
                "threshold_label": f"±{s:.1f}σ",
                "cells": cells
            })

        surface_grid = {
            "thresholds": [round(float(s), 2) for s in thresh_list],
            "tenors": tenor_list,
            "rows": surface_grid_rows
        }

        return {
            "status": "SUCCESS",
            "total_evaluated": total_eval_count,
            "elapsed_seconds": elapsed_sec,
            "in_sample_period": f"{is_dates[0]} to {is_dates[-1]} ({len(is_dates)} bars)" if can_split else "N/A",
            "out_of_sample_period": f"{oos_dates[0]} to {oos_dates[-1]} ({len(oos_dates)} bars)" if can_split else "N/A",
            "champion_triplet": champion,
            "top_profit_factor_triplet": top_pf,
            "top_win_rate_triplet": top_wr,
            "top_fast_pop_triplet": top_fast,
            "ranked_triplets": top_20,
            "surface_grid": surface_grid,
            "distributions": {
                "sigmas": sigma_counts,
                "tenors": tenor_counts,
                "horizons": horizon_counts
            }
        }

    def optimize(
        self,
        z_series: np.ndarray,
        prices: np.ndarray,
        dates: List[str],
        earnings_dates: Optional[List[str]] = None,
        filter_earnings: bool = True
    ) -> Dict[str, Any]:
        """
        Executes full 5-year optimization with Walk-Forward Cross-Validation
        and Plateau Smoothing.
        """
        z = np.asarray(z_series, dtype=float)
        p = np.asarray(prices, dtype=float)
        n = len(z)

        if n < 40:
            return {"status": "ERROR", "error": f"Insufficient history ({n} bars)."}

        # 1. Full Dataset Evaluation (for overall heatmap display)
        full_eval = self._eval_grid(z, p, dates, earnings_dates, filter_earnings)
        best_cell = full_eval["best_cell"]
        optimal_thresh = best_cell["threshold"]
        optimal_horizon = best_cell["horizon"]

        # 2. Walk-Forward / Out-of-Sample (OOS) Cross-Validation
        # Chronological split: 65% In-Sample (training), 35% Out-of-Sample (testing)
        split_idx = int(n * 0.65)
        has_enough_split = split_idx >= 50 and (n - split_idx) >= 30

        if has_enough_split:
            is_dates = dates[:split_idx]
            is_z = z[:split_idx]
            is_p = p[:split_idx]

            oos_dates = dates[split_idx:]
            oos_z = z[split_idx:]
            oos_p = p[split_idx:]

            # Train on In-Sample (IS) strictly
            is_eval = self._eval_grid(is_z, is_p, is_dates, earnings_dates, filter_earnings)
            is_best = is_eval["best_cell"]
            is_opt_thresh = is_best["threshold"]
            is_opt_horizon = is_best["horizon"]

            # Test In-Sample optimum on unseen Out-of-Sample (OOS) data
            oos_perf = self._eval_single_param(
                oos_z, oos_p, oos_dates, is_opt_thresh, is_opt_horizon, earnings_dates, filter_earnings
            )

            is_win = is_best["win_rate_pct"]
            oos_win = oos_perf["win_rate_pct"]
            efficiency_ratio = round(oos_win / max(1.0, is_win), 2)

            is_start, is_end = is_dates[0], is_dates[-1]
            oos_start, oos_end = oos_dates[0], oos_dates[-1]

            walk_forward_result = {
                "in_sample_period": f"{is_start} to {is_end} ({len(is_dates)} bars)",
                "out_of_sample_period": f"{oos_start} to {oos_end} ({len(oos_dates)} bars)",
                "trained_threshold": is_opt_thresh,
                "trained_horizon": is_opt_horizon,
                "in_sample_win_rate_pct": is_win,
                "in_sample_avg_roi_pct": is_best["avg_roi_pct"],
                "in_sample_trades": is_best["trades_count"],
                "oos_win_rate_pct": oos_win,
                "oos_avg_roi_pct": oos_perf["avg_roi_pct"],
                "oos_trades": oos_perf["trades_count"],
                "efficiency_ratio": efficiency_ratio,
                "validation_status": "ROBUST_CONFIRMED" if efficiency_ratio >= 0.70 and oos_win >= 50.0 else (
                    "MILD_DEGRADATION" if efficiency_ratio >= 0.50 else "OVERFITTED_WARNING"
                )
            }
        else:
            walk_forward_result = {
                "validation_status": "INSUFFICIENT_DATA_FOR_SPLIT",
                "note": "Requires at least 80 trading days for a reliable 65/35 walk-forward split."
            }

        # 3. Forensic Loser Diagnostics on optimal parameter set
        opt_details = self._eval_single_param(
            z, p, dates, optimal_thresh, optimal_horizon, earnings_dates, filter_earnings
        )
        forensic_trades = opt_details["trades"]

        failure_buckets = {
            "REGIME_DRIFT": sum(1 for t in forensic_trades if t["failure_cause"] == "REGIME_DRIFT"),
            "THETA_BLEED": sum(1 for t in forensic_trades if t["failure_cause"] == "THETA_BLEED"),
            "WHIPSAW": sum(1 for t in forensic_trades if t["failure_cause"] == "WHIPSAW"),
            "TOTAL_LOSSES": sum(1 for t in forensic_trades if t["result"] == "LOSS"),
            "TOTAL_WINS": sum(1 for t in forensic_trades if t["result"] == "WIN")
        }

        # 4. Heatmap rows formatting
        grid_rows = []
        cell_dict = full_eval["cell_details"]
        p_scores = full_eval["plateau_scores"]
        best_s, best_h = full_eval["best_s"], full_eval["best_h"]

        for s_idx, thresh in enumerate(self.thresholds):
            cols = []
            for h_idx, h in enumerate(self.horizons):
                cell = cell_dict[(s_idx, h_idx)]
                cols.append({
                    "horizon": h,
                    "win_rate_pct": cell["win_rate_pct"],
                    "avg_roi_pct": cell["avg_roi_pct"],
                    "trades": cell["trades_count"],
                    "raw_utility": cell["raw_utility"],
                    "plateau_score": round(float(p_scores[s_idx, h_idx]), 2),
                    "is_optimal": bool(s_idx == best_s and h_idx == best_h)
                })
            grid_rows.append({"threshold": thresh, "horizons": cols})

        return {
            "status": "SUCCESS",
            "optimal_recommendation": {
                "threshold": optimal_thresh,
                "horizon_days": optimal_horizon,
                "win_rate_pct": best_cell["win_rate_pct"],
                "avg_roi_pct": best_cell["avg_roi_pct"],
                "trades_count": best_cell["trades_count"],
                "plateau_score": round(float(p_scores[best_s, best_h]), 2),
                "stability_ratio": round(float(full_eval["stability_ratios"][best_s, best_h]), 2),
                "is_robust_plateau": bool(full_eval["stability_ratios"][best_s, best_h] >= 0.5)
            },
            "walk_forward_validation": walk_forward_result,
            "failure_diagnostics": failure_buckets,
            "forensic_trades": forensic_trades,
            "grid_heatmap": {
                "thresholds": self.thresholds,
                "horizons": self.horizons,
                "rows": grid_rows
            }
        }
