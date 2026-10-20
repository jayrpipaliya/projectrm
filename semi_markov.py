"""
PROJECT RM: SEMI-MARKOV REGIME MODEL
======================================
Fits a discrete-state Semi-Markov Model over the Kalman Z-score time series
produced by ResidualAnomalyModel.  Each (stock, market, sector) triple gets its
own model capturing:

  1. Transition matrix P  [3 × 3]  — where the regime goes next
  2. Per-state sojourn distribution — how long the regime lasts
     • Phase 1: Geometric  (memoryless, fast MLE)
     • Phase 2: Weibull    (duration-dependent hazard, selected by AIC)
  3. Survival statistics for the CURRENT run:
     • days_in_regime        – consecutive trading days in current state
     • exit_prob_1d / 3d / 5d – P(state change within k days | survived N days)
     • mean_residual_life    – E[days until exit | survived N days]

States
------
  0  UNDERPERFORMING   z ≤  -threshold
  1  NEUTRAL           -threshold < z < +threshold
  2  OUTPERFORMING     z ≥  +threshold

Usage
-----
    from semi_markov import SemiMarkovFitter, label_states

    # z_series: np.ndarray of z_score_1d values (chronological)
    smf = SemiMarkovFitter(threshold=1.8)
    result = smf.fit(z_series)

    # result keys:
    #   transition_matrix, state_names, sojourn_fits,
    #   current_state, days_in_regime,
    #   exit_prob_1d, exit_prob_3d, exit_prob_5d, mean_residual_life,
    #   n_transitions, n_observations, fit_quality
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_NAMES = {0: "UNDERPERFORMING", 1: "NEUTRAL", 2: "OUTPERFORMING"}
MIN_SOJOURN_SAMPLES = 5          # minimum runs needed to fit a distribution
MAX_MRL_DAYS = 60               # cap on mean residual life output (safety)


# ---------------------------------------------------------------------------
# State Labeling
# ---------------------------------------------------------------------------
def label_states(z_series: np.ndarray, threshold: float = 1.8) -> np.ndarray:
    """
    Convert a Z-score time series to a discrete state sequence.

    Returns np.ndarray of int8:
        0 = UNDERPERFORMING  (z <= -threshold)
        1 = NEUTRAL
        2 = OUTPERFORMING    (z >= +threshold)
    """
    states = np.ones(len(z_series), dtype=np.int8)   # default: NEUTRAL
    states[z_series <= -threshold] = 0
    states[z_series >= threshold]  = 2
    return states


# ---------------------------------------------------------------------------
# Sojourn Run Extraction
# ---------------------------------------------------------------------------
def extract_sojourn_runs(states: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract consecutive runs from a state sequence.

    Returns list of (state, duration) tuples, one per run.
    Example: [0,0,1,1,1,2] → [(0,2), (1,3), (2,1)]
    """
    if len(states) == 0:
        return []
    runs = []
    cur_state = int(states[0])
    cur_len = 1
    for s in states[1:]:
        s = int(s)
        if s == cur_state:
            cur_len += 1
        else:
            runs.append((cur_state, cur_len))
            cur_state = s
            cur_len = 1
    runs.append((cur_state, cur_len))
    return runs


# ---------------------------------------------------------------------------
# Sojourn Distribution Fitters
# ---------------------------------------------------------------------------
class GeometricFit:
    """
    Geometric (memoryless) sojourn distribution.
    P(T = t) = (1-p)^(t-1) * p
    MLE: p = 1 / mean(durations)
    """
    dist_name = "geometric"

    def __init__(self):
        self.p = 0.5          # exit probability per day
        self.n_samples = 0

    def fit(self, durations: List[int]) -> "GeometricFit":
        """Fit from list of observed sojourn lengths."""
        if not durations:
            return self
        self.n_samples = len(durations)
        mean_dur = float(np.mean(durations))
        # MLE for geometric: p = 1/mean
        self.p = max(1e-4, min(1.0 - 1e-4, 1.0 / mean_dur))
        return self

    def survival(self, t: int) -> float:
        """P(T > t) = (1-p)^t"""
        if t <= 0:
            return 1.0
        return float((1.0 - self.p) ** t)

    def exit_prob(self, days_already: int, k: int) -> float:
        """P(exit within next k days | survived days_already days)."""
        S_now = self.survival(days_already)
        if S_now < 1e-10:
            return 1.0
        S_future = self.survival(days_already + k)
        return float((S_now - S_future) / S_now)

    def mean_residual_life(self, days_already: int) -> float:
        """E[T - days_already | T > days_already] for geometric = 1/p always."""
        # Geometric is memoryless — MRL is always 1/p regardless of days_already
        return float(min(MAX_MRL_DAYS, 1.0 / self.p))

    def aic(self, durations: List[int]) -> float:
        """AIC = 2k - 2*log_likelihood (k=1 param for geometric)."""
        if not durations or self.p <= 0:
            return 1e9
        log_liks = [(t - 1) * np.log(1 - self.p) + np.log(self.p) for t in durations]
        return 2.0 * 1 - 2.0 * sum(log_liks)

    def to_dict(self) -> Dict:
        return {"dist": self.dist_name, "p": round(self.p, 6), "mean_duration": round(1.0 / self.p, 2), "n_samples": self.n_samples}


class WeibullFit:
    """
    Discrete Weibull sojourn distribution (Nakagawa & Osaki 1975 parameterization).
    Survival: S(t) = q^(t^beta)    q in (0,1), beta > 0
    beta < 1 → decreasing hazard (long streaks become MORE stable)
    beta > 1 → increasing hazard (long streaks become LESS stable → mean-reversion)
    beta = 1 → geometric (memoryless)

    Parameters fitted by minimizing negative log-likelihood via grid search
    (avoids scipy dependency on the discrete form).
    """
    dist_name = "weibull"

    def __init__(self):
        self.q = 0.5
        self.beta = 1.0
        self.n_samples = 0

    def _log_likelihood(self, durations: List[int], q: float, beta: float) -> float:
        """Log-likelihood for discrete Weibull."""
        if q <= 0 or q >= 1 or beta <= 0:
            return -1e9
        ll = 0.0
        for t in durations:
            # P(T=t) = S(t-1) - S(t) = q^((t-1)^beta) - q^(t^beta)
            s_prev = q ** ((t - 1) ** beta) if t > 1 else 1.0
            s_curr = q ** (t ** beta)
            prob = s_prev - s_curr
            if prob < 1e-12:
                prob = 1e-12
            ll += np.log(prob)
        return ll

    def fit(self, durations: List[int]) -> "WeibullFit":
        """Grid search MLE over (q, beta)."""
        if not durations:
            return self
        self.n_samples = len(durations)

        best_ll = -1e9
        best_q, best_beta = 0.5, 1.0

        # Coarse grid: q in (0,1), beta in (0.2, 4.0)
        q_vals = np.linspace(0.05, 0.95, 19)
        beta_vals = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0])

        for q in q_vals:
            for beta in beta_vals:
                ll = self._log_likelihood(durations, float(q), float(beta))
                if ll > best_ll:
                    best_ll = ll
                    best_q = float(q)
                    best_beta = float(beta)

        # Fine grid around best
        q_fine = np.linspace(max(0.01, best_q - 0.1), min(0.99, best_q + 0.1), 21)
        beta_fine = np.linspace(max(0.1, best_beta - 0.5), best_beta + 0.5, 21)

        for q in q_fine:
            for beta in beta_fine:
                ll = self._log_likelihood(durations, float(q), float(beta))
                if ll > best_ll:
                    best_ll = ll
                    best_q = float(q)
                    best_beta = float(beta)

        self.q = best_q
        self.beta = best_beta
        return self

    def survival(self, t: int) -> float:
        """S(t) = q^(t^beta)"""
        if t <= 0:
            return 1.0
        return float(self.q ** (t ** self.beta))

    def exit_prob(self, days_already: int, k: int) -> float:
        """P(exit within next k days | survived days_already days)."""
        S_now = self.survival(days_already)
        if S_now < 1e-10:
            return 1.0
        S_future = self.survival(days_already + k)
        return float((S_now - S_future) / S_now)

    def mean_residual_life(self, days_already: int, max_horizon: int = 500) -> float:
        """E[T - days_already | T > days_already] computed numerically."""
        S_now = self.survival(days_already)
        if S_now < 1e-10:
            return 0.0
        # MRL = sum_{u > days_already} S(u) / S(days_already)
        total = 0.0
        for u in range(days_already + 1, days_already + max_horizon):
            s = self.survival(u)
            if s < 1e-10:
                break
            total += s
        return float(min(MAX_MRL_DAYS, total / S_now))

    def aic(self, durations: List[int]) -> float:
        """AIC = 2k - 2*LL (k=2 params for Weibull)."""
        ll = self._log_likelihood(durations, self.q, self.beta)
        return 2.0 * 2 - 2.0 * ll

    def to_dict(self) -> Dict:
        return {
            "dist": self.dist_name,
            "q": round(self.q, 6),
            "beta": round(self.beta, 4),
            "hazard_shape": "increasing" if self.beta > 1.05 else ("decreasing" if self.beta < 0.95 else "constant"),
            "n_samples": self.n_samples
        }


def _best_fit(durations: List[int]) -> Any:
    """Fit both Geometric and Weibull; return whichever has lower AIC."""
    geo = GeometricFit().fit(durations)
    if len(durations) < MIN_SOJOURN_SAMPLES:
        return geo   # Not enough data for 2-param Weibull

    wei = WeibullFit().fit(durations)
    if wei.aic(durations) < geo.aic(durations) - 2.0:  # 2-unit AIC penalty for extra param
        return wei
    return geo


# ---------------------------------------------------------------------------
# Transition Matrix Estimation
# ---------------------------------------------------------------------------
def _estimate_transition_matrix(runs: List[Tuple[int, int]], n_states: int = 3) -> np.ndarray:
    """
    Count state-to-state transitions (from run i to run i+1).
    Returns row-normalized P matrix (row i = P(go to j | in state i)).
    Rows with zero transitions are initialized to uniform.
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(runs) - 1):
        from_state = runs[i][0]
        to_state   = runs[i + 1][0]
        counts[from_state, to_state] += 1.0

    P = np.zeros_like(counts)
    for i in range(n_states):
        row_sum = counts[i].sum()
        if row_sum > 0:
            P[i] = counts[i] / row_sum
        else:
            # No observed transitions from this state → uniform fallback
            P[i] = np.ones(n_states) / n_states
        P[i, i] = 0.0   # Self-transitions don't count in semi-Markov
        row_sum2 = P[i].sum()
        if row_sum2 > 0:
            P[i] /= row_sum2
        else:
            P[i] = np.ones(n_states) / n_states
            P[i, i] = 0.0
            if P[i].sum() > 0:
                P[i] /= P[i].sum()

    return P


# ---------------------------------------------------------------------------
# Main Fitter Class
# ---------------------------------------------------------------------------
class SemiMarkovFitter:
    """
    Fit a Semi-Markov Model from a Z-score time series.

    Parameters
    ----------
    threshold : float
        Z-score magnitude that defines regime boundaries.
        Default 1.8 (matches existing ResidualAnomalyModel convention).
    """

    def __init__(self, threshold: float = 1.8):
        self.threshold = threshold
        # Fitted attributes (set after .fit())
        self.transition_matrix: Optional[np.ndarray] = None
        self.sojourn_fits: Dict[int, Any] = {}
        self.runs: List[Tuple[int, int]] = []
        self.state_sequence: Optional[np.ndarray] = None

    def fit(self, z_series: np.ndarray) -> Dict[str, Any]:
        """
        Fit the Semi-Markov Model from a Z-score time series.

        Parameters
        ----------
        z_series : np.ndarray
            1-D array of z_score_1d values, oldest first.

        Returns
        -------
        dict with keys:
            transition_matrix       – list-of-lists  [3][3]
            state_names             – {0: ..., 1: ..., 2: ...}
            sojourn_fits            – {state: dict}
            current_state           – int  (0/1/2)
            current_state_name      – str
            days_in_regime          – int
            exit_prob_1d            – float  (0–1)
            exit_prob_3d            – float  (0–1)
            exit_prob_5d            – float  (0–1)
            mean_residual_life      – float  (days)
            n_transitions           – int
            n_observations          – int
            fit_quality             – str  ("GOOD" / "LIMITED" / "INSUFFICIENT")
        """
        if z_series is None or len(z_series) < 10:
            return self._insufficient_result("Z-series too short (<10 observations)")

        z = np.asarray(z_series, dtype=float)
        self.state_sequence = label_states(z, self.threshold)
        self.runs = extract_sojourn_runs(self.state_sequence)

        if len(self.runs) < 3:
            return self._insufficient_result("Too few regime transitions to fit model")

        # Build sojourn duration lists per state
        sojourn_by_state: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        for state, dur in self.runs[:-1]:   # Exclude last run (may be censored)
            sojourn_by_state[state].append(dur)

        # Fit sojourn distributions
        self.sojourn_fits = {}
        for s in range(3):
            durs = sojourn_by_state[s]
            if durs:
                self.sojourn_fits[s] = _best_fit(durs)
            else:
                # No observed runs for this state → geometric with p=0.3 (3.3-day default)
                geo = GeometricFit()
                geo.p = 0.3
                geo.n_samples = 0
                self.sojourn_fits[s] = geo

        # Estimate transition matrix
        self.transition_matrix = _estimate_transition_matrix(self.runs)

        # Current state info (last run)
        cur_state, days_in = self.runs[-1]
        fit = self.sojourn_fits[cur_state]

        ep1  = fit.exit_prob(days_in, 1)
        ep3  = fit.exit_prob(days_in, 3)
        ep5  = fit.exit_prob(days_in, 5)
        mrl  = fit.mean_residual_life(days_in)

        # Fit quality
        n_trans = len(self.runs) - 1
        if n_trans >= 20:
            quality = "GOOD"
        elif n_trans >= 8:
            quality = "LIMITED"
        else:
            quality = "INSUFFICIENT"

        return {
            "transition_matrix": self.transition_matrix.tolist(),
            "state_names": STATE_NAMES,
            "sojourn_fits": {s: fit.to_dict() for s, fit in self.sojourn_fits.items()},
            "current_state": int(cur_state),
            "current_state_name": STATE_NAMES[cur_state],
            "days_in_regime": int(days_in),
            "exit_prob_1d": round(ep1, 4),
            "exit_prob_3d": round(ep3, 4),
            "exit_prob_5d": round(ep5, 4),
            "mean_residual_life": round(mrl, 2),
            "n_transitions": int(n_trans),
            "n_observations": int(len(z)),
            "fit_quality": quality,
        }

    def _insufficient_result(self, reason: str) -> Dict[str, Any]:
        return {
            "transition_matrix": None,
            "state_names": STATE_NAMES,
            "sojourn_fits": {},
            "current_state": 1,          # Assume neutral when unknown
            "current_state_name": "NEUTRAL",
            "days_in_regime": 0,
            "exit_prob_1d": 0.0,
            "exit_prob_3d": 0.0,
            "exit_prob_5d": 0.0,
            "mean_residual_life": 0.0,
            "n_transitions": 0,
            "n_observations": 0,
            "fit_quality": "INSUFFICIENT",
            "error": reason,
        }

    # ------------------------------------------------------------------
    # Serialization helpers (for regime_cache.db)
    # ------------------------------------------------------------------
    def to_cache_dict(self) -> Dict:
        """Return a JSON-serializable snapshot of the fitted model."""
        if self.transition_matrix is None:
            return {}
        return {
            "transition_matrix": self.transition_matrix.tolist(),
            "sojourn_fits": {str(s): f.to_dict() for s, f in self.sojourn_fits.items()},
        }

    @classmethod
    def from_cache_dict(cls, d: Dict, threshold: float = 1.8) -> "SemiMarkovFitter":
        """Reconstruct a fitter from a cached dict (skips refitting)."""
        obj = cls(threshold=threshold)
        if not d:
            return obj
        P = d.get("transition_matrix")
        if P:
            obj.transition_matrix = np.array(P)
        fits_raw = d.get("sojourn_fits", {})
        for s_str, fd in fits_raw.items():
            s = int(s_str)
            if fd.get("dist") == "weibull":
                f = WeibullFit()
                f.q = fd["q"]
                f.beta = fd["beta"]
                f.n_samples = fd.get("n_samples", 0)
            else:
                f = GeometricFit()
                f.p = fd["p"]
                f.n_samples = fd.get("n_samples", 0)
            obj.sojourn_fits[s] = f
        return obj

    def score_current_state(self, current_state: int, days_in: int) -> Dict[str, Any]:
        """
        Given a current state and days already spent in it, return survival stats.
        Use this when replaying from cache without re-fitting.
        """
        if current_state not in self.sojourn_fits:
            return {
                "current_state": current_state,
                "current_state_name": STATE_NAMES.get(current_state, "UNKNOWN"),
                "days_in_regime": days_in,
                "exit_prob_1d": 0.0,
                "exit_prob_3d": 0.0,
                "exit_prob_5d": 0.0,
                "mean_residual_life": 0.0,
            }
        fit = self.sojourn_fits[current_state]
        return {
            "current_state": current_state,
            "current_state_name": STATE_NAMES[current_state],
            "days_in_regime": days_in,
            "exit_prob_1d": round(fit.exit_prob(days_in, 1), 4),
            "exit_prob_3d": round(fit.exit_prob(days_in, 3), 4),
            "exit_prob_5d": round(fit.exit_prob(days_in, 5), 4),
            "mean_residual_life": round(fit.mean_residual_life(days_in), 2),
        }


# ---------------------------------------------------------------------------
# Convenience wrapper — builds Z-series from ResidualAnomalyModel output
# ---------------------------------------------------------------------------
def fit_from_analysis(analysis: Dict[str, Any], threshold: float = 1.8) -> Dict[str, Any]:
    """
    Convenience function: extract the z_score_1d history from the full
    ResidualAnomalyModel analysis dict and fit the SMM.

    analysis must contain: analysis["history"] — list of dicts with "z_1d" keys.
    Falls back to latest_metrics["z_score_1d"] if history is missing.
    """
    history = analysis.get("history", [])
    if history:
        z_series = np.array([float(h.get("z_1d", 0.0)) for h in history])
    else:
        # Last resort: single-point — model will return INSUFFICIENT
        z_val = analysis.get("latest_metrics", {}).get("z_score_1d", 0.0)
        z_series = np.array([float(z_val)])

    smf = SemiMarkovFitter(threshold=threshold)
    return smf.fit(z_series)


# ---------------------------------------------------------------------------
# Quick self-test (run as __main__)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Semi-Markov Model — self-test")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Simulate a Z-series with regime structure
    T = 300
    z = np.zeros(T)
    state = 1  # start neutral
    for t in range(1, T):
        # Random walk with mean-reversion toward 0
        z[t] = z[t-1] * 0.85 + rng.normal(0, 0.6)
        # Occasionally spike
        if rng.random() < 0.04:
            z[t] += rng.choice([-2.5, 2.5])

    smf = SemiMarkovFitter(threshold=1.8)
    result = smf.fit(z)

    print(f"\nObservations   : {result['n_observations']}")
    print(f"Transitions    : {result['n_transitions']}")
    print(f"Fit quality    : {result['fit_quality']}")
    print(f"\nCurrent state  : {result['current_state_name']} (state {result['current_state']})")
    print(f"Days in regime : {result['days_in_regime']}")
    print(f"\nExit prob 1d   : {result['exit_prob_1d']:.1%}")
    print(f"Exit prob 3d   : {result['exit_prob_3d']:.1%}")
    print(f"Exit prob 5d   : {result['exit_prob_5d']:.1%}")
    print(f"Mean resid life: {result['mean_residual_life']:.1f} days")

    print("\nSojourn fits:")
    for s, fd in result['sojourn_fits'].items():
        print(f"  State {s} ({STATE_NAMES[s]}): {json.dumps(fd)}")

    print("\nTransition matrix (P[from][to]):")
    P = result['transition_matrix']
    header = "        UNDER   NEUTRAL   OVER"
    print(header)
    for i, row in enumerate(P):
        print(f"  {STATE_NAMES[i][:5]:5s}  {row[0]:.3f}    {row[1]:.3f}    {row[2]:.3f}")

    print("\nSelf-test PASSED")
