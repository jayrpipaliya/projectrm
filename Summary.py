"""
================================================================================
PROJECT RM: "RELATIVELY MISPRICED"
QUANTITATIVE MODEL ARCHITECTURE & TOP-TO-BOTTOM SYSTEM SUMMARY
================================================================================
Classification: Institutional Quantitative Specification & System Documentation
Status: Production (v3.4.0)
Live Deployment: https://project-rm-ten.vercel.app

TABLE OF CONTENTS:
1. Executive Overview & Mission
2. Core Mathematical Pipeline: Factor Orthogonalization & Kalman Filter (QMLE)
3. Module A: Continuous-Time Ornstein-Uhlenbeck (OU) Snap-Back Physics
4. Module B: Two-Sided Page's CUSUM Stealth Drift & Regime Shift Detection
5. Module C: Discrete Semi-Markov Regime Duration & Hazard Rate Engine
6. Module D: 3D Parameter Manifold Optimization & Plateau Smoothing Regularization
7. Module E: Vectorized Closed-Form Black-Scholes ATM Derivative Engine
8. Module F: Corporate Earnings Announcement Blackout Risk Gating
9. The Seven Edge Pages: Operational Summary & Navigation Topology
   - 9.1 Dynamic Edge (Cockpit & Execution Blueprint)
   - 9.2 Rare Edge (Watchlist Champion Registry & Pre-Close MOC Screener)
   - 9.3 Tri-Surface Edge (3D Optimum Surface Grid & Walk-Forward Validation)
   - 9.4 Cadence Edge (Weekly Cashflow Velocity & Proximity Radar)
   - 9.5 Atlas Edge (Price Action & Orthogonal Kalman Trajectory)
   - 9.6 Dossier Edge (Trade Diary & Forensic Audit Ledger)
   - 9.7 Diagnostic Edge (Any-Ticker Explorer, Stationarity & Warehouse Telemetry)
10. SQL Warehouse, Data Hygiene & Hybrid Infrastructure Architecture


================================================================================
1. EXECUTIVE OVERVIEW & MISSION
================================================================================
Project RM ("Relatively Mispriced") is an institutional quantitative trading
and risk management platform. The core thesis is that idiosyncratic equity
price action, when decoupled from broad market benchmarks (e.g., SPY, QQQ, DIA)
and sector ETF peers (e.g., IGV, XLK, SMH, XLE), exhibits statistically
significant, localized mean-reverting stationary dynamics.

The system answers four foundational questions for every trading day:
  1. "Is today's single-stock return an idiosyncratic anomaly or standard factor co-movement?"
  2. "If anomalous, is it an overstretched elastic band primed to mean-revert rapidly?"
  3. "Or is it a stealth structural regime drift that will continue trending?"
  4. "What is the optimal causal execution triplet (Entry Barrier σ*, Contract Tenor τ*,
     and Holding Horizon H*) to monetize this dislocation via ATM derivative contracts?"


================================================================================
2. FACTOR ORTHOGONALIZATION & STATE-SPACE KALMAN FILTER (QMLE)
================================================================================
Raw price ratios suffer from non-stationarity, long-term drift, and beta asymmetries.
Project RM implements a rigorous dynamic linear state-space decomposition.

2.1 Sector-to-Market Orthogonalization (Gram-Schmidt):
    Let y_t = ln(P_asset,t / P_asset,t-1)
    Let r_M,t = ln(P_market,t / P_market,t-1)
    Let r_S,t = ln(P_sector,t / P_sector,t-1)

    gamma = Cov(r_M, r_S) / Var(r_M)
    r_S_perp,t = r_S,t - gamma * r_M,t
    X_t = [1, r_M,t, r_S_perp,t]

2.2 State-Space Model Formulation:
    State Equation:       beta_t = beta_{t-1} + w_t,    w_t ~ N(0, Q)
    Observation Equation: y_t = X_t * beta_t + v_t,      v_t ~ N(0, R)
    where beta_t = [alpha_t, beta_mkt,t, beta_sec,t]^T.

2.3 Recursive Kalman Filter Equations:
    1. State Prediction:       beta_{t|t-1} = beta_{t-1|t-1}
    2. Covariance Prediction:  P_{t|t-1} = P_{t-1|t-1} + Q
    3. Innovation Residual:    e_t = y_t - X_t * beta_{t|t-1}
    4. Innovation Variance:    S_t = X_t * P_{t|t-1} * X_t^T + R
    5. Optimal Kalman Gain:    K_t = (P_{t|t-1} * X_t^T) / S_t
    6. State Update:           beta_{t|t} = beta_{t|t-1} + K_t * e_t
    7. Covariance Update:      P_{t|t} = (I - K_t * X_t) * P_{t|t-1}

2.4 Instantaneous Standardized Innovation Z-Score:
    Z_t = e_t / sqrt(S_t) ~ N(0, 1)

2.5 Quasi-Maximum Likelihood Estimation (QMLE):
    The process noise diagonal Q and observation noise scalar R are optimized
    via L-BFGS-B minimization of the Gaussian negative log-likelihood:
    L(Q, R) = 0.5 * sum_{t=1}^T [ ln(2 * pi * S_t) + (e_t^2 / S_t) ]


================================================================================
3. MODULE A: CONTINUOUS-TIME ORNSTEIN-UHLENBECK (OU) SNAP-BACK PHYSICS
================================================================================
3.1 SDE Formulation:
    dZ_t = theta * (mu - Z_t) * dt + sigma_OU * dW_t
    where theta is the restoring force speed and mu -> 0 is the long-term attractor.

3.2 Discrete AR(1) Transition:
    Z_t = a + b * Z_{t-1} + epsilon_t
    theta = -ln(b) / dt
    Half-Life (tau): t_{1/2} = ln(2) / theta  (in trading days)

3.3 Stationarity Condition:
    0 < b < 1  <=>  theta > 0 (Confirms mean-reverting stationary dynamics).


================================================================================
4. MODULE B: TWO-SIDED PAGE'S CUSUM STEALTH DRIFT MONITOR
================================================================================
To detect persistent directional leakage without large single-day shocks:
    S_t^+ = max(0, S_{t-1}^+ + Z_t - k)   [Upper cumulative drift]
    S_t^- = max(0, S_{t-1}^- - Z_t - k)   [Lower cumulative drift]
    where k = 0.5 (reference allowance) and h = 4.0 (decision boundary).

Alert Logic:
    If S_t^+ >= h: STEALTH BULLISH DRIFT (Factor decoupling / upward trend).
    If S_t^- >= h: STEALTH BEARISH DRIFT (Idiosyncratic breakdown / downward trend).


================================================================================
5. MODULE C: DISCRETE SEMI-MARKOV REGIME DURATION & HAZARD ENGINE
================================================================================
Discretizes the continuous Z-score into 3 volatility regimes:
  State 0: UNDERPERFORMING (Z_t <= -sigma*)
  State 1: NEUTRAL         (-sigma* < Z_t < +sigma*)
  State 2: OUTPERFORMING   (Z_t >= +sigma*)

5.1 Embedded Markov Transition Matrix P [3 x 3]:
    p_{ij} = P(J_{n+1} = j | J_n = i) with zero diagonal (p_{ii} = 0).

5.2 Dwell-Time Hazard Distribution Selection via AIC:
    - Geometric Null:   P(T = t) = (1 - p)^{t-1} * p  (Memoryless constant hazard).
    - Discrete Weibull: S(t) = q^{t^beta},  q in (0,1), beta > 0.
      Hazard function:  lambda(t) = 1 - q^{t^beta - (t-1)^beta}.
      * beta > 1: Increasing hazard over time (mean-reverting pressure).
      * beta < 1: Decreasing hazard (streak persistence).
      * beta = 1: Reduces to Geometric.

5.3 Conditional Exit Probability & Mean Residual Life (MRL):
    P(Exit <= N + k | T > N) = [S(N) - S(N + k)] / S(N)
    MRL(N) = E[T - N | T > N] = [1 / S(N)] * sum_{t=N+1}^inf S(t)


================================================================================
6. MODULE D: 3D PARAMETER OPTIMIZATION MANIFOLD & PLATEAU SMOOTHING
================================================================================
Jointly sweeps the 3-dimensional execution parameter manifold:
  Trigger Barrier (sigma) in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
  Contract Tenor (tau)     in [7, 10, 14, 21, 35, 45, 60, 90] DTE
  Holding Horizon (H)      in [1, 2, 3, 4, 5, 7, 10, 12, 14, 15, 18, 21] Sessions

6.1 Raw Cell Utility Function:
    U(sigma, H) = (WinRate% / 100) * AvgROI% * sqrt(N_trades)

6.2 3x3 Plateau Smoothing Kernel:
    P(sigma_i, H_j) = (1 / |N|) * sum_{(r,c) in N(i,j)} U(sigma_r, H_c)
    Plateau Stability Ratio: rho = P(sigma, H) / U(sigma, H)
    Cells with rho >= 0.70 are confirmed as non-overfitted robust plateaus.

6.3 Causal 65/35 Walk-Forward Out-of-Sample (OOS) Cross-Validation:
    - In-Sample (IS): First 65% of historical bars.
    - Out-of-Sample (OOS): Final untouched 35% of historical bars.
    - OOS Efficiency Ratio: eta = WinRate_OOS / WinRate_IS.
    - Status: ROBUST_CONFIRMED (eta >= 0.70, WR_OOS >= 45%), MILD_DEGRADATION, OVERFIT_REJECTED.


================================================================================
7. MODULE E: VECTORIZED CLOSED-FORM BLACK-SCHOLES ATM DERIVATIVE ENGINE
================================================================================
Ultra-fast closed-form Black-Scholes pricing using the Gauss error function erf(x):
    Phi(x) = 0.5 * [1 + erf(x / sqrt(2))]
    d1 = [ln(S/K) + (r + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    Call = S * Phi(d1) - K * exp(-r*T) * Phi(d2)
    Put  = K * exp(-r*T) * [1 - Phi(d2)] - S * [1 - Phi(d1)]

Execution Specifications:
  - Strike Selection: K = round(S_entry) (At-The-Money).
  - Volatility Proxy: sigma_entry = max(0.12, sqrt(252) * std_20d(log_returns)).
  - Non-Linear Theta Decay: Decay% = [1 - sqrt(max(0.001, (tau - H) / tau))] * 100%.
  - Profit Factor (PF): Gross Wins / |Gross Losses|.


================================================================================
8. MODULE F: CORPORATE EARNINGS ANNOUNCEMENT BLACKOUT RISK GATING
================================================================================
Options entered directly before earnings announcements suffer from binary gap risk
and severe post-announcement implied volatility crush.

Implementation:
  - Historical corporate earnings announcement dates are tracked in SQL warehouse.
  - Blackout Window: The announcement session plus the immediate subsequent session:
    B = {t_earn, t_earn + 1}.
  - Any signal firing on t in B is automatically blocked from execution:
    Status = BLOCKED_EARNINGS.
  - Preserves mathematical model edge from unhedged binary event shocks.


================================================================================
9. THE SEVEN EDGE PAGES: OPERATIONAL SUMMARY & TOPOLOGY
================================================================================

9.1 Dynamic Edge ('cockpit'):
    - Central interactive quant cockpit.
    - Real-time Kalman beta decomposition (Market & Sector Orthogonal).
    - Multi-horizon residual Z-scores (1-day, 5-day, 20-day).
    - Continuous Ornstein-Uhlenbeck half-life reversion monitor.
    - Two-sided Page's CUSUM drift alert engine.
    - Unified Trade Execution Blueprint with target strike, tenor, and holding horizon.

9.2 Rare Edge ('gold-ledger'):
    - Watchlist Champion Registry tracking globally optimal triplets (sigma*, tau*, H*)
      across 21 core liquid watchlist assets.
    - Pre-Close 3:45 PM Market-On-Close (MOC) Screener: Scans between 3:45 PM and 3:58 PM EST
      to identify active trigger barrier breaches (|Z_t| >= sigma*).
    - Executive summary KPI cards and signal cadence metrics styled with clean grey borders
      and grey/white text.

9.3 Tri-Surface Edge ('tenor-sweep'):
    - Interactive 3D parameter surface explorer across trigger barriers and option tenors.
    - 2D heatmap grid toggleable by Profit Factor, Win Rate, ROI %, and Dollar P&L.
    - Full day-by-day holding duration curve (1 to 21 sessions).
    - Single dedicated "Run 3D Global Optimum" header button with plateau smoothing.

9.4 Cadence Edge ('income-strategy'):
    - High-velocity weekly cashflow engine with short holding periods (1 to 5 sessions).
    - 3:45 PM to 4:00 PM MOC Proximity Radar:
      Evaluates standard deviation distance to trigger barrier:
      delta_sigma = max(0.0, sigma* - |Z_t|).
    - Multi-asset basket simulator calculating aggregate portfolio arrival frequency
      (trades/wk), weighted win rate, and blended profit factor.

9.5 Atlas Edge ('price-action'):
    - Multi-timeframe technical structure and candlestick geometry.
    - Overlays the Kalman orthogonal reconstructed price channel against actual price bars.
    - Highlights confluence between technical support/resistance and residual triggers.

9.6 Dossier Edge ('trades-log'):
    - Forensic walk-forward trade execution journal.
    - Session-by-session trade lifecycle tracking from entry to exit.
    - Corporate earnings announcement blackout filter toggle and audit counters.
    - Forensic loss categorization (REGIME_DRIFT, THETA_BLEED, WHIPSAW).

9.7 Diagnostic Edge ('diagnostics'):
    - Interactive Any-Ticker Explorer: Input box allowing on-demand querying of any equity
      symbol (e.g. AAPL, NVDA, TSLA, SPY).
    - Pulls historical OHLCV bars and computes annualized 20-day Realized Volatility.
    - Renders dynamic stock price/volatility charts and live option chains.
    - Econometric stationarity testing: Augmented Dickey-Fuller (ADF) & Phillips-Perron tests.
    - Innovation whiteness testing: Ljung-Box Q-statistic.
    - Discrete Semi-Markov regime duration hazard analytics.
    - SQL Warehouse size and table telemetry.


================================================================================
10. SQL WAREHOUSE, DATA HYGIENE & INFRASTRUCTURE ARCHITECTURE
================================================================================
10.1 Relational SQLite Schema (data/warehouse.db):
    - stocks: Master ticker symbol registry, benchmark mapping, sync timestamps.
    - stock_bars: Historical daily OHLCV bars (Primary Key: ticker, trade_date).
    - option_contracts: Master options contract registry with standardized OCC symbology.
    - option_bars: Daily OHLCV option contract bars.
    - realized_volatility: Pre-computed rolling 10d, 20d, 30d annualized volatility.
    - earnings_calendar: Corporate earnings announcement schedule.
    - ticker_optimal_params: Walk-forward validated parameters and plateau stability metrics.
    - champion_gold_ledger: Watchlist champion parameters, current Z, and MOC signal status.
    - champion_gold_trades: Individual trade execution journal records and performance metrics.

10.2 Hybrid Market Data Pipeline:
    - Primary Connector: Massive / Polygon.io REST API for live daily bars and option chains.
    - Fallback Connectors: yfinance multi-year historical loader and synthetic option chains.
    - Caching Layer: Sub-millisecond SQLite queries eliminate HTTP 429 rate limits.

10.3 Deployment & Server Execution:
    - Local Environment: Standalone multi-threaded Python server (server.py) on port 8050.
    - Production Environment: Serverless Python request handler (api/index.py) on Vercel
      aliased to https://project-rm-ten.vercel.app.

================================================================================
END OF ARCHITECTURAL SUMMARY
================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
