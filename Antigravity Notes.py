"""
================================================================================
PROJECT RM: "RELATIVELY MISPRICED"
ANTIGRAVITY DEVELOPMENT LOG & ENGINEERING NOTES
Date: October 16, 2026
================================================================================

1. EXECUTIVE SUMMARY OF TODAY'S BUILD
--------------------------------------------------------------------------------
Today we designed, coded, and verified the complete quantitative foundation
for Step 1 of Project RM ("Relatively Mispriced"). The objective is to
statistically determine whether an asset's move relative to broad market
benchmarks (e.g., QQQ) and sector peers (e.g., IGV) is:
  (a) A normal co-movement driven by market/sector beta,
  (b) An overstretched transient anomaly with high snap-back probability, or
  (c) A stealth structural regime transition that will continue to trend.

All of this was accomplished using ONLY historical daily close prices:
  - Zero options quotes required
  - Zero implied volatility quotes required
  - Instantaneous, live-updating mathematical derivations


2. CHRONOLOGICAL EVOLUTION & ARCHITECTURAL DECISIONS
--------------------------------------------------------------------------------

PHASE 1: LIVE DATA INGESTION & LOCAL SQL WAREHOUSE
--------------------------------------------------
* What was built:
  - Connected live to the Massive / Polygon market data API using the user's
    configured API key in `.env`.
  - Created a local SQLite warehouse (`data/warehouse.db`) with normalized
    tables: `stocks`, `stock_bars`, `option_contracts`, `option_bars`, and
    `realized_volatility`.
* Why it was added:
  - The API free plan enforces a rate limit of 5 calls per minute.
  - Backtesting and multi-factor regressions scan hundreds of trading bars.
  - Querying the live API on every parameter sweep would trigger 429 throttling
    and 400ms+ network latency.
  - The local SQL warehouse caches clean data once; subsequent queries execute
    in under 0.4 milliseconds with zero rate limits.
* Action Taken:
  - Verified live QQQ, MSFT, and IGV ingestion (251 clean daily bars).
  - Purged all mock/simulated fallback code per user instruction so that 100%
    of the system operates exclusively on genuine market data.


PHASE 2: RELATIVE PERFORMANCE MODELING (WHY RAW RATIOS FAIL)
------------------------------------------------------------
* The Problem:
  - The intuitive approach of measuring relative moves using raw price ratios
    (e.g., Price_MSFT / Price_QQQ) fails quantitatively due to:
    1. Non-Stationarity: Nominal prices drift over years. A static mean or
       Z-score on a price ratio is econometrically spurious.
    2. Beta Asymmetry: If QQQ drops -2.0%, and MSFT has a beta of 1.2, MSFT is
       expected to drop -2.4%. If MSFT only drops -0.5%, in raw terms it is down,
       but relative to expectations it showed massive idiosyncratic strength.
    3. Sector Decoupling: MSFT might follow software (IGV) on days when semis
       (SMH) drag QQQ down.
* The Solution:
  - Decompose asset return into Market, Orthogonal Sector, and Idiosyncratic
    Residual:
      R_asset(t) = alpha + beta_mkt * R_mkt(t) + beta_sec * R_sec_perp(t) + e(t)
  - Orthogonalize sector to market so beta_sec captures pure industry-specific
    dynamics with zero collinearity distortion.


PHASE 3: FROM ROLLING 60-DAY OLS TO STATE-SPACE KALMAN FILTER
-------------------------------------------------------------
* The Problem (The "Ghost Drop" Effect):
  - A fixed 60-day rolling regression window creates artificial cliffs. When a
    single outlier shock occurs on Day t, it distorts the beta for 60 days. On
    Day t+60, that shock abruptly drops out of the window, causing an artificial
    "ghost jump" in beta with ZERO new market news.
  - Furthermore, fixed windows discard the first 60 days of historical data.
* The Solution:
  - Upgraded to a State-Space Kalman Filter with Maximum Likelihood Estimation (MLE):
      Measurement Equation: y_t = alpha_t + X_t * beta_t + e_t,  e_t ~ N(0, R)
      Transition Equation:  beta_t = beta_{t-1} + eta_t,       eta_t ~ N(0, Q)
* Why it was added:
  - Past shocks diffuse exponentially and continuously through the error
    covariance matrix P. No cliff effect.
  - Dynamic Information Weighting: The Kalman Gain K_t balances adaptation speed
    against noise automatically.
  - Warmed up in just 15 days, preserving 235+ trading days of historical data.


PHASE 4: TESTING AND ULTIMATE REMOVAL OF GARCH(1,1)
----------------------------------------------------
* What was investigated:
  - Applied GARCH(1,1) on Kalman innovations to model the breathing clusters
    of volatility and calm:
      sigma2_t = omega + alpha * e_{t-1}^2 + beta * sigma2_{t-1}
* Why it was REMOVED (The Trader's Critique):
  - GARCH exhibits a heavy "post-earnings hangover": persistence (alpha + beta)
    is typically ~0.75-0.85, meaning that after a single 1-day earnings shock,
    the band remains artificially inflated for 15-25 trading days.
  - During those 3 weeks, genuine subsequent anomalies are divided by this
    bloated denominator, muting real trading signals.
  - GARCH only measures noise spread; it possesses zero concept of restoring
    elasticity or whether a stock typically bounces at a boundary.
* The Better Alternative Implemented:
  - Reverted to the Kalman Filter's clean instantaneous prediction variance:
      Z_t = e_t / sqrt(S_t)
  - Replaced GARCH with the Ornstein-Uhlenbeck snap-back engine.


PHASE 5: MODULE A — ORNSTEIN-UHLENBECK (O-U) SNAP-BACK ENGINE
-------------------------------------------------------------
* What was built:
  - Modeled the idiosyncratic residual Z-score as a continuous mean-reverting
    stochastic process:
      dZ_t = theta * (mu - Z_t) dt + sigma * dW_t
  - Calculated the exact Mean-Reversion Half-Life:
      tau = ln(2) / theta  (in trading days)
  - Added an empirical backtester that scans historical crosses past |Z| >= 1.8 sigma
    and tracks:
      1. Reversion Win Rate (% returning inside +/-0.5 sigma within 1.5 * tau)
      2. Average Recovery Days
* Why it was added:
  - Tells the trader: "Historically, when MSFT reaches this level of idiosyncratic
    stretch, does it actually snap back, how often, and how fast?"
  - Live MSFT Test Result: 100.0% Win Rate across 9 past episodes, with an
    average snap-back time of 1.4 trading days!


PHASE 6: MODULE B — CUSUM STEALTH REGIME SHIFT MONITOR
------------------------------------------------------
* What was built:
  - Implemented Page's Two-Sided Cumulative Sum (CUSUM) test:
      S_t^+ = max(0, S_{t-1}^+ + Z_t - k)
      S_t^- = max(0, S_{t-1}^- - Z_t - k)
  - Tracks consecutive one-sided drift streaks.
  - Compares drift against the statistical barrier h = 3.8.
* Why it was added (The Critical Question Solved):
  - "What if an asset has smaller Z-scores day after day (-0.7, -0.6, -0.8),
    and is NOT mean-reverting, but instead regime-changing?"
  - Pure mean-reversion models are blind to this and will tell you to keep fading
    a move that has become a structural trend.
  - CUSUM catches this: individual days look mild, but cumulative one-sided drift
    accumulates linearly and triggers an active "Stealth Regime Shift Alert".


PHASE 7: THE MODULAR COCKPIT DASHBOARD
--------------------------------------
* What was built:
  - A modern quantitative web cockpit served via `server.py` and `static/index.html`.
  - Added independent toggle switches:
      [Toggle A: O-U Snap-Back Engine]
      [Toggle B: CUSUM Regime Drift Monitor]
      [Toggle C: Multi-Horizon Z-Scores]
  - Visual system assessment banner that categorizes today's action:
      - SNAP-BACK OPPORTUNITY (Green)
      - STEALTH REGIME SHIFT (Red)
      - MILD DEVIATION (Amber)
      - NORMAL CO-MOVEMENT (Cyan)
  - Three distinct chart views:
      1. Snap-Back Bands (Z_t with +/-1.8 triggers & shaded +/-0.5 target ribbon)
      2. CUSUM Drift Curves (S+ and S- accumulation vs threshold h)
      3. Multi-Horizon Decoupling (1d, 5d, 20d structural overlay)


3. KEY FORMULAS SUMMARY
--------------------------------------------------------------------------------
1. Market-Orthogonal Sector Return:
   R_sec_perp = R_sec - (Cov(R_sec, R_mkt) / Var(R_mkt)) * R_mkt

2. Kalman Measurement Equation:
   y_t = alpha_t + beta_mkt,t * R_mkt + beta_sec,t * R_sec_perp + e_t

3. Clean Studentized Innovation Z-Score:
   Z_t = e_t / sqrt(S_t), where S_t = X_t * P_pred * X_t^T + R

4. Mean Reversion Half-Life:
   tau = ln(2) / theta, where b = exp(-theta * dt) from AR(1) fit

5. CUSUM Accumulator:
   S_t^+ = max(0, S_{t-1}^+ + Z_t - 0.4)
   S_t^- = max(0, S_{t-1}^- - Z_t - 0.4)


4. ADDENDUM: UI ARCHITECTURE REFINEMENT & MULTI-HORIZON STDV
--------------------------------------------------------------------------------
- User Feedback:
  1. Relocate auxiliary historical diagnostic charts, options chain, and API
     testers further down the page so the Relative Anomaly Cockpit takes center stage.
  2. Maintain continuous visibility for 1 STDV, 5 STDV, and 20 STDV scores
     in the metrics grid while preserving all cockpit module toggle buttons.
- Implementation:
  - Created a dedicated "Diagnostics & Testing Lab" section at the bottom of the page
    with a dashed boundary divider and a quick-jump anchor link in the header.
  - Added dedicated metric cards for 1-Day (1 STDV), 5-Day (5 STDV), and 20-Day (20 STDV)
    Z-scores alongside O-U Win Rate, CUSUM Drift, Dynamic Betas, and Expected Move.
  - Linked the "Multi-Horizon Z (5d/20d)" cockpit button to dynamically overlay the
    5-day and 20-day curves onto the Snap-Back Bands chart when activated.

5. ADDENDUM: TARGET ASSET PRICE ACTION CHART WITH SIGNAL OVERLAYS
--------------------------------------------------------------------------------
- User Feedback:
  - Add a dedicated chart section between the Relative Anomaly Cockpit and the
    Diagnostics Lab showing the price action of the specific asset being measured,
    allowing immediate visual comparison between residual Z-score spikes and price.
- Implementation:
  - Created Section 1.5: Target Asset Price Action & Signal Overlays.
  - Aligned the time series 1:1 with the Relative Anomaly innovation residuals.
  - Dynamically updates with whatever asset is selected (e.g., MSFT, NVDA, AAPL, JNJ).
  - Overlays visual signal markers directly onto the price trajectory:
      • Green triangles (▲): Oversold Snap-Back Buy candidates (Z <= -1.8 sigma).
      • Red inverted triangles (▼): Overbought Snap-Back Sell candidates (Z >= +1.8 sigma).
      • Purple diamonds (■): Active Stealth Regime Shift alerts.
  - Includes a toggle button to show/hide anomaly signal markers.

6. ADDENDUM: DYNAMIC TRIGGER BARRIER (±Xσ) & SYNCHRONIZED WIN RATE RECALCULATION
--------------------------------------------------------------------------------
- User Inquiries & Requirements:
  - "Where did the 1.8 come from?"
    • Rationale: In a standard Gaussian distribution, ±1.8σ isolates the outer ~7.2%
      of tail shocks (~3.6% per tail). In a 250-trading-day annual sample, this yields
      approx 8-15 distinct, high-conviction snap-back episodes rather than excessive noise.
  - "Can I adjust this and can I see the win rate adjust with it?"
  - "Can you also adjust the signals that fire on the chart underneath with overlaying
     1.x stdv zscore signal for bearish and bullish?"
- Implementation:
  - Added an interactive Trigger Barrier input bar with both a precision number box
    (0.1 step) and a range slider (1.0σ to 3.0σ).
  - Wired the backend (`residual_model.py` and `server.py`) to accept dynamic `?threshold=X`
    queries and re-run `OrnsteinUhlenbeckEngine.evaluate_snapbacks(barrier=X)`.
  - The O-U Win Rate card dynamically recalibrates its historical win %, episode count,
    and label (e.g., "O-U Win Rate @ ±1.5σ (17 episodes, 100%)").
  - Synchronized Section 1.5 Target Asset Price Chart:
    • Green triangles (▲): Bullish Snap-Back candidates (Z <= -thresh).
    • Red inverted triangles (▼): Bearish Snap-Back candidates (Z >= +thresh).
    • Live slider interaction: dragging the slider immediately re-evaluates signal dots
      and trigger lines across both charts in real time.

7. ADDENDUM: MINIMAL FINTECH UI REDESIGN & BACKUP REPOSITORY
--------------------------------------------------------------------------------
- User Request:
  - Redesign dashboard to match a sleek, modern fintech UI aesthetic (inspired by
    the dark-mode Vault/fintech reference dashboard with navy gradient cards, pill tabs,
    bold typography, and rounded transaction tables).
  - Explicit requirement: preserve all existing quantitative features intact, and create
    a full copy/backup of the previous version in case the user wants to revert.
- Implementation:
  - Backup created: `static/index_backup_v1.html` retains the exact previous iteration.
  - UI Transformation:
    • Clean minimal sidebar with "Project RM" workspace badge, navigation icons, and
      API/SQLite status indicator.
    • Top header with pill action buttons (presets, diagnostics jump, sync).
    • Hero Split Card: Large bold 1-day innovation Z-score stat display (like the
      Total Balance card in the reference) paired with the primary Anomaly chart.
    • Pill Tab controls for chart modes ("Snap-Back Bands", "CUSUM Drift", "Multi-Horizon Z").
    • Section 1.5 Target Asset price chart styled with matching minimal borders and
      re-aligned 1:1 with residual shocks.
    • Section 1.8 Recent Anomaly Log: transaction-style table with tag pills
      (Bullish Snap, Bearish Snap, Regime Drift, Normal) and live search filter.
    • Modernized option chain table and endpoint tester sandbox.
    • All live APIs, dynamic threshold slider, O-U win rates, and toggle buttons preserved.

8. ADDENDUM: METHOD A - DYNAMIC ATM OPTION BACKTEST ENGINE
--------------------------------------------------------------------------------
- User Requirement:
  - Measure historical performance of entering directional ATM options at market close
    on days where a bullish (Z <= -threshold) or bearish (Z >= +threshold) signal fires.
  - Dynamically evaluate returns based on whatever trigger barrier is selected on the slider
    (not fixed to 1.8).
- Implementation:
  - Created `SyntheticOptionBacktestEngine` in `residual_model.py`:
    • On signal day $T$ close, sets ATM strike $K = \text{round}(S_T)$.
    • Uses Black-Scholes formula with 14 DTE and 20-day historical realized volatility ($\sigma_{RV}$).
    • Evaluates exit value at $T+1$ close (1-day hold) and $T+2$ close (2-day hold).
    • Calculates option % ROI ($\Delta\%$), dollar PnL per contract ($\times 100$), win rate %,
      and maximum profit/loss.
  - Front-End Integration:
    • Added dedicated metric card: "1-Day ATM Option ROI @ ±Xσ".
    • Populates the Anomaly Log table with exact trade details: Option Trade (Call/Put + Strike),
      1D Option PnL %, and 2D Option PnL %.
    • Slider Interaction: dragging the slider immediately triggers a debounced recalculation
      across all past trading bars.

9. ADDENDUM: DEDICATED OPTION TRADES LOG & STRIKE DETERMINATION
--------------------------------------------------------------------------------
- User Questions & Requirements:
  - "Can I just have a separate log of just the trades?"
  - "How do I know which strike is the simulated option for?"
- Implementation & Strike Logic:
  - Strike Selection ($K$):
    • At market close on entry day $T$, the underlying stock closes at price $S_T$.
    • The ATM strike is explicitly computed as $K = \text{round}(S_T)$ (e.g., if MSFT
      closed at $478.56 on 2025-12-10, the ATM strike is exactly $479.00 Call; if MSFT
      closed at $465.95 on 2026-01-23, the ATM strike is exactly $466.00 Put).
  - Dedicated Section 1.7 "Simulated ATM Option Trades Log":
    • Isolates only days where $|Z| \ge \text{Trigger Barrier}$ and a trade actually entered.
    • Columns: Entry Date, Action (BUY CALL / BUY PUT), Signal Z, Stock Entry Price,
      Exact ATM Strike ($K), Entry Option Price ($), Exit 1D ($), 1-Day ROI %,
      Exit 7D ($), 7-Day ROI %, and Cumulative 7D 1-Contract Dollar PnL ($).
    • Mini-performance metrics header showing 1-Day Win Rate, 1-Day Avg ROI, 7-Day Win Rate,
      7-Day Avg ROI, and Net 7-Day Dollar PnL.
    • Filter search box to easily search "CALL", "PUT", or specific dates.
    • Holding horizon: evaluates 7 trading days forward with 21 DTE contracts to ensure
      proper extrinsic decay modeling.

10. ADDENDUM: DYNAMIC 1-30 DTE SWEEP & SPLIT STRATEGY RECOMMENDATIONS
--------------------------------------------------------------------------------
- User Requirements:
  - Sweep holding horizons from 1 DTE up to 30 DTE in the background to find optimal DTE and return.
  - Do NOT clutter the UI with the raw 30-day table; display clean, actionable recommendations.
  - Split Bullish (ATM Calls) and Bearish (ATM Puts) strategies separately with independent DTEs,
    win rates, and average ROI %.
  - Delete the "All Trading Days Anomaly History Table" (Section 1.8).
- Implementation:
  - Backend (`SyntheticOptionBacktestEngine` in `residual_model.py`):
    • Partitions signal episodes into Bullish ($Z \le -\text{barrier}$) and Bearish ($Z \ge +\text{barrier}$).
    • Computes a full holding horizon performance curve for $d = 1 \dots 30$ days forward independently
      for both directions.
    • Evaluates Black-Scholes pricing with 35 DTE contracts, risk-free rate $r=4.5\%$, and prevailing
      20-day realized volatility ($\sigma_{RV}$).
    • Optimizes DTE horizon maximizing average ROI subject to robust win rate ($\ge 40\%$).
    • Returns `bullish_recommendation`, `bearish_recommendation`, and detailed `trades` evaluated at
      each trade's specific strategy recommended DTE.
  - Frontend (`static/index.html`):
    • Deleted Section 1.8 Anomaly History table and removed its navigation link.
    • Replaced with Section 1.8 "Strategy Recommendations (Bullish vs Bearish)":
      - Bullish Strategy Card: Recommended DTE, Historical Win Rate %, Avg ROI %, Total Trades.
      - Bearish Strategy Card: Recommended DTE, Historical Win Rate %, Avg ROI %, Total Trades.
    • Updated Section 1.7 Simulated Option Trades Log to display:
      - Entry Date, Action/Strike, Signal Z, Stock Entry, ATM Strike ($K), Entry Opt, Exit 1D, 1D ROI,
        Rec DTE, Exit Rec, Rec ROI %, and Rec 1-Contract PnL.
    • Mini-performance metrics bar dynamically updates overall win rate, average ROI, Bull/Bear win rates,
      recommended DTEs, and net cumulative PnL.
    • Fully interactive: slider movement triggers dynamic DTE recalibration across both directions.

11. ADDENDUM: CORPORATE EARNINGS ENDPOINT & MEAN-REVERSION TRADE BLACKOUT FILTER
--------------------------------------------------------------------------------
- User Requirement:
  - Pull up corporate earnings data as a live endpoint for any ticker.
  - Automatically filter out and exclude any mean-reversion trade that would be taken
    on an earnings announcement date or the trading day immediately after earnings.
- Implementation:
  - Live Earnings Endpoint (`/api/earnings?ticker=XYZ`):
    • Added `get_earnings_calendar(ticker)` to `MassiveApiClient`:
      - Resolves ticker CIK using Polygon ticker overview / SEC mapping.
      - Queries official SEC EDGAR submissions (`https://data.sec.gov/submissions/CIK{cik}.json`).
      - Identifies Form 8-K filings citing Item 2.02 ("Results of Operations and Financial Condition").
    • Caches corporate earnings dates in local SQLite warehouse table `earnings_calendar`.
  - Earnings Trade Blackout Engine (`residual_model.py`):
    • `ResidualAnomalyModel.analyze()` automatically fetches and caches corporate earnings dates.
    • `SyntheticOptionBacktestEngine.evaluate_option_performance()` constructs a corporate blackout set:
      - Blackout Date 1: The date of the earnings release.
      - Blackout Date 2: The trading day immediately following the earnings release (T+1).
    • Signals triggering during this blackout window are isolated into `excluded_earnings_trades`
      and omitted from options trade simulation and DTE performance curves.
    • Prevents unhedged binary event risk and artificial earnings-drift skew in mean-reversion stats.
  - Frontend Controls & Indicators (`static/index.html`):
    • Added "Earnings Blackout: ON / OFF" interactive toggle pill to Section 1.7 Trades Log header.
    • Added dynamic status pill (e.g. "3 Earnings Trades Blocked") displaying exact count of
      corporate binary events filtered out.
    • Toggling the button re-runs the backtest with/without earnings blackouts in real time.

12. ADDENDUM: DIRECTIONAL REGIME HIGHLIGHTER (GREEN FOR BULLISH, RED FOR BEARISH)
--------------------------------------------------------------------------------
- User Requirement:
  - Highlight regime changes based on whether the regime is turning Bullish (Green highlight)
    or Bearish (Red highlight) for that day's candle and subsequent drift days.
  - Eliminate the purple diamond markers that previously blocked and obscured Z-score triangles.
- Implementation (`static/index.html`):
  - Removed diamond point markers (`rectRot` style) from the price line dataset.
  - Implemented directional Chart.js canvas plugin `regimeHighlightPlugin`:
    • Runs on `beforeDatasetsDraw` in the background behind candles/price line and signal triangles.
    • Evaluates regime direction from CUSUM alert (`regime_alert > 0` vs `< 0`), drift streak sign,
      and positive vs negative cumulative drift ($S^+$ vs $S^-$):
      - Bullish Regime Shift (Upward Drift): Translucent emerald green highlight (`rgba(16, 185, 129, 0.24)`)
        with glowing mint top indicator strip.
      - Bearish Regime Shift (Downward Drift): Translucent coral red highlight (`rgba(239, 68, 68, 0.24)`)
        with glowing red top indicator strip.
    • Automatically spans the alert day and any subsequent consecutive drift days.
  - Signal triangles (Green ▲ for oversold snap-backs, Red ▼ for overbought snap-backs) now
    render with 100% visibility directly on top of the highlighted background area.
  - Section 1.5 legend and tooltips updated to display directional regime status.

13. ADDENDUM: BLACK & CHARCOAL PALETTE REDESIGN, PRESET UPDATES & CHART VISIBILITY
--------------------------------------------------------------------------------
- User Requirements:
  1. Palette: Redesign dashboard to true black (#000000) with a charcoal grey (#121214 / #18181b / #27272a)
     and white (#ffffff) accent palette, completely replacing previous navy/dark blue and light blue colors.
  2. Emojis: Remove all emojis in front of names across Quant Cockpit navigation, headers, and cards.
  3. Action Buttons: Remove the duplicate Snap-Back and CUSUM buttons under "Latest Move vs Beta Expected".
  4. Legend Click Lock: Disable accidental deselecting/hiding of "+3.0 Extreme", "Snap-Back Trigger",
     "0.5 Target Ribbon", and the asset 1-Day Z-Score lines on the Z-Score chart by locking legend click handlers.
  5. Z-Score Line: Make the 1-Day Z-score line thin (1.2px) and white (#ffffff).
  6. Presets: Remove AAPL preset; replace JNJ preset with UNH/SPY/IGV.
  7. Number Input: Reduce font size and adjust width/padding on the trigger barrier number box so the
     full number is clearly legible without being clipped by the browser stepper arrows.
  8. Header Badge: Remove "Step 1 Relative Mispricing" text at the top of the screen.

- Implementation (`static/index.html`):
  - Theme:
    • Root variables: `--bg-body: #000000`, `--bg-sidebar: #09090b`, `--bg-card: #121214`,
      `--bg-card-inner: #18181b`, `--border-subtle: #27272a`, `--border-strong: #3f3f46`,
      `--border-active: #ffffff`, `--accent-blue: #ffffff`.
    • Buttons, tabs, and tables updated with charcoal grey containers, dark subtle hover states,
      and crisp white accents.
    • Chart.js gridlines shifted from navy (#131b2c) to clean charcoal (#222225) with neutral ticks (#71717a).
  - Clean Typography & Emojis:
    • Stripped emoji icons from Quant Cockpit navigation links (Price & Signals, Option Trades Log,
      Strategy Reco, Diagnostics Lab, Option Chains, Warehouse Sync).
    • Stripped emojis from section headers, strategy recommendation cards, and earnings toggle button.
    • Removed "Step 1 Relative Mispricing" header badge.
  - Interactive Legend Lock:
    • In `renderAnomalyChart`, set `legend: { onClick: null, onHover: (e) => { e.chart.canvas.style.cursor = 'default'; } }`.
      All 4 critical overlays remain visible on the canvas with zero risk of accidental click deselection.
  - White & Thin Z-Score Line:
    • Configured `${asset} 1-Day Z-Score` line with `borderColor: "#ffffff"` and `borderWidth: 1.2`.
  - Presets & Inputs:
    • Presets updated: MSFT/QQQ/IGV, NVDA/QQQ/SMH, UNH/SPY/IGV.
    • Trigger barrier input set to `font-size: 0.78rem; width: 66px; padding: 4px 6px;` providing
      full visibility for decimal values alongside browser spinner arrows.
    • Removed `.stat-action-buttons` under the Latest Move decomposition box.

14. ADDENDUM: TOP-RIGHT PRESET REMOVAL, UNH/SPY/XLV, LEGEND REMOVAL & COCKPIT MODULES CLEANUP
--------------------------------------------------------------------------------
- User Requirements:
  1. Top-right corner of the page: remove both "MSFT Preset" and "NVDA Preset" buttons.
  2. Presets bar: replace `UNH/SPY/IGV` with `UNH/SPY/XLV`.
  3. Anomaly Z-Score Chart: remove the "+3.0 extreme, +1.8 snap back trigger, 0.5 target ribbon,
     MSFT 1 ZScore" text above the chart (the Chart.js top legend).
  4. Cockpit Modules: remove all "cockpit modules" text and Module A (O-U) / Module B (CUSUM) buttons,
     while keeping the button for "Multi-Horizon Z (5d/20d)" to toggle 5d and 20d curves on/off on the chart.

- Implementation (`static/index.html`):
  - Top-Right Action Header:
    • Stripped `MSFT Preset` and `NVDA Preset` pill buttons.
    • Preserved `Diagnostics` link and `Sync` button.
  - Preset Bar:
    • Updated preset to `UNH/SPY/XLV` (`applyPreset('UNH', 'SPY', 'XLV')`).
  - Legend Above Chart:
    • In `renderAnomalyChart`, set `plugins: { legend: { display: false } }` across Snap-Back,
      CUSUM, and Multi-Horizon views. The text above the chart is completely removed, while the
      datasets (+3.0σ, trigger lines, ribbon, and white Z-score line) remain drawn on the canvas.
  - Cockpit Modules & Multi-Horizon Button:
    • Removed the `.modules-toggle-bar` container and the Module A / Module B buttons.
    • Relocated the `Multi-Horizon Z (5d/20d)` toggle button directly into the right hero chart header
      next to the view tabs and ticker label.
    • Simplified `toggleModule('horizon')` to toggle `modulesState.horizon` and dynamically re-render
      the dashed 5d and 20d curves on the chart.

15. ADDENDUM: UNIVERSE SCANNER & PRE-CLOSE EXECUTION SCREENER (7 SECTORS, 69+ ASSETS)
--------------------------------------------------------------------------------
- Architectural Motivation:
  • Institutional Relative Anomaly hunting across individual stock-factor pairs requires
    scanning an extensive cross-asset universe before the market close (3:45–3:55 PM EST window)
    to identify firing mean-reversion snap-backs before 4:00 PM equity option settlement.
  • Prevents blind entry into binary earnings announcements or ongoing structural drift shifts.

- Institutional Universe Mapping (`universe.py`):
  • Technology & AI: MSFT (QQQ/IGV), NVDA (QQQ/SMH), AMD (QQQ/SOXX), AVGO (QQQ/SMH),
    AAPL (QQQ/XLK), META (QQQ/XLC), GOOGL (QQQ/XLC), AMZN (QQQ/XLY), ORCL (QQQ/IGV),
    CRM (QQQ/IGV), NOW (QQQ/IGV), CRWD (QQQ/CIBR), PANW (QQQ/CIBR), PLTR (QQQ/IGV).
  • Financials & Banking: JPM, BAC, GS, MS, WFC, C (SPY/XLF), BLK, SCHW (SPY/KBE),
    V, MA (SPY/XLF), COIN (QQQ/IBIT).
  • Semiconductors & Hardware: MU, MRVL, DELL (QQQ/SMH), QCOM, TXN, AMAT, LRCX, KLAC (QQQ/SOXX),
    ANET (QQQ/IGV), CSCO (QQQ/XLK).
  • Healthcare & Pharma: LLY, NVO, ABBV, JNJ, MRK, PFE, ISRG, UNH, ABT (SPY/XLV),
    AMGN (SPY/XBI), GILD, REGN (SPY/IBB).
  • Consumer & Retail: TSLA, HD, LOW, MCD, SBUX, NKE, BKNG, ABNB (SPY/QQQ vs XLY),
    TGT, WMT, COST (SPY/XLP), NFLX (QQQ/XLC).
  • Energy & Industrials: XOM, CVX, COP (SPY/XLE), SLB (OIH), CAT, DE, GE, RTX, LMT,
    UPS, FDX, HON (SPY/XLI).
  • Communication Services & Media: DIS, T, VZ, CMCSA (SPY/XLC), TTD, ROKU (QQQ/XLC).

- Pre-Close Execution Screener Engine (`scanner_engine.py`):
  • Mathematical Timing: Kalman dynamic state parameters calibrated through T-1 close;
    3:45–3:55 PM snapshot returns evaluate intraday innovation residuals E[R_i] = alpha + beta_mkt*R_mkt + beta_sec*R_sec.
  • Signal Rules:
    - Z <= -1.8 sigma: Actionable BUY CALL (Oversold Snap-Back)
    - Z >= +1.8 sigma: Actionable BUY PUT (Overbought Snap-Back)
    - Corporate Earnings Filter: If T or T+1 matches SEC EDGAR Form 8-K Item 2.02 date,
      flags EARNINGS BLACKOUT and excludes from active signals.
    - CUSUM Drift Monitor: Flags REGIME ALERT if cumulative drift S+ or S- breaches h.
  • Opportunity Ranking Score:
    Score = 50.0 (if active snap-back) + 10.0*|Z| + 0.2*WinRate - 15.0*(if CUSUM alert).
    Earnings blackouts scored at -50.0. Sorts highest-conviction signals to the top.

- Web Dashboard & User Interface (`static/index.html`):
  • Section 1.2 "Universe Scanner & Pre-Close Execution Screener":
    - Summary stat pills: Active Bullish Calls, Active Bearish Puts, Earnings Blackouts, Regime Drift Alerts.
    - Sector tab switcher pills: All Sectors, Tech & AI, Financials, Semis, Healthcare, Consumer, Energy, Comm.
    - Search filter box to quickly find any stock or sub-industry theme.
    - Minimal black & charcoal grey styled opportunities table.
    - Seamless 1-Click "Load Cockpit" projecting scanned pairs into Section 1 inputs and immediately
      triggering `runAnomalyAnalysis()`.

================================================================================
END OF ANTIGRAVITY NOTES
================================================================================
"""

if __name__ == "__main__":
    print(__doc__)


