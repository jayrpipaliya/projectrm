# Project RM: “Relatively Mispriced” (v3.4.0 Production)

Quantitative framework designed to identify and monetize localized anomalies in single-stock equities by orthogonalizing asset moves against broad market benchmarks (e.g., SPY, QQQ, DIA) and sector ETFs (e.g., IGV, SMH, XLK, XLE).

---

## 🌐 Production Deployment
* **Live Web Dashboard:** [https://project-rm-ten.vercel.app](https://project-rm-ten.vercel.app)
* **Local Development Server:** `http://localhost:8050`

---

## 📌 Architecture & Seven Edge Application Modules

The system is partitioned into seven dedicated modules ("Edges"):

1. **Dynamic Edge (`cockpit`)**: Real-time orthogonal dual-factor Kalman filter tracking, instantaneous innovation $Z$-scores, continuous Ornstein-Uhlenbeck (OU) half-life reversion, two-sided Page's CUSUM drift detection, and unified trade execution blueprint.
2. **Rare Edge (`gold-ledger`)**: Watchlist Champion Registry for 21 core liquid assets, 3:45–3:58 PM EST Market-On-Close (MOC) execution screener, and real-time barrier breach signals with clean grey border/text styling.
3. **Tri-Surface Edge (`tenor-sweep`)**: 3D joint parameter optimization manifold $\mathcal{S}(\sigma^*, \tau^*, H^*)$, 2D parameter heatmaps (Profit Factor, Win Rate, ROI %, Dollar P&L), $3 \times 3$ plateau smoothing regularization, and 65/35 walk-forward OOS cross-validation.
4. **Cadence Edge (`income-strategy`)**: High-velocity weekly cashflow engine, 3:45–4:00 PM MOC standard deviation ($\sigma$) proximity radar ($\delta_\sigma = \max(0.0, \, \sigma^* - |Z_t|)$), and blended multi-asset portfolio cashflow simulation.
5. **Atlas Edge (`price-action`)**: Multi-timeframe technical action, candlestick geometry, and Kalman orthogonal reconstructed price trajectory overlays.
6. **Dossier Edge (`trades-log`)**: Forensic trade journal, individual trade lifecycle simulation, and corporate earnings announcement blackout gating ($\{t_{\text{earn}}, t_{\text{earn}}+1\}$).
7. **Diagnostic Edge (`diagnostics`)**: Interactive Any-Ticker Explorer (input box to pull on-demand OHLCV bars & option chains for any symbol), econometric stationarity tests (ADF & Phillips-Perron), innovation whiteness (Ljung-Box), and Discrete Semi-Markov regime duration hazard analytics.

---

## 🔬 Mathematical Formulations

### 1. Dual-Factor Orthogonal State-Space Kalman Filter (QMLE)
* **Sector Orthogonalization:** $r_{S \perp M, t} = r_{S, t} - \frac{\text{Cov}(r_M, r_S)}{\text{Var}(r_M)} r_{M, t}$
* **Measurement Equation:** $y_t = \mathbf{X}_t \boldsymbol{\beta}_t + v_t$, where $v_t \sim \mathcal{N}(0, R)$
* **State Equation:** $\boldsymbol{\beta}_t = \boldsymbol{\beta}_{t-1} + \mathbf{w}_t$, where $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$
* **Standardized Innovation:** $Z_t = \frac{e_t}{\sqrt{S_t}} = \frac{y_t - \mathbf{X}_t \hat{\boldsymbol{\beta}}_{t|t-1}}{\sqrt{\mathbf{X}_t \mathbf{P}_{t|t-1} \mathbf{X}_t^T + R}}$
* **QMLE Optimization:** $\arg\min_{\mathbf{Q}, R} \frac{1}{2} \sum_{t=1}^T \left( \ln(2\pi S_t) + \frac{e_t^2}{S_t} \right)$ via L-BFGS-B.

### 2. Ornstein-Uhlenbeck (OU) Mean-Reversion SDE
* **Continuous SDE:** $dZ_t = \theta (\mu - Z_t) dt + \sigma_{OU} dW_t$
* **Discrete Calibration:** $Z_t = a + b Z_{t-1} + \epsilon_t \implies \theta = -\frac{\ln(b)}{\Delta t}$
* **Half-Life:** $t_{1/2} = \frac{\ln(2)}{\theta} = -\frac{\ln(2) \cdot \Delta t}{\ln(b)}$ (in trading days).

### 3. Page's Two-Sided CUSUM Drift Detector
* **Upper & Lower Accumulators:** $S_t^+ = \max(0, S_{t-1}^+ + Z_t - k)$, $S_t^- = \max(0, S_{t-1}^- - Z_t - k)$ with $k = 0.5, h = 4.0$.

### 4. Discrete Semi-Markov Regime Duration Model
* **Embedded Transition Matrix:** $\mathbf{P} = [p_{ij}]_{3 \times 3}, p_{ii} = 0$.
* **Discrete Weibull Hazard Distribution:** $S(t) = q^{t^\beta}$, hazard rate $\lambda(t) = 1 - q^{t^\beta - (t-1)^\beta}$.
* **Conditional Exit Probability:** $\mathbb{P}(\text{Exit} \le N + k \mid T > N) = \frac{S(N) - S(N + k)}{S(N)}$.

### 5. 3D Parameter Manifold Optimization & Plateau Regularization
* **Cell Utility:** $U(\sigma, H) = \left(\frac{WR}{100}\right) \times \overline{ROI} \times \sqrt{N_{\text{trades}}}$
* **3x3 Plateau Smoothing:** $P(\sigma, H) = \frac{1}{|\mathcal{N}|} \sum_{(\sigma', H') \in \mathcal{N}} U(\sigma', H')$
* **Walk-Forward Validation:** 65% In-Sample / 35% Out-of-Sample, efficiency ratio $\eta = \frac{WR_{\text{OOS}}}{WR_{\text{IS}}}$.

### 6. Vectorized Closed-Form Black-Scholes ATM Derivative Engine
* **Analytic Solution:** Closed-form evaluation using Gauss error function $\text{erf}(x)$, with ATM strike $K = \text{round}(S_{\text{entry}})$ and 20-day historical realized volatility proxy.
* **Corporate Earnings Blackout:** Excludes trades entering on $\{t_{\text{earn}}, t_{\text{earn}}+1\}$ to prevent binary volatility collapse.

---

## 🚀 Quick Start

### 1. Configure Environment
```env
MASSIVE_API_KEY=your_polygon_api_key_here
```

### 2. Run the Web Dashboard Locally
```powershell
py -u server.py
```
Open `http://localhost:8050` in your browser.

---

## 🗂️ File Directory
```
Project RM/
├── .env                       # API Key & environment configuration
├── config.py                  # Configuration & path manager
├── api_client.py              # Massive / Polygon Stock & Options client with fallbacks
├── yfinance_loader.py         # Multi-year historical loader & fallback engine
├── warehouse.py               # SQLite Data Warehouse (9 normalized tables)
├── residual_model.py          # Dual-State Kalman Filter, OU SDE, CUSUM & Black-Scholes
├── semi_markov.py             # Semi-Markov Model with Discrete Weibull hazard fitting
├── threshold_optimizer.py      # 3D parameter manifold optimizer with plateau smoothing
├── scanner_engine.py          # Multi-universe screening & factor calculations
├── server.py                  # Local HTTP server & API endpoint router
├── api/
│   └── index.py               # Vercel serverless request handler
├── Summary.py                 # Comprehensive quantitative architecture specification
├── static/
│   └── index.html             # Multi-module quant cockpit (All 7 Edges)
└── data/
    └── warehouse.db           # SQLite local storage
```
