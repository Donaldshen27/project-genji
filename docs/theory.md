# An Integrated, Reproducible Architecture for Industry‑Aware Equity Trading

**Model 1: Industry Regime Engine · Model 2: Stock‑Alpha Engine · RL‑Controlled Portfolio Optimizer**
*Version 1.0 — Methods & Protocols*

---

## Abstract

We present a reproducible, production‑grade system that separates **signals** from **decisions** and unifies them in a single portfolio construction layer:

* **Model 1 (Industry Regime Engine)** detects **trend**, **volatility**, and **systemic correlation** regimes at the **industry** level and produces probabilities and transition risk.
* **Model 2 (Stock‑Alpha Engine)** predicts **industry‑relative forward returns** for individual stocks from fundamental, technical, and microstructure features.
* A **portfolio optimizer** combines both sources with a **risk model**, **transaction‑cost model**, and a **reinforcement‑learning (RL) controller** that adaptively tunes optimizer hyperparameters (risk aversion, turnover budgets, industry caps, volatility target) to current regimes. The RL agent never bypasses feasibility: a convex program (QP/SOCP) remains the single source of truth for weights.

We specify data, labels, feature engineering, models, training/validation (including **purged / combinatorial‑purged CV**), cost modeling, constraints, and RL training so that independent teams can replicate the pipeline end‑to‑end.

---

## 1. Problem Setup & Notation

* Universe at date (t): stocks ( \mathcal{U}_t = {1,\ldots,N_t} ) partitioned into industries ( \mathcal{I} = {1,\ldots, I} ) (e.g., GICS industry or Fama‑French‑style groupings).
* Prices: adjusted close (P_{i,t}); returns ( r_{i,t} = \ln(P_{i,t}/P_{i,t-1}) ).
* Tradeable industry proxy series: ( R^{(\text{ind})}_{j,t} ) (industry ETF or index total return) for ( j \in \mathcal{I} ).
* Portfolio weights ( w_t \in \mathbb{R}^{N_t} ) (long‑only or long/short).
* Risk model: factor exposure matrix (B_t \in \mathbb{R}^{N_t \times K}), factor covariance ( \Sigma^f_t ), idiosyncratic variances (D_t) (diag). Asset covariance ( \Sigma_t = B_t \Sigma^f_t B_t^\top + D_t ) (with shrinkage if estimated).
* Transaction cost model ( C(\Delta w_t) ) with L1 (turnover) and L2 (impact) components.
* Volatility target ( \sigma^**t ) and regime probabilities ( p^{(\cdot)}*{j,t} ) from Model 1.

We operate at daily EOD cadence; all decisions are made with information available at (t) and implemented for (t{+}1).

---

## 2. Data, Splits, and Reproducibility

### 2.1 Data Sources (point‑in‑time)

* **Prices/volumes**: adjusted OHLCV; corporate‑action clean.
* **Fundamentals**: PIT fundamentals & analyst estimates (avoid restatements).
* **Borrow/short data** (if long/short): availability and fees.
* **Industry mappings**: PIT industry membership.
* **Macro/market** (for Model 1): broad index, VIX or proxy, yield curve, credit spreads.
* **Industry proxies**: sector/industry ETFs or tradable indices.

### 2.2 Split Protocol

* **Chronological train/validation/test** with **expanding walk‑forward**:

  * Example (US 2003‑2025):

    * Train‑1: 2003‑2012 → Val‑1: 2013
    * Train‑2: 2003‑2013 → Val‑2: 2014
    * …
    * Hold‑out Test: 2020‑2025 (untouched until final).
* Within each train/val block for Model 2: **Purged K‑Fold** or **Combinatorial Purged Cross‑Validation (CPCV)** with an **embargo** window (e) days to handle label overlap.
* **Random seeds**: fix and record (e.g., `global_seed=42` for all libraries).
* **Version control**: freeze code, config, and environment (Conda/Poetry lock).

### 2.3 Environment (reference)

```text
Python ≥ 3.10
numpy, pandas, scipy
scikit-learn ≥ 1.3
xgboost ≥ 1.7  (or lightgbm ≥ 3.3)
statsmodels ≥ 0.14
cvxpy ≥ 1.4 (OSQP / Clarabel solvers)
ruptures ≥ 1.1 (or custom BOCPD implementation)
torch ≥ 2.0 (RL), stable-baselines3 ≥ 2.0
```

---

## 3. Model 1 — Industry Regime Engine

### 3.1 Targets (ontology)

Per industry (j), estimate probabilistic states at (t):

* **Trend** ( \mathcal{T} \in {\text{Bull}, \text{Bear}, \text{Sideways}} )
* **Volatility** ( \mathcal{V} \in {\text{Calm}, \text{Turbulent}, \text{Crisis}} )
* **Systemic correlation** ( \mathcal{S} \in {\text{Diversifying}, \text{Systemic}} )
* **Transition risk** ( \pi_{j,t}^{\text{flip}} ) (probability of regime change in next (k) days)

### 3.2 Features (industry‑level)

* Returns/momentum (5/10/21/63‑day), MA gaps, drawdown z‑scores
* Volatility (realized σ: 10/21/63/252d; range‑based)
* **Cross‑industry correlation**: DCC(1,1) on industry proxy returns → mean corr., top eigenvalue
* Macro deltas: yield‑curve slope, credit spread changes (optional)

### 3.3 Methods

* **Fast breaks**: **Bayesian Online Changepoint Detection (BOCPD)** on (volatility, trend slope) streams → break probability (b_{j,t}) & severity ( \nu_{j,t} ).
* **Persistent states**:

  * **HMM‑Trend**: 2–3 state Gaussian HMM on filtered returns (e.g., returns + MA gap).
  * **HMM‑Vol**: 2–3 state Gaussian HMM on log‑volatility.
* **Systemic correlation**: DCC(1,1) on an industry/sector panel; classify **Systemic** if mean corr. and top eigenvalue exceed percentile thresholds.

### 3.4 Fusion & Smoothing

* Convert detectors to state probabilities (p_{j,t}(\mathcal{T}), p_{j,t}(\mathcal{V}), p_{j,t}(\mathcal{S})).
* **Hierarchical reconciliation**: Market → Sector → Industry (cap contradictions).
* **Temporal smoothing**: EMA of probabilities (half‑life 2–5d fast, 10–15d slow).
* **Transition risk**: logistic/XGBoost forecaster on recent probability slopes, BOCPD severity, and macro deltas.

**Outputs per industry (j):**

```json
{
  "p_trend": {"bull":0.58, "bear":0.23, "sideways":0.19},
  "p_vol": {"calm":0.40, "turbulent":0.45, "crisis":0.15},
  "p_sys": {"diversifying":0.42, "systemic":0.58},
  "p_flip_10d": 0.27
}
```

---

## 4. Model 2 — Stock‑Alpha Engine (Industry‑Relative)

### 4.1 Labels

For stock (i) in industry (j(i)), define **industry‑relative forward returns**:
[
y_{i,t}^{(k)} = \left( \sum_{\tau=1}^{k} r_{i,t+\tau} \right) ;-; \left( \sum_{\tau=1}^{k} R^{(\text{ind})}_{j(i),t+\tau} \right),
\quad k \in {21, 63}
]
(Use log returns; align to prediction made at (t) for (t{+}1 \ldots t{+}k).)

### 4.2 Features (point‑in‑time)

* **Fundamental**: value (E/P, B/P, EV/EBITDA), quality (ROE, ROA, margins), growth, leverage, accruals; analyst estimate level and surprise.
* **Technical**: 12–1 momentum, 1–4 week reversal, MA gaps (20/50/200), RSI, volatility/drawdown stats.
* **Microstructure**: volume/turnover ratios, Amihud illiquidity, short interest/borrow.
* **Preprocessing**:

  * winsorize per vintage (1st/99th pct),
  * **demean and z‑score within industry** at each (t),
  * propagate only past information (no look‑ahead).

### 4.3 Models & Stacking

* Baselines: **Ridge** and **Elastic Net**.
* Non‑linear: **XGBoost/LightGBM** (handle missingness, interactions).
* Optional: small **TCN/LSTM** on short price windows per stock.
* **Stacking**: meta‑ridge blends predictions.
* **Calibration**: isotonic regression to map predicted scores → bp expected alpha for each horizon (k).
* **Uncertainty**: ensemble variance or residual bootstrap → ( \widehat{\text{Var}}[\alpha_{i,t}^{(k)}] ).

**Daily output per stock** (merged across horizons, variance‑weighted):
[
\mu^{\text{stock}}*{i,t} = \sum*{k} \omega^{(k)}*t , \widehat{\alpha}^{(k)}*{i,t},\quad
\omega^{(k)}*t \propto \left( \widehat{\text{Var}}[\alpha*{i,t}^{(k)}] \right)^{-1}
]

### 4.4 Neutralization (recommended)

* Because labels are industry‑relative, **industry neutrality** is implicit.
* To avoid unintended factor bets, optionally regress predicted alphas on risk‑model factors and take residuals:
  [
  \tilde{\mu}^{\text{stock}}*{t} = \mu^{\text{stock}}*{t} - B_t \left( (B_t^\top W B_t)^{-1} B_t^\top W \mu^{\text{stock}}_{t} \right)
  ]
  with cross‑sectional weight matrix (W) (e.g., inverse idio variance).

---

## 5. Portfolio Construction — Convex Core + RL Controller

### 5.1 Industry Views from Model 1

Convert regime probabilities to **industry expected‑return tilts** and **risk budgets**:

* **Tilt vector ( \mu^{\text{ind}}_t \in \mathbb{R}^{N_t} )**:
  Each stock (i) inherits a small additive tilt from its industry (j(i)):
  [
  \mu^{\text{ind}}*{i,t} = \kappa*{\text{trend}} \cdot \big(p_{j,t}(\text{Bull}) - p_{j,t}(\text{Bear})\big)
  ]
  (scale (\kappa_{\text{trend}}) in bp; optional negative tilt in Bear).
* **Risk budgets/caps**: map (p_{j,t}(\text{Systemic/Crisis})) to stricter **per‑industry exposure caps** and lower **volatility targets**.

### 5.2 Objective and Constraints

We solve a **cost‑aware mean‑variance program** with regime‑aware parameters:

[
\max_{w_t} \quad
w_t^\top (\tilde{\mu}^{\text{stock}}*{t} + \mu^{\text{ind}}*{t})
;-; \frac{\gamma_t}{2} w_t^\top \Sigma_t w_t
;-; \lambda_{1,t} |\Delta w_t|*1
;-; \lambda*{2,t} |\Delta w_t|_2^2
]

s.t.

* **Budget**: ( \mathbf{1}^\top w_t = 1 ) (long‑only) or net/gross bounds for L/S.
* **Per‑asset bounds**: ( l_i \le w_{i,t} \le u_i ).
* **Industry exposures**: ( A_{\text{ind}} w_t \in [L^{\text{ind}}_t, U^{\text{ind}}_t] ).
* **Style/country bands** (if benchmark‑relative): ( A_{\text{fac}} w_t \in [L^{\text{fac}}_t, U^{\text{fac}}_t] ).
* **Turnover**: ( |\Delta w_t|_1 \le \tau_t ).
* **Vol‑target**: enforce ( w_t^\top \Sigma_t w_t \le (\sigma^*_t)^2 ) or use it via (\gamma_t).

**Notes on implementability**

* L1 term handled with auxiliary variables → QP.
* Use **OSQP/Clarabel** (CPU) or a commercial solver.
* Covariance shrinkage (e.g., Ledoit‑Wolf) if estimating (\Sigma_t) in‑house.

### 5.3 RL Controller (Recommended Option A)

**Role:** RL **does not** set weights. It **chooses the optimizer’s dynamic hyperparameters** and caps in response to regimes and market conditions.

* **State (s_t)** (observed at EOD):

  * Regimes: (p_{j,t}(\mathcal{T}), p_{j,t}(\mathcal{V}), p_{j,t}(\mathcal{S}), \pi^{\text{flip}}_{j,t}) aggregated (mean, dispersion) and per‑industry.
  * Correlation diagnostics: DCC mean corr., top eigenvalue, cross‑industry concentration.
  * Alpha diagnostics: cross‑sectional spread of (\tilde{\mu}^{\text{stock}}_t), fraction > threshold, predicted uncertainty.
  * Portfolio diagnostics: current risk, active industry exposures, crowding proxy, turnover used YTD.
  * Cost state: recent realized slippage, liquidity buckets.
* **Action (a_t)** (continuous vector, bounded):
  [
  a_t = \big[\gamma_t, \lambda_{1,t}, \lambda_{2,t}, \tau_t, ; \text{per‑industry caps}, ; \sigma^*_t, ; \text{hedge intensity}\big]
  ]
* **Policy**: PPO (or CPO) with LSTM/TCN encoder (small) over recent states (e.g., 20–60 days).
* **Reward (r_{t+1})**:
  [
  r_{t+1} = \underbrace{w_t^\top r_{t+1}}_{\text{gross P&L}}

  * \underbrace{C(\Delta w_t)}_{\text{costs}}
  * \eta_{\text{DD}}\cdot \mathbb{1}{\text{drawdown breach}}
  * \eta_{\text{vol}}\cdot (\hat{\sigma}_{t+1}-\sigma^**t)*+^2
    ]
    (Use realized next‑day return; additional penalties for violating turnover/vol budgets.)
* **Safety**: All actions are **clipped** to pre‑approved ranges and passed to the convex optimizer; feasibility is guaranteed by the optimizer, not the RL policy.

**Training**

* **Environment**: market‑replay of daily returns with cost model; episodic windows spanning multiple regimes.
* **Curriculum**: start with fixed hyperparameters; gradually hand over degrees of freedom to RL.
* **Validation**: purged/CPCV across time‑blocks; selection by **out‑of‑sample DSR‑adjusted Sharpe**.
* **Monitoring**: action distributions, constraint‑violation counts, ablation vs. non‑RL baseline.

*(Advanced Option B—RL proposes weights with a projection/safety layer—is omitted here for brevity; the training environment and safety projection must be specified if used.)*

---

## 6. Backtesting, Validation, and Statistics

### 6.1 Leakage & Overlap Controls

* Purge look‑aheads at feature level; align labels to (t{+}1\ldots t{+}k).
* Embargo window (e) for CV folds > forecast horizon.
* Refit schedules: Model 1 weekly; Model 2 monthly/quarterly; RL controller weekly; risk model daily.

### 6.2 Cross‑Validation

* **Model 2**: CPCV over time (folds are contiguous blocks); embargo of (e \ge k_{\max}).
* **Regime forecaster (Model 1 transition)**: similar CPCV.
* **RL**: walk‑forward evaluation over disjoint periods; report per‑period metrics.

### 6.3 Reported Metrics

* **Performance**: Annualized return, volatility, Sharpe, Sortino, max DD, Calmar; **Deflated Sharpe Ratio (DSR)** on validation/test.
* **Costs**: turnover, realized slippage, borrow.
* **Attribution**: industry tilts (Model 1), stock selection (Model 2), RL controller effect (delta vs fixed‑param optimizer), trading costs.
* **Regime‑conditional**: performance by ( \mathcal{V} ) and ( \mathcal{S} ) states, and around transitions.
* **Breadth & risk**: effective number of bets, factor exposures, industry concentration, capacity (ADV‑scaled).

---

## 7. Execution & Operations

* Convert daily target (w_t) to orders with **POV/VWAP** schedule matched to liquidity buckets.
* Enforce **minimum trade thresholds** (e.g., 2–5 bps of NAV) to avoid dust.
* Real‑time risk checks: beta/industry/factor bands; pre‑trade cost estimates; borrow availability for shorts.
* Post‑trade analytics: slippage vs. model; feedback to cost parameters.

---

## 8. Reproducible Config Template (excerpt)

```yaml
# data.yaml
universe:
  region: "US"
  liquidity:
    min_adv_usd: 5e6
    min_price: 3.0
industries: "GICS"   # or "FF49"
horizons: [21, 63]

# model1.yaml
bocpd:
  base_hazard_days: 90
  severity_metric: "studentT"
hmm_trend:
  n_states: 3
  retrain_freq: "W"
hmm_vol:
  n_states: 3
dcc:
  window_days: 252
  systemic_thresholds:
    mean_corr_pct: 80
    eig1_pct: 80
smoothing:
  half_life_fast: 3
  half_life_slow: 12

# model2.yaml
features:
  winsor_pct: [0.01, 0.99]
  standardize_within: "industry"
models:
  - ridge: {alpha: 3.0}
  - xgboost: {max_depth: 6, n_estimators: 400, eta: 0.05, subsample: 0.8, colsample_bytree: 0.8}
stacking:
  meta: ridge
calibration: "isotonic"
neutralization: {style_neutral: true}

# optimizer.yaml
risk_model: "barra_like"
cov_shrinkage: "ledoit_wolf"
objective:
  gamma: 8.0          # RL can overwrite
  lambda1: 5e-4       # RL can overwrite
  lambda2: 2e-6       # RL can overwrite
turnover:
  tau: 0.25           # of NAV per day; RL can overwrite
bounds:
  long_only: true
  per_name: [0.0, 0.02]
  per_industry: [-0.05, 0.05]
vol_target:
  base: 0.12          # annualized; RL can overwrite
regime_tilts:
  kappa_trend_bp: 10

# rl.yaml
algo: "PPO"
policy: "LSTM"
state_history_days: 40
clip_range: 0.2
gae_lambda: 0.95
gamma: 0.99
learning_rate: 3e-4
batch_size: 4096
train_steps_per_update: 5
entropy_coef: 0.005
action_bounds:
  gamma: [4.0, 20.0]
  lambda1: [1e-4, 2e-3]
  lambda2: [1e-6, 1e-4]
  tau: [0.05, 0.35]
  vol_target: [0.08, 0.15]
```

---

## 9. End‑to‑End Algorithm (pseudocode)

```python
for each trading day t:

    # === Model 2: stock alphas ===
    X_t = build_features_stocks(t)                 # PIT features, industry-standardized
    mu_stock_t, var_stock_t = model2_predict(X_t)  # bp, per stock

    # === Model 1: industry regimes ===
    Z_t = build_industry_features(t)               # returns, vol, DCC, macro
    P_trend_t, P_vol_t, P_sys_t, P_flip_t = model1_predict(Z_t)
    mu_ind_t = map_regimes_to_industry_tilts(P_trend_t)       # small bp adders
    caps_t, vol_target_t = regime_to_caps_vol(P_vol_t, P_sys_t)

    # === Risk & costs ===
    Sigma_t, B_t = build_risk_model(t)             # factor + idio; shrinkage as needed
    cost_params_t = estimate_costs(t)              # L1 turnover, L2 impact

    # === RL controller chooses optimizer knobs ===
    s_t = build_rl_state(P_trend_t, P_vol_t, P_sys_t, P_flip_t,
                         DCC_stats(t), mu_stock_t, var_stock_t,
                         current_portfolio(), recent_costs(), caps_t, vol_target_t)
    a_t = RL_policy(s_t)                           # [gamma, lambda1, lambda2, tau, per-ind caps, vol_target]
    gamma_t, lambda1_t, lambda2_t, tau_t, ind_caps_t, sigma_star_t = sanitize_actions(a_t, caps_t, vol_target_t)

    # === Convex optimization ===
    mu_total_t = neutralize(mu_stock_t) + lift_to_stock(mu_ind_t)
    w_star_t = solve_qp(mu_total_t, Sigma_t, gamma_t,
                        lambda1_t, lambda2_t, tau_t,
                        bounds=per_name_bounds, industry_caps=ind_caps_t,
                        factor_bands=factor_bands, vol_target=sigma_star_t)

    # === Execution ===
    orders_t = schedule_trades(w_prev, w_star_t, cost_params_t)   # POV/VWAP, min trade threshold
    send_orders(orders_t)

    # === RL reward (next day) will be computed from realized P&L and costs ===
```

---

## 10. Ablations & Sanity Checks

* **No‑RL baseline**: fixed (\gamma,\lambda_1,\tau,\sigma^*) and regime‑aware caps only; compare Sharpe/DSR.
* **No‑regime tilt**: set (\mu^{\text{ind}}=0) to measure incremental value of Model 1 in the optimizer (not in alpha).
* **Cost sensitivity**: double L1/L2; verify monotone trade‑off.
* **Neutralization off**: check factor drift; confirm optimizer constraints still contain it.
* **Regime‑conditional attribution**: gains should concentrate when Model 1 signals elevated predictability or when DCC flags systemic risk (risk‑reduction effect).

---

## 11. Limitations & Risk Controls

* **Data sparsity in crises**: use multi‑cycle CV; do not tune only on crisis windows.
* **Model creep**: SHAP drift/feature drift dashboards; rule‑based fallbacks if drift exceeds thresholds.
* **Capacity**: enforce ADV caps; stress test impact non‑linearity.
* **Governance**: parameter change log; frozen configs; change‑management approvals.

---

## 12. Releasing Artifacts for Reproducibility

* **/configs**: YAML files from Section 8.
* **/data_schema**: parquet/arrow schemas; PIT keys; industry mappings.
* **/notebooks**: diagnostics for calibration, CV, attribution.
* **/reports**: fixed‑seed validation & test summaries (metrics + plots).
* **/model_cards**: Model 1, Model 2, RL policy with inputs/outputs, training windows, seeds, and known caveats.

---

## Appendix A — Key Formulae

* **DCC(1,1)** updates (per industry pair):
  ( Q_t = (1-a-b)\bar{Q} + a,\epsilon_{t-1}\epsilon_{t-1}^\top + b, Q_{t-1} ),
  ( R_t = \text{diag}(Q_t)^{-1/2} Q_t \text{diag}(Q_t)^{-1/2} ).
* **Ledoit–Wolf shrinkage** (sketch): ( \Sigma^\text{shrunk} = \delta F + (1-\delta)S ), target (F) (e.g., constant correlation), sample (S); (\delta) estimated from moments.
* **Deflated Sharpe Ratio (DSR)**: compute Sharpe on OOS and deflate by number of trials/effective tests and non‑normality parameters (kurtosis, skew).

---

## Appendix B — Hyperparameter Defaults (starting points)

* **Model 1**:

  * BOCPD base hazard ≈ 1/90; severity threshold ~0.7 (tune via CPCV).
  * HMM‑Trend: 3 states; HMM‑Vol: 3 states; retrain weekly.
  * DCC window 252d; systemic thresholds at 80th percentile for mean corr. & eig‑1.
* **Model 2**:

  * XGB: depth 6, 400 trees, eta 0.05, subsample 0.8, colsample 0.8; early stopping 40.
  * Stacking meta ridge α=3.
  * Calibration: isotonic per horizon.
* **Optimizer**:

  * Base (\gamma=8), (\lambda_1=5\cdot10^{-4}), (\lambda_2=2\cdot10^{-6}), (\tau=25%) NAV/day, vol target 12% p.a.
* **RL (PPO)**:

  * LR (3\cdot10^{-4}), GAE λ=0.95, γ=0.99, clip 0.2, entropy 0.005, batch 4k, 5 epochs/update, LSTM hidden 64.
  * Action bounds as in config; observation history 40 days.

---

## Conclusion

This document specifies a clear, modular, and **reproducible** methodology that:

1. **Separates** regime detection (industry context) and stock alpha (idiosyncratic edge),
2. **Unifies** them in a disciplined portfolio optimizer with a proper risk and cost model, and
3. Uses **RL as a controller**—not a free‑hand trader—to adapt aggressiveness to prevailing regimes while preserving feasibility and governance.

Use the provided configs, schedules, and pseudocode to stand up a baseline, then iterate with ablations and monitoring until operationally stable.
