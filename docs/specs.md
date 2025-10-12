Below is a **production‑level, zero‑ambiguity** specification for **all sprints (S0–S4)** of the industry‑aware equity trading system. It **locks scope, interfaces, configs, tests, metrics, and acceptance gates**, building directly on the **plan** and **Sprint 0 specs** you provided.  

---

## 0) Global, non‑negotiable invariants (apply to **every** sprint)

**Repository & environment**

* Repo name: `quant-system` (single monorepo).
* Package manager: **uv**; Python **3.11** pinned by `.python-version`; **`uv.lock` is committed** and must be used with `uv run --locked` / `uv sync --locked`. CI uses `astral-sh/setup-uv`. 
* Core libs: `pyqlib` for backtest/data layer; `cvxpy` + **OSQP/Clarabel** for QP; `scikit-learn`, `xgboost` for ML; `torch`/`stable-baselines3` for RL (S3+). Exact ranges remain as in `pyproject.toml`; resolved versions are locked in `uv.lock`. 
* Directory skeleton (authoritative): as in the Sprint 0 spec (configs/, src/, tests/, scripts/, data/, data/qlib/, reports/). **No new top‑level folders may be added without RFC.** 

**Data & calendars**

* Regions: US & CN. **Qlib `region` MUST equal the dataset at `provider_uri`** (`data/qlib/us_data`, `data/qlib/cn_data`). 
* Prices: US via **yfinance**; CN via **Tushare** (A‑share). **Persist raw with actions and derive explicit adj factors** as in S0 spec. 
* Universe: **Monthly reconstitution** (last trading day of region calendar), **N=800**, eligibility `price ≥ 3.0`; CN extra rules per S0 spec. 

**Modeling & optimization backbone**

* **Signal/decision separation**: Model 2 (stock alpha) → expected alphas; Model 1 (industry regimes) → tilts/caps; **single convex optimizer is the source of truth** for weights; RL (S3) only tunes optimizer hyperparameters within pre‑approved bounds. 
* Risk model: covariance with Ledoit‑Wolf shrinkage (S0 baseline), or factor+idio in later sprints if added; must expose (\Sigma_t) and (optionally) (B_t). 
* Cost model: L1 (turnover) + L2 (impact); parameters are scalar or liquidity‑bucketed (S2). 

**Reproducibility & seeds**

* Global seed: **42** for all libraries; model seeds must be **passed via configs** and asserted in logs. Walk‑forward/CPCV splits **exactly** match the plan for any sprint that trains models. 

**Ops, telemetry, and CI gates**

* CI must run: `uv sync --locked --all-extras --dev`, `ruff`, `pytest -q`. **All tests must pass (0 failures)**; **coverage ≥ 80%** across `src/` (enforced from S1). CI template is the S0 workflow with coverage step added in S1. 
* Logging: Python `logging` JSON lines, UTC timestamps, level `INFO` by default, `DEBUG` on `--debug`. Logs must include `git_sha`, `config_hash`, `global_seed`, `region`, and split window IDs.
* Reports written under `reports/<sprint>_<region>/…`. **HTML reports are immutable artifacts** (filename includes short `git_sha`).

---

## 1) Sprint 0 — **Deterministic baseline (US & CN)** — *frozen* (already provided; made production‑grade here)

**Objective (unchanged):** One‑command E2E baseline per region: data → Model 2 → risk/cost QP → Qlib backtest → HTML report. **No RL; no Model 1.** 

**Deliverables & interfaces (authoritative)**

* `Makefile` targets: `data_us`, `data_cn`, `train_us`, `train_cn`, `opt_US`, `opt_CN`, `backtest_us`, `backtest_cn`, `report_US`, `report_CN`, `e2e_us`, `e2e_cn`. **These names and dependencies are frozen.** 
* Config files (frozen keys): `configs/data_{us,cn}.yaml`, `configs/model2_{region}.yaml`, `configs/optimizer.yaml`, `configs/backtest_{region}.yaml`. Keys as in S0 spec; **new keys are disallowed in S0**. 
* Modules must exist and expose CLIs:

  * `python -m src.data.ingest_yf_us --config …`
  * `python -m src.data.ingest_ak_cn --config …`
  * `python -m src.data.build_universe --config …`
  * `python -m src.data.build_qlib_dataset --config …`
  * `python -m src.model2.train --config …`
  * `python -m src.optimizer.solve_qp --config configs/optimizer.yaml --region {US|CN}`
  * `python -m src.backtest.run_backtest --config configs/backtest_{region}.yaml`
  * `python -m src.backtest.report --region {US|CN} --out reports/sprint0_{REGION}/report.html` 

**Non‑functional requirements**

* Runtime budget (from clean clone):

  * `make data_us` ≤ 6h on 8 vCPU / 32 GB RAM; `make data_cn` ≤ 6h; training per region ≤ 2h; backtest ≤ 1h.
* Memory: peak RSS ≤ 24 GB on the above profile.
* CI total ≤ 30 min (small smoke subsets; full runs are manual).

**Acceptance tests (must all pass)**

* *Data smoke*: `qlib.init(provider_uri=…, region=…)` returns without error for both regions. 
* *Universe rules*: `tests/test_universe_rules.py` asserts counts, price/volume filters, month‑end boundaries (±0 business‑day tolerance = 0). 
* *Model 2 CV hygiene*: CPCV with **63‑day embargo** is honored; deterministic predictions under seed=42. 
* *Optimizer behavior*: tightening `turnover.tau` strictly decreases turnover; lowering `vol_target.base` strictly decreases (w^\top \Sigma w); industry caps respected within **1e‑6**. 
* *Backtest parity*: toy 3‑asset case P&L within **±1 bp** of analytical; full pipeline emits HTML with metrics (Ann. Return/Vol/Sharpe, Max DD, Turnover, Costs, industry exposures, holdings). 

> This codifies S0 as the immutable baseline. All items mirror your S0 spec and the plan’s separation of signals/decisions.  

---

## 2) Sprint 1 — **Industry Regime Engine + PIT industries**

**Objective:** Implement **Model 1** (per‑industry **trend**, **volatility**, **systemic correlation**, **transition risk**) and **map to optimizer inputs** (industry tilts + exposure caps + volatility target adjustments). Introduce **PIT industry membership** (CN) and keep **US “frozen‑at‑first‑seen”** as specified.  

### 2.1 Scope (included / excluded)

* **Included**

  * HMM‑Trend (3‑state), HMM‑Vol (3‑state) per industry; BOCPD fast breaks; DCC(1,1) cross‑industry correlation panel; probability smoothing; hierarchical reconciliation; transition‑risk forecaster. 
  * Deterministic mapping → `mu_ind_t` (bp tilts), per‑industry exposure caps, `vol_target` scaling. 
  * CN **PIT** industry membership snapshots; **US remains static** (frozen‑at‑first‑seen) in S1 per S0 spec future‑work note. 
* **Excluded**

  * RL controller (S3).
  * Any alternative factor risk model (stick to S0 covariance).

### 2.2 Artifacts & modules

* New config: `configs/model1_{region}.yaml` (**authoritative keys below**).
* New modules:

  * `src/model1/features.py` — build industry panel (returns, vol streams, DCC inputs).
  * `src/model1/hmm.py` — fit/predict HMM Trend/Vol (Gaussian).
  * `src/model1/bocpd.py` — BOCPD on vol & slope streams.
  * `src/model1/dcc.py` — DCC(1,1) using rolling 252d window; outputs mean corr & top eigenvalue.
  * `src/model1/fusion.py` — probability fusion (EMAs), hierarchical reconciliation (Market→Sector→Industry).
  * `src/model1/forecaster.py` — transition risk forecaster (logistic or XGBoost).
  * `src/model1/mapping.py` — deterministic map to `{industry_tilts_bp, industry_caps, vol_target}`.
  * `src/data/industries_cn_pit.py` — PIT industry snapshots for CN.
* CLI:

  * `python -m src.model1.run --config configs/model1_{region}.yaml --out data/model1/{region}/states.parquet`
  * Outputs a **daily table**: `date, industry_id, p_trend_{bull,bear,sideways}, p_vol_{calm,turbulent,crisis}, p_sys_{diversifying,systemic}, p_flip_10d, tilt_bp, cap_long, cap_short`.

### 2.3 Authoritative configs

```yaml
# configs/model1_{region}.yaml
hmm_trend: { n_states: 3, retrain_freq: "W", random_state: 42 }
hmm_vol:   { n_states: 3, retrain_freq: "W", random_state: 42 }
bocpd: { base_hazard_days: 90, severity_metric: "studentT", prob_threshold: 0.7 }
dcc: { window_days: 252, systemic_thresholds: { mean_corr_pct: 80, eig1_pct: 80 } }
smoothing: { half_life_fast: 3, half_life_slow: 12 }   # EMA on probabilities
mapping:
  kappa_trend_bp: 10              # tilt scale in basis points
  cap_widen_bp: 0                 # S1: keep per-ind caps from optimizer.yaml unless systemic/crisis
  crisis_tighten_factor: 0.5      # multiply per-ind caps when p_sys(systemic)>=0.5 or p_vol(crisis)>=0.3
  vol_target_floor: 0.08          # annualized
  vol_target_ceiling: 0.15
  vol_target_base: 0.12
  vol_target_map:                 # linear interpolation on p_vol(crisis)
    crisis_weight_floor: 0.0, crisis_weight_ceiling: 1.0
pit_industries:
  cn_source: "Tushare_sw_l1_snapshot"
  us_mode: "frozen_at_first_seen" # S1 policy
```

(Parameters follow the plan’s defaults and S0 future‑sprint guidance.)  

### 2.4 Integration into optimizer

* Extend `src.optimizer.solve_qp` to **load** `tilt_bp` and **add** `mu_ind_t` (converted to return units) to `mu_total_t`; **tighten** per‑industry caps when systemic/crisis thresholds are met; **scale** volatility target inside the same QP run. 
* **No change to API**: still called via `opt_{US|CN}`.

### 2.5 Tests & acceptance (must all pass)

* *Probability sanity*: for each `date, industry_id`, (\sum p_\text{trend} = 1), (\sum p_\text{vol} = 1), (\sum p_\text{sys} = 1) (abs error ≤ 1e‑9); unit test `test_model1_probs_sum_to_one.py`. 
* *Reconciliation*: child probs deviate from parent at most **0.15** unless overridden by `cap contradictions` rule; test `test_model1_hierarchy.py`. 
* *Mapping monotonicity*: increasing `p_sys(systemic)` or `p_vol(crisis)` **never** widens caps nor raises vol target; increasing (p_\text{trend}(\text{Bull})-p_\text{trend}(\text{Bear})) **strictly increases** tilt (bp). `test_model1_mapping_monotone.py`. 
* *Integration*: when `p_sys(systemic)≥0.5`, the effective per‑industry exposure interval is multiplied by `0.5±1e‑6`; `w^\top\Sigma w` ≤ mapped vol‑target². `test_optimizer_caps_from_model1.py`. 
* *PIT membership (CN)*: daily snapshot **never** assigns multiple industries per stock; monthly diffs consistent (≤ 5% churn MoM on average over 2015‑2024). `test_cn_pit_industries.py`. (US stays S0 “frozen‑at‑first‑seen” per policy.) 
* *Reporting*: `report_{REGION}` adds a “Regime Panel” section (charts + table of tilts/caps/vol target).

**Definition of Done (S1):** All above tests pass; CI coverage ≥ 80%; `make e2e_us`/`e2e_cn` produce reports including regime sections; optimizer behavior respects model‑derived caps/vol mapping across **10 randomized dates** in 2022–2024 with **0 violations**.

---

## 3) Sprint 2 — **Cost calibration & execution realism**

**Objective:** Replace fixed λ₁/λ₂ with **liquidity‑bucketed** parameters fit from **realized shortfall**; introduce **POV/VWAP** scheduling; enforce **ADV caps** and **min trade notional** at execution layer.  

### 3.1 Artifacts & modules

* Config changes (extend existing):

  * `configs/optimizer.yaml` gains:

    ```yaml
    costs:
      bucket_scheme: { by: "adv_percentile", n_buckets: 6, edges_pct: [0,5,10,25,50,75,100] }
      fit_method: "winsorized_ols"     # or "quantile_reg"
      lambda1_per_bucket_bp: null      # learned & persisted
      lambda2_per_bucket: null         # learned & persisted
    execution:
      style: "POV"                     # or "VWAP"
      pov: { target_pov: 0.1, min_child_notional_usd: 1000 }
      vwap: { buckets: 6 }
      min_trade_nav_bp: 2
      adv_cap_bp: 5
    ```
* New modules:

  * `src.optimizer.costs_fit.py` — fit λ₁/λ₂ using historical (simulated) shortfall by bucket.
  * `src.backtest.execution.py` — **POV/VWAP** order scheduling for Qlib executor; applies per‑bucket pacing, `min_trade_nav_bp`, and `adv_cap_bp`.
  * `src.backtest.slippage_log.py` — capture realized shortfall vs mid for each child order.

### 3.2 Fitting protocol

* Bucketing: rolling **252‑day median ADV**; bin by edges `[0,5),[5,10),[10,25),[25,50),[50,75),[75,100]` percentile of **universe** on each day.
* Response: **bps shortfall** per order; Features: (|\Delta w|), symbol ADV bucket, volatility bucket (10/21d realized).
* Fit **linear λ₁ (L1)** on absolute ( |\Delta w|) and **quadratic λ₂ (L2)** on ( |\Delta w|^2) by bucket with **winsorized OLS (1/99)**. Persist per‑bucket parameters to `data/costs/{region}/lambda_params.parquet`.

### 3.3 Optimizer & executor changes

* `solve_qp` reads the **current day’s** bucket for each security and loads λ₁/λ₂ from the persisted table; objective remains **cost‑aware mean‑variance**. 
* `WeightsToOrdersStrategy` is replaced by `ScheduledOrdersStrategy` calling `execution.py` to produce child orders according to `execution.style`.
* ADV cap: per name, **aggregate child notional** per day ≤ `adv_cap_bp` of **ADV × price**; trades smaller than `min_trade_nav_bp` NAV are dropped.

### 3.4 Tests & acceptance

* *Monotone cost sensitivity*: doubling λ₁ halves turnover (±10%) on a fixed alpha snapshot; doubling λ₂ halves **impact‑proxied** risk increase (Δ(w^\top\Sigma w)) for same alphas; tests `test_cost_monotonicity.py`. 
* *Execution policy*: `POV` schedule uses **target_pov ± 0.02** across the session; `VWAP` child order counts match config bucketization; test `test_execution_schedules.py`.
* *Caps & thresholds*: no symbol exceeds `adv_cap_bp` and no child order below `min_child_notional_usd`; test `test_exec_caps_thresholds.py`.
* *Business acceptance*: on 2022‑01‑01 → 2024‑12‑31 OOS, **median daily shortfall** under `POV 10%` is **≤ baseline MOO – 5 bps**; `IQR` (p75–p25) improves by **≥ 10%**. Data & test harness fixed in repo (repro runs).

**Definition of Done (S2):** All the above tests pass; reports include **Cost Diagnostics** (per‑bucket λ₁/λ₂ table, shortfall distributions, turnover vs. cost curves). The optimizer and executor are **feature‑flagged** to allow toggling S2 on/off in CI integration tests.

---

## 4) Sprint 3 — **RL controller (PPO) for optimizer knobs**

**Objective:** Add **RL controller** that **only** selects optimizer hyperparameters (γ, λ₁, λ₂, τ, per‑industry caps multiplier, vol‑target) within **safe bounds**. The QP remains the single source of truth. Train on market‑replay with cost model from S2 and regimes from S1. 

### 4.1 Observation/action/reward (**exact** definitions)

* **State (s_t)** (EOD, vectorized):

  * Aggregates of Model 1: mean & std across industries for (p_\text{trend}, p_\text{vol}, p_\text{sys}); top‑eigenvalue and mean DCC correlation; mean (\pi^{flip}_{10d}). (dims fixed by number of states). 
  * Alpha diagnostics: cross‑sectional stdev of (\tilde{\mu}^{stock}_t); fraction of names with (|\mu|>) 10 bps; ensemble variance percentile. 
  * Portfolio diagnostics: current (w^\top\Sigma w), realized turnover (30d), active industry exposures L∞, crowding proxy (Herfindahl).
  * Cost state: past‑5d average shortfall, bucket mix.
  * **History length:** 40 days; encoder: LSTM(64).
* **Action (a_t)** (continuous, element‑wise clipped to bounds):

  * (\gamma \in [4.0, 20.0]), (\lambda_1 \in [1e{-}4, 2e{-}3]), (\lambda_2 \in [1e{-}6, 1e{-}4]), (\tau \in [0.05, 0.35]), **caps multiplier** (\in [0.5, 1.0]), **vol‑target** (\in [0.08, 0.15]). **These bounds are final.** 
* **Reward (r_{t+1})**:
  [
  r_{t+1} = (w_t^\top r_{t+1}) - C(\Delta w_t) - \eta_{DD}\cdot \mathbf{1}*{DD} - \eta*{vol}\cdot(\hat\sigma_{t+1}-\sigma^**t)*+^2,
  ]
  with (\eta_{DD}=5), (\eta_{vol}=50). `DD` triggers when peak‑to‑trough over last 63d > 10%.

### 4.2 Training protocol (deterministic)

* Algo: **PPO** with `clip=0.2, gae_lambda=0.95, gamma=0.99, lr=3e-4, batch=4096, epochs=5`, LSTM policy (hidden=64), seed=42. 
* **Environment:** market replay using daily returns, Model 1 outputs, and S2 cost model. Episodes are contiguous 2‑year windows; curriculum: first 20k steps with fixed knobs (behavioural cloning warm‑up off), then fully active actions.
* Splits (US & CN identical): Train: **2010‑01‑01–2018‑12‑31**, Val: **2019‑01‑01–2019‑12‑31**, Test (hold‑out): **2020‑01‑01–2025‑06‑30**. (Exact dates frozen; extend test forward in future releases.) 
* Checkpoints every 1M steps; **early stop** on **Val DSR** plateaus (no improvement ≥ 0.02 over 5 evals).

### 4.3 Integration

* New module: `src.rl.controller.py` exposing `choose_knobs(state)->Knobs`.
* `solve_qp` receives the sanitized knobs; if an infeasible combination is proposed (e.g., τ too small to reach target), **QP feasibility dominates** and RL action is clipped; violation counters recorded. 
* Feature flag: `configs/rl.yaml` with `enabled: true|false` and all PPO hyperparams mirrored from the plan defaults. 

### 4.4 Tests & acceptance

* *Safety*: across the entire **2020–2025H1** test, **0 feasibility violations** (QP solved every day). `test_rl_safety_projection.py`. 
* *Performance gating*: Split test into **three disjoint periods**: 2020–2021, 2022–2023, 2024–H1 2025. In **each** period:

  * `Sharpe_RL ≥ Sharpe_fixed + 0.10`, and
  * `DSR_RL ≥ DSR_fixed + 0.10` (with DSR computed as in plan). If any period fails, build fails. 
* *Action bounds adherence*: histogram of actions shows **100% within bounds**; counts per bound‑hit < 5% per dimension.
* *Ablations*: *No‑RL* and *No‑regime tilt* runs reproduced; **delta metrics** added to report. 

**Definition of Done (S3):** All tests pass; report includes **RL diagnostics** (action distributions, reward curves, constraint‑violation counts). RL can be disabled via config with no code changes.

---

## 5) Sprint 4 — **Governance, reporting, & change control**

**Objective:** Productionize governance: **model cards**, **frozen config bundles**, **automated nightly reports**, **ablations**, and **change‑management** with approvals and audit trail.  

### 5.1 Artifacts

* `/model_cards/`:

  * `model1_{region}.md`: inputs/outputs, training windows, seeds, known caveats; metrics by regime.
  * `model2_{region}.md`: features/labels, CPCV scheme, calibration plots.
  * `rl_policy_{region}.md`: obs/action/reward, bounds, training logs, eval metrics. 
* `/configs/bundles/<date>_<git_sha>/` — **immutable** tarball of all YAMLs used in a run, with `config_hash.txt`.
* Nightly CI job `nightly.yml` that runs **backtests** for last 3 calendar years per region and publishes `/reports/<YYYY-MM-DD>/summary.html` with performance/exposures/turnover/cost attribution/regime breakdowns. 
* `/reports/ablations/` — standardized *No‑RL*, *No‑regime tilt*, *Cost sensitivity (×2)* outputs. 

### 5.2 Process & gates

* **Change log** (`CHANGELOG.md`) entry required for **any** parameter change in `configs/`; PR template enforces it.
* **Two‑person review** for changes affecting `src/model1`, `src/model2`, `src/optimizer`, `src/rl`; CI must attach **before/after** ablation diffs.
* **Drift monitoring**: SHAP/feature drift dashboards for Model 2; if drift **KS‑p < 0.01** on any top‑10 feature, auto‑flag run as “needs review” and add banner in report. 
* **Capacity checks**: ADV caps and impact stress run weekly with “double impact” scenario; results appended to nightly report. 

### 5.3 Tests & acceptance

* *Artifact completeness*: model cards render with all required fields; config bundles contain **every** YAML referenced in logs; SHA in report matches `git rev-parse --short HEAD`.
* *Nightly job*: completes in < 3 h on a 16‑vCPU/64 GB runner; publishes summary page with all KPIs and ablations.
* *Governance*: PRs changing configs without change‑log **fail CI**; ablation deltas are attached (HTML).

**Definition of Done (S4):** Nightly artifacts for both regions exist for 7 consecutive days; governance checks prevent unlogged parameter drift; dashboards are linked from report landing page.

---

## 6) Cross‑sprint specs (common to S1–S4)

**Data schemas (Parquet/Arrow)**

* `data/model1/{region}/states.parquet` — schema:
  `date: date32, industry_id: string, p_trend_bull: float32, p_trend_bear: float32, p_trend_sideways: float32, p_vol_calm: float32, p_vol_turbulent: float32, p_vol_crisis: float32, p_sys_diversifying: float32, p_sys_systemic: float32, p_flip_10d: float32, tilt_bp: float32, cap_long: float32, cap_short: float32`.
* `data/costs/{region}/lambda_params.parquet` — schema:
  `asof_date: date32, bucket_id: int8, lambda1_bp: float32, lambda2: float32, adv_edge_low: float32, adv_edge_high: float32`.

**Naming conventions & IDs**

* `industry_id` is **region‑scoped** (`US: <yfinance_sector>.<static_id>`, `CN: swL1.<code>`); free‑text labels not used in joins.
* **All dates are exchange local**; logs include both local date and UTC timestamp.

**Metrics & formulas**

* **Sharpe** and **DSR** computed exactly as in plan; DSR is **the** model selection criterion for Model 2/RL where applicable. 
* **Risk parity & constraints**: QP objective/constraints and L1/L2 costs are exactly those in the plan’s formulation; L1 handled via auxiliary variables (QP). **No solver change without RFC.** 

---

## 7) Command quick‑sheet (operators)

```bash
# Baseline (S0) — unchanged
make e2e_us
make e2e_cn

# Sprint 1 — run Model 1, then optimizer + backtest
uv run --locked python -m src.model1.run --config configs/model1_us.yaml --out data/model1/US/states.parquet
make opt_US && make backtest_us && make report_US

# Sprint 2 — fit costs, run scheduled execution
uv run --locked python -m src.optimizer.costs_fit --region US
make opt_US && make backtest_us && make report_US

# Sprint 3 — train RL, then inference run
uv run --locked python -m src.rl.train --config configs/rl.yaml --region US
uv run --locked python -m src.rl.eval --config configs/rl.yaml --region US
make opt_US && make backtest_us && make report_US

# Sprint 4 — nightly summary (CI or local)
uv run --locked python -m src.backtest.run_nightly --since 3y --regions US CN
```

---

### Why these specs are consistent

* They **extend S0** (uv projects, Qlib engine, cvxpy QP, data sources, configs, CI) with **zero API churn** and explicit acceptance gates. 
* They follow the **system architecture in the plan**: clear split of Model 1 (industry regimes), Model 2 (stock alpha), convex optimizer, and an **RL controller that never bypasses feasibility**. 

---

> **Appendix—traceability to your sources**
> • **Architecture & methods** (Model 1 states, Model 2 labels/features, optimizer objective/constraints, RL controller role, CPCV/embargo, DSR): from **plan.md**.  
> • **Tech stack, repo layout, configs, Makefile, CI, S0 scope/acceptance and future‑sprints outline**: from **sprints0specs.md**.  
