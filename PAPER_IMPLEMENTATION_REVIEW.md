# Paper ↔ Implementation Correctness Review (Adversarial Audit)

- **Date:** 2026-07-20
- **Auditor:** independent adversarial correctness review (read-only)
- **Repository state:** `a333d1f` (Task 10: address audit findings), branch `main`, clean tree at review start
- **Scope:** Verify the implementation faithfully follows the paper across four layers plus anti-drift:
  - (a) ROADMAP_STAMP_NEW.md vs paper/
  - (b) execution_plans/ vs roadmap
  - (c) code vs execution plans
  - (d) anti-drift: shipped code vs paper directly (end-to-end)
  - plus the method/logical-correctness checklist (H_lag, background projection P_Q^⊥,
    identifiability diagnostics, IASA estimator, merge/grouping, footprints, residual
    adequacy, experiments E1–E10 + observed mode, numerical/statistical validity)
- **Method:** static analysis of paper (ground truth), roadmap, execution plans, and code;
  math re-derived from the paper and compared to what code computes; adversarial scenario
  construction. Any executable checks run read-only on a SLURM GPU node.
- **Report discipline:** findings appended incrementally as confirmed; coverage log kept
  honest about what was and was not examined.

**Severity scale:** Critical (paper-facing result wrong) / High (method deviates from paper in a way that could change conclusions) / Medium (real deviation, limited blast radius) / Low (minor drift or fragility) / Info (observation, no action needed).

---

## 1. Summary & overall verdict

**Verdict: the method library is a faithful — in places exemplary — implementation of the paper's mathematics; the drift is concentrated in the experiment/observed layer, where four findings (F1, F2, F3+F15, F5) must be resolved before any result is presented as the paper's.**

**What is right.** Every core quantity was re-derived from the paper and traced into code, and `model/iasa/*` checks out: the open-boundary lagged puff operator (kernel normalization, release-time basis weights, exit-⇒-removal with no renormalization, mass accounting as diagnostics only, lag changing fingerprints but never row count), the thin-SVD background projector (P_Q^⊥ identity, idempotence, rank cap, stress labeling), the identifiability diagnostics (padded σ_J, τ_num, κ=∞ on deficiency, eligible-only coherence/ray-distance with JSON-null N/A, absorption), the projected-FISTA NNLS estimator (correct Lipschitz constant, KKT residual, fixed-zero mask with index maps, θ=ΦCᵀ reconstruction), the conservative merge system (cross-source eligible edges, deterministic components, retained triggers, no refit, global-unresolved rule), per-sensor footprints (exact decomposition, inherited identifiability), the refitted parametric-bootstrap adequacy check (correct coordinates, refit-per-replicate, add-one p, uncalibrated ⇒ no verdict), transport/inventory ensemble type separation, and the optional refinement with the paper's acceptance gates. The S1–S7 gate harness genuinely tests these claims (including the wind-diversity inequality, the span-absorbed negative control, and a statistically calibrated S7 study). No mathematical or statistical correctness bug was found in this layer; the only method-level nit is the σ_J-zeroing convention (F11).

**What is wrong (ranked).**
- **F2 (Critical):** the paper's PM2.5 observation-mask machinery (M_O; "PM2.5 is never imputed") is not implemented anywhere; the observed mode mean-fills missing PM2.5, and the background builder structurally cannot accept masked rows.
- **F3 (Critical as paper-facing):** the observed New Delhi mode is a pilot stub — no kriged transported-baseline subtraction, constant activity basis for all 7 columns, no fixed-zero mask, 3-day window — and does not implement the observed study the paper defines.
- **F1 + F15 (High):** the paper says FieldFormer-imputed gridded wind everywhere; the shipped default is a kernel interpolator, and the live "real wind" runs actually use an observed-station-mean city sequence with zero-fill (no imputed product exists on disk). Either the paper text or the pipeline must change; the Task 9D non-adoption logic itself was sound.
- **F5/F17 (High):** committed experiment configs are pilot-scale (20×20/T=24) and even the upgraded live configs (40×40/T=72) renormalize inventories to unit max, silently overriding the paper's declared own-p99 normalization.
- **F4 (High):** observed-mode percentage shares are coefficient-magnitude ratios across incomparable proxy units, exactly what the paper's fraction-of-fitted-sensor-signal rule forbids.
- **F6–F10, F16 (Medium):** E4's "historical" wind ensemble is synthetic mislabeled as historical and selected layouts are missing; E8 validates the adequacy engine on an abstract design rather than the platform; E5's parametric axis is direction-only; E2/E9/E10 use synthetic blobs instead of inventory maps; E10 never tests footprint localization; traffic is modeled as 4 sources instead of the paper's 1-source/slot-bases design.
- **F11–F14 (Low/Info):** σ_J not zeroed on numerical rank deficiency in reports; adequacy bootstrap ignores a nonzero prior; hardcoded provenance literals; inventory tag on plain fits.

**Recommendation.** Before Task 11 turns `summaries/` into paper tables: implement M_O selection end-to-end; rebuild the observed mode per `7.evaluation.tex` (gridded imputed wind, per-group bases + F₀ mask, kriged baseline, masked rows, sensor-signal shares); reconcile the FieldFormer text with the kernel reality (recommend: revise the paper and report the 9D validation); pin paper-facing configs at the paper platform with the declared normalization; and fix the E4 "historical" label. The library needs no rework.

---

---

## 2. Layer (a): Roadmap vs Paper

**Checked (full read of ROADMAP_STAMP_NEW.md against paper/3–9):**
- Projected model `H_tilde = P_Q_perp H_lag`, source-major/basis-minor columns, J=KB — matches 3.setup.tex eqs. (stacked_response, projected_model).
- Wind conversion `Ux = -WS sin(WD π/180), Vy = -WS cos(WD π/180)` — matches eq. wind_direction_conversion.
- Puff kernel `exp(-½‖x-z‖²_{Σ⁻¹})/(2π|Σ|^{1/2})`, covariance `R diag(σ∥² age, σ⊥² age) Rᵀ` with `effective_age = max(t-τ, t_min)` — matches eqs. gaussian_kernel, dispersion_covariance (roadmap resolves the paper's ambiguous `σ∥²(t-τ)` as variance ∝ age; a legitimate instantiation of the paper's "simple covariance model").
- Open-boundary semantics (exit → removal, no reflect/wrap/clamp/renormalize; boundary/truncation loss diagnostic-only) — matches 4.method.tex §plume_response.
- Lag rule: smallest L with `η_L = ‖H(L+Δ)-H(L)‖_F / max(‖H(L+Δ)‖_F, ε) ≤ τ_L=1e-3`, fitted ĉ never used — matches eq. lag_convergence.
- Diagnostics: τ_num = max(N,J)·ε_mach·σ₁; padded σ_J; κ=∞ when r_num<J; τ_σ noise-scale effective rank; v_j, weak set, eligible-only coherence/ray distance with N/A for weak pairs; absorption = ‖H_removed_j‖/‖H_lag_j‖ = ‖P_Q h_j‖/‖h_j‖ — all match 5.theory.tex.
- Merge: eligible cross-source edges, deterministic connected components, `is_conservative`, A–B–C chain retained edges, global unresolved warning, grouped sums with NO refit — matches 5.theory §source_merging. (Note: the older commented-out algorithm in 6.algorithm.tex lines 141–272 described a grouped REFIT; the active §Procedure text and 5.theory forbid silent refit. Roadmap follows the active text — correct.)
- Fit: projected FISTA, Lipschitz step from σ₁, monotone restart, KKT stopping — matches 6.algorithm §Procedure + 9.appendix §pytorch_solver.
- Adequacy: refitted parametric bootstrap, B=1000, α=0.05, add-one p-value, uncalibrated ⇒ no pass/fail — matches 6.algorithm §model_adequacy.
- Experiments 1–10 definitions and observed-mode restrictions — match 8.evaluation.tex one-for-one, including the E8 in-span negative control and the "never assigned synthetic recovery metrics" rule for observed mode.
- Platform: 40×40 crop (21:61, 16:56), own-p99 normalization, 32 sensors with Pusa averaging, 21,960 hourly stamps, traffic single-source diurnal slot bases, rank-4 primary Q — match 7.evaluation.tex.

**Findings:** none so far — the roadmap is a faithful, in places more-operational, translation of the paper. Gate S7 and Tasks 9C/9D framing are roadmap additions consistent with the paper's direction (recorded in §7).

---

## 3. Layer (b): Execution plans vs Roadmap

**Examined:** `task10_2026-07-20.md` (full read); `task5_v3`, `task6_v2`, `task7`, `task8`, `task9`, `task9b`, `task9c`, `gateS7` (objectives/contracts sections). Earlier plan revisions (`task5_v1/v2`, `paper_*`, `task3*`, `task4`, `task6`, `task6a`, `task9a`, `task9d`) were not read line-by-line; the shipped code was audited directly against roadmap+paper instead, so a plan-level error there would have surfaced at layers (c)/(d).

**Findings:** none material. The plans quote the paper's equations and the roadmap's contracts accurately (Task 7 plan restates every 5.theory quantity including padded σ_J and N/A conventions; Task 8 plan pins projected FISTA as the sole solver with the calibrated/uncalibrated adequacy contract; Task 9/9B/9C plans restate the merge, footprint, and refinement-acceptance rules verbatim). Two observations:
- The Task 10 plan itself *introduced* the reduced default platform ("Grid: reduced default Nx=Ny=20, T=24 (config-overridable) so a full sweep runs in minutes") — i.e. the platform-scale drift (F5) originates at the plan layer, is documented, and is config-overridable (and the live evaluation configs do override it to 40×40/T=72). The roadmap/paper never authorized a *default* below the paper platform for paper-facing tables; this remains tracked under F5.
- The Task 10 plan promises for the observed mode "real PM2.5 + mask + gridded imputed wind + declared proxy bases" — the code (layer c) delivers none of the last three (F2, F3, F15, F16).

---

## 4. Layer (c): Code vs Execution plans

Code matches its plans essentially everywhere in `model/iasa/*` (APIs, shapes, contracts, gate behaviors as specified — spot-verified against the Task 5/7/8/9/9B/9C plan contracts). The deviations are concentrated in Task 10 (details in the numbered findings):
- Plan E2: "shift a copy of one **inventory map**" → code shifts a synthetic Gaussian blob (F9).
- Plan E5: "parametric wind **speed/direction/dispersion** perturbations" → code implements direction only (F8).
- Plan E8: "declared Gaussian cov, **1000 reps**" → committed config 200 (live eval config 1000); platform-based omission → abstract random orthonormal design (F7).
- Plan E10: "footprints **localize responsible upwind cells**" → no localization check in E10 (present only in the footprints gate) (F10).
- Plan observed mode: "real PM2.5 + **mask** + **gridded imputed wind** + **declared proxy bases**" → mean-filled PM2.5, city-mean fallback or constant wind, single constant basis, no mask (F2, F3, F15, F16).
- Plan provenance: "primary Q / lag rule / fixed-zero mask / inventory version recoverable" → runio does stamp these (verified: `config.resolved.json` carries platform config, inventory version, lag rule, git SHA, device/dtype); the *declared-before-fit* claim, however, is a literal (F13).

---

## 5. Anti-drift: Code vs Paper directly

### F1 — Paper says FieldFormer wind imputation; shipped default is a kernel interpolator
- **Severity:** High (paper-facing description vs actual pipeline)
- **Layers:** (d), also (a)-adjacent (roadmap documents the divergence; paper does not)
- **Paper:** `4.method.tex` §Masked Wind Preparation ("The observed station vectors and masks are supplied to a FieldFormer wind imputer… we query the trained model on every response-grid cell"), `7.evaluation.tex` §Wind Preparation ("The New Delhi runs use the FieldFormer-imputed grid wind field").
- **Code:** `model/iasa/wind.py:401` — `build_gridded_wind_field` defaults to `KernelCoordinateQueryImputer` (normalized Gaussian-kernel interpolation); `gridded_new_delhi_wind_field` (wind.py:566) uses that default. Task 9D trained a 2-vector FieldFormer checkpoint but did **not** adopt it (held-out vector RMSE 1.12 beats kernel 1.55 but loses to city-mean 1.06; ROADMAP lines 1607–1638), so the paper-facing New Delhi wind is kernel-interpolated.
- **Drift:** any observed-New-Delhi result produced now is generated with a different wind imputer than the paper describes. The imputer is recorded in provenance (`metadata["imputer"]`), so it is not silent in artifacts — but it contradicts the paper text.
- **Who should change:** the paper. Either (i) rewrite §Masked Wind Preparation / §Wind Preparation to describe the coordinate-query interpolation actually used (with FieldFormer as an evaluated-but-not-adopted alternative + the validation numbers), or (ii) hold observed-mode results until a learned imputer beats both baselines. The decision logic in Task 9D (`recommended_default` requiring wins over kernel AND city-mean) is itself sound.

### F2 — The paper's PM2.5 observation-mask row selection (M_O) is not implemented anywhere; the observed mode IMPUTES missing PM2.5
- **Severity:** Critical
- **Layers:** (d) anti-drift, (a) roadmap-vs-paper is faithful (roadmap requires it: "Do not impute pm25; apply its observed-row selection identically to Y, H_lag, Q, and row metadata"), so this is a code gap.
- **Paper:** `3.setup.tex` eq. observation_selection ("The observation mask is applied identically to every row-aligned object… an alignment mismatch is an error… Wind may be imputed, but missing PM2.5 values are not"); `7.evaluation.tex` §Government Observations ("PM2.5 is never imputed. Its flattened valid-value mask defines M_O"); `9.appendix.tex` §Implementation Invariants ("The PM2.5 observation mask then selects the same ordered rows from Y, H_Φ^lag, Q, and metadata").
- **Code:** repo-wide grep finds no masked-row selection: `model/iasa/response.py:564` hardcodes `"observation_mask": None`; `model/iasa/background.py:_validated_rows` (background.py:39–67) **requires complete time-major sensor blocks**, so a masked (incomplete) row set cannot even build Q; `experiments/iasa_pol/experiments.py:_observed_Y` (experiments.py:987–1016) fills missing PM2.5 rows with the **observed column mean** ("Missing observations are filled with the observed column mean so the fit sees a complete vector").
- **Failure scenario:** New Delhi has ~9.5% missing PM2.5. Mean-filling injects a flat pseudo-signal into ~10% of rows. The rank-4 background absorbs some of it, but sensor-hours whose true concentration deviates from the global mean systematically bias ĉ, the residual diagnostics, and the adequacy statistic (mean-filled rows have near-zero noise, deflating T_res's null consistency). The paper's guarantee "an alignment mismatch is an error rather than an invitation to reorder silently" has no enforcement point because selection never happens.
- **Suggested fix:** implement M_O selection as a first-class operation (select ordered rows of Y/H/Q/row-metadata after response construction and Q construction on the full grid, as the paper specifies), relax `_validated_rows` to accept masked row sets (or build Q pre-mask and select rows), and delete the mean-fill path. Until then, no observed-mode number should be treated as paper-facing.

### F3 — Observed New Delhi mode is a pilot stub that contradicts the paper's observed-mode definition on wind, record length, bases, domain, and baseline
- **Severity:** Critical (as a paper-facing artifact; acceptable only as an explicitly-labeled smoke)
- **Layers:** (d); also (c) — Task 10's plan promised "gridded imputed wind" for observed mode.
- **Paper:** `7.evaluation.tex` §Study Modes ("The observed-data mode combines real PM2.5, its observation mask, the imputed New Delhi wind field, the named normalized inventories, and the declared proxy temporal bases"); §Forward Models (observed runs subtract a declared zero-source transported kriged initial-condition baseline); platform = 40×40, 32 sensors, 21,960 hours.
- **Code:** `experiments/iasa_pol/configs/observed.json` sets `"wind_kind": "constant"`, `T=24`, `grid_shape=[20,20]`; `observed_new_delhi` (experiments.py:919–984) uses `default_basis("constant", T)` (one constant basis for all sources — no traffic diurnal slots, no kiln blocks, no fixed-zero mask), never subtracts any initial-condition baseline (`ResponseConfig` only supports `baseline_policy="zero_source"`, response.py:226, and the baseline vector is all zeros), and runs on the 20×20 downsampled platform.
- **Failure scenario:** a "normalized proxy contribution" produced by regressing one day of mean-filled PM2.5 onto constant-wind puff fingerprints with constant activity bases bears no relation to the study the paper describes; if these numbers reach the paper's observed-results tables they are wrong in transport, time coverage, activity model, and baseline simultaneously.
- **Suggested fix:** observed mode must consume `gridded_new_delhi_wind_field` (through `GriddedWindSampler`), the full (or a declared long) record, the paper's per-source temporal bases + fixed-zero mask, the 40×40 platform, kriged transported-baseline subtraction, and masked-row selection (F2). The good news: every ingredient except M_O already exists in the repo (kriging in `sim/polsim.py`, gridded sampler in response.py, per-source bases in activity.py).
- **Amendment (live runs):** the un-committed evaluation configs (`evaluation/iasa_pol/configs/observed.json`) upgrade to 40×40, T=72, `wind_kind: "real"`. This removes the constant-wind and 20×20 objections for the live runs, but T=72 (3 days of 21,960 hours), constant-basis activity, no fixed-zero mask, no baseline subtraction, mean-filled PM2.5 (F2), and city-mean fallback wind (F15) all still hold.

### F4 — Observed-mode "normalized proxy contributions" are coefficient-magnitude shares, not the paper's fraction-of-fitted-sensor-signal
- **Severity:** High
- **Layers:** (d)
- **Paper:** `8.evaluation.tex` §Metrics ("A reported percentage, if used, is explicitly a fraction of the fitted inventory-attributed sensor signal and not a physical-emission share"); `9.appendix.tex` ("A percentage is emitted only with an explicit denominator identifying it as a fraction of fitted inventory-attributed sensor signal").
- **Code:** `experiments/iasa_pol/experiments.py:960–964` — `normalized_contributions = |c_k| / Σ|c|` over raw coefficients.
- **Failure scenario:** the four inventories are independently normalized (own p99 / unit max), so coefficients are in incomparable per-source proxy units; a source with a small map scale gets a large coefficient for the same physical signal. Coefficient-share percentages are therefore meaningless across sources — exactly the artifact the paper's rule exists to prevent. The correct denominator is in sensor-signal space (e.g. per-source fitted contribution sums `Σ_t Σ_b H_{(s,t),(k,b)} ĉ_kb`, already computed by `decompose_per_sensor`).
- **Suggested fix:** compute shares from fitted sensor-space contributions per source (and per report group), with the explicit denominator.

### F5 — Controlled-experiment platform drifts from the paper platform: 20×20 grid, T=24, unit-max renormalization, deduped sensors
- **Severity:** High (Critical if these runs feed the paper tables unchanged)
- **Layers:** (d); (c) partially — the Task 10 plan itself adopted the small platform.
- **Paper:** `7.evaluation.tex` — 40×40 domain (crop rows 21:61, cols 16:56), each map "divided by its own cropped 99th percentile", 32 regulatory sensors, hourly record.
- **Code:** `experiments/iasa_pol/nd_platform.py:46–52` — `PlatformConfig(grid_shape=(20,20), T=24)`; `_block_downsample` (nd_platform.py:109–125) block-means 40×40→20×20 and **renormalizes each map to unit max** (a second normalization on top of / instead of the paper's own-p99); `_regulatory_grid_cells` (nd_platform.py:143–173) dedupes the 32 stations into distinct 20×20 cells (fewer effective sensors). All ten committed configs pin this platform.
- **Failure scenario:** every reported σ_J, coherence, rank, and recovery number is for a different (coarser, easier/harder in unknowable ways) inverse problem than the paper's platform. Comparisons within an experiment stay internally valid (one-factor discipline is respected), but none of the absolute numbers can be transplanted into the paper's result sections, and claims like "32 regulatory sensors" would be false for these artifacts.
- **Suggested fix:** either add paper-scale configs (grid 40×40, full sensor set, longer T) as the paper-facing runs — treating the 20×20/T=24 configs as smoke — or change the paper's platform section. Also drop the unit-max renormalization when running at native 40×40 (at native resolution `_block_downsample` still rescales p99-normalized maps to unit max, silently changing the declared normalization — see F17).
- **Amendment (live runs):** `evaluation/iasa_pol/configs/*` (the configs actually generating results, resolved provenance confirms 40×40, T=72, lag 12, cuda, git `a333d1f`) substantially reduce this drift: native grid, no downsampling, regulatory stations map to (mostly) distinct cells. Remaining gaps at paper scale: T=72 vs the paper's hourly record, the unit-max renormalization (F17), and synthetic blob sources in the controlled experiments (F9). The **committed** configs and summaries remain pilot-scale; whichever set feeds Task 11 must be the paper-scale one.

### F6 — Experiment 4: "historical" wind ensemble is synthetic and mislabeled; real-wind axis disabled; selected layouts missing
- **Severity:** High (provenance mislabeling), Medium (missing axes)
- **Layers:** (d), (c)
- **Paper:** `8.evaluation.tex` E4 — wind kinds include "real imputed New Delhi wind"; "Separate historical and simulated window ensembles"; sensor variants include "layouts selected to increase σ_J or reduce maximum eligible coherence".
- **Code:** `experiments/iasa_pol/experiments.py:457–480` `_wind_window_ensemble` — the `"historical"` family is `make_wind("multi", …)` (multi-direction **synthetic**) with shifted seeds, commented "'historical' windows: shifted diurnal phases as proxy real windows", then emitted under the key `historical`; `configs/exp04.json` sets `include_real_wind: false`, so no real-wind column exists in the committed E4 run; only regulatory/random/downwind layouts exist — no σ_J-optimizing or coherence-reducing selected layouts.
- **Failure scenario:** a results table showing a "historical" wind-window distribution that was never computed from the historical record is a provenance error of exactly the kind the paper's discipline (ensemble tagging, declared provenance) is designed to exclude.
- **Suggested fix:** implement historical windows as contiguous slices of the imputed New Delhi wind record (the loader exists), enable the real-wind kind, and either implement the selected-layout variants or strike them from the paper's E4 sentence.
- **Amendment (live runs):** `evaluation/iasa_pol/configs/exp04.json` sets `include_real_wind: true` (20 ensemble members) — but "real" resolves to the observed-station-mean fallback (F15), and the `"historical"` ensemble family remains synthetic `multi_direction` labeled historical. The mislabeling finding stands.

### F7 — Experiment 8 runs on an abstract random orthonormal design, not the New Delhi platform; replicate count below paper spec
- **Severity:** Medium-High
- **Layers:** (d) (the roadmap correctly restates the paper; the code substitutes a synthetic design)
- **Paper:** `8.evaluation.tex` E8 — "Controlled trials omit either a residual-visible source outside span[H_lag,Q] or an aligned source whose signature lies in that span" on the New Delhi platform; adequacy protocol with 1,000 refitted replicates (`9.appendix.tex` §Adequacy contracts: "Paper runs use α=0.05 and 1000 refitted replicates").
- **Code:** `experiments/iasa_pol/experiments.py:720–824` — H is a seeded random N(0,1) matrix orthonormalized by QR (N=48, 4 columns), `n_replicates=200` (config), empty background. No transport operator, no inventory, no Q span component in the "aligned" case.
- **What is RIGHT about it (verified adversarially):** the in-span negative control is *genuine*: the fit uses columns 0–2 with column 3 fixed-zero-masked; the truth adds a real omitted component `v_inspan = H_fit·w, w>0` scaled to equal energy with the out-of-span case; NNLS absorbs it exactly, so non-rejection with a real omission is demonstrated. (Any in-span omission is mathematically equivalent to a coefficient shift — that equivalence *is* the paper's absorbability point, so this is not a sham control.) Equal-energy matching makes the power comparison fair. The out-of-span component is exactly orthogonal by QR construction, giving clean power.
- **Drift:** the paper's E8 is about missing sources under the platform's transport geometry (where "outside the span" is a physical statement about wind/inventory geometry); the implementation demonstrates only the statistical engine (redundant with Gate S7). Also `span([H_lag, Q])` is exercised with empty Q, so background absorption of an omitted source is untested. And 200 < 1000 replicates.
- **Suggested fix:** add a platform-based E8 variant: omit one real inventory (e.g. generate Y with 3 sources + an omitted 4th placed upwind of sensors = residual-visible; and an omitted copy/shift of a fitted inventory or a Q-aligned smooth signal = absorbed), with 1,000 replicates for paper-facing runs.
- **Amendment (live runs):** `evaluation/iasa_pol/configs/exp08.json` uses `n_replicates: 1000, n_trials: 100` — the replicate-count part is resolved for the live runs; the abstract-design and empty-Q parts stand.

### F8 — Experiment 5: parametric transport axis covers wind direction only; PDE generator uses zero initial condition
- **Severity:** Medium
- **Layers:** (d)
- **Paper:** E5 "Parametric perturbations vary wind speed, wind direction, and dispersion within the puff family"; `7.evaluation.tex` says the E5 PDE generator uses kriged ICs "subtracted through their matched zero-source trajectory before source fitting".
- **Code:** `experiments/iasa_pol/experiments.py:506` — only `wind_direction_perturbations_deg`; no speed or dispersion perturbation rows. `edge_hold_pde.simulate_edge_hold_observations` starts from `U = 0` (edge_hold_pde.py:74) — no kriged IC (hence nothing to subtract; internally consistent but not what the paper describes).
- **Verified NOT a problem (adversarial check on scaling):** the structural Y is rescaled by a single global least-squares α to the puff-consistent magnitude (experiments.py:560–563). This cannot hide the structural mismatch: a global scale is absorbed by the free nonnegative coefficients anyway; only the *shape* discrepancy drives residuals and adequacy rejection. The generator is labeled `edge_hold_pde` (edge_hold_pde.py:27, in returned metadata) and is never substituted for the puff response — verified in `_experiment_5_structural`, which fits with the open-boundary puff H.
- **Suggested fix:** add speed and dispersion perturbation rows; either adopt zero-IC in the paper's E5 description or implement the kriged-IC + matched-baseline-subtraction path.

### F9 — Controlled experiments use synthetic Gaussian-blob sources; paper says controlled mode preserves the New Delhi inventories (E2 explicitly shifts "a copy of one inventory map")
- **Severity:** Medium
- **Layers:** (d)
- **Paper:** `7.evaluation.tex` §Study Modes ("The controlled mode preserves New Delhi inventories and sensor geometry while assigning known nonnegative source–basis coefficients"); E2: "shifting a copy of one inventory map by a controlled spatial offset".
- **Code:** E1/E2/E3/E4/E5/E6/E7/E9/E10 all use `compact_source` Gaussian blobs with names `src_a`/`src_b`… (e.g. experiments.py:293–297, 338–341); no experiment shifts an actual inventory map; E4/E10 replace the regulatory observer with random/downwind layouts even in rows not exercising the layout axis (E10 uses only the downwind layout).
- **Failure scenario:** conclusions about e.g. coherence-vs-offset or footprint localization on idealized blobs may not transfer to the extended, overlapping real proxy maps (population density covers most of the domain; kilns are peripheral); a reader of the paper expects the platform inventories.
- **Suggested fix:** where the paper names inventories (E2 especially), use the real maps; keep blob variants as extra controlled rows if desired.

### F10 — Experiment 10 never tests footprint localization; observed mode reports no per-monitor footprints
- **Severity:** Medium
- **Layers:** (c), (d)
- **Paper:** E10 — "we check that the per-sensor footprints … localize the responsible upwind cells"; observed mode reports "per monitor, the fitted source-group contributions and their spatial cells of origin".
- **Code:** `experiment_10` (experiments.py:869–913) checks only nonnegativity and the contribution-sum identity; it computes footprint fields but never compares them to the known source origins (no localization metric). `observed_new_delhi` computes no footprints at all.
- **Suggested fix:** add a localization metric (e.g. distance from footprint mass centroid / argmax to the true upwind source center, or fraction of footprint mass within r cells of the origin) and per-monitor footprint output in observed mode (Task 11 may plan this, but E10's claim is then untested at Task 10). Note: the **footprints gate** (`run_iasa_sanity.py:run_footprints_gate`, lines 1479–1487) DOES check upwind localization (east-sensor footprint peak upwind, mass ordering) — so the property is exercised at gate level, just not in E10 where the paper puts it.

### F15 — No imputed wind product exists; "real" wind in the live evaluation runs is the observed-station-mean fallback with zero-fill
- **Severity:** High
- **Layers:** (d); provenance labeling itself is correct per roadmap
- **Paper:** `7.evaluation.tex` §Wind Preparation — "The New Delhi runs use the FieldFormer-imputed grid wind field"; `4.method.tex` — wind is imputed by a trained coordinate-query model.
- **Code/state:** `data/new_delhi_wind_imputed.npz` **does not exist** in the repo. `make_wind("real")` (nd_platform.py:209–217) calls `real_new_delhi_wind_sequence(..., allow_observed_fallback=True)`, which therefore returns `_observed_new_delhi_wind_sequence` (wind.py:80–107): the per-hour **mean of observed station vectors**, with **zero vector** for any hour with no valid station, provider `real_observed_new_delhi`, `imputed_product_missing: True`. The live result-generation configs (`evaluation/iasa_pol/configs/*.json`: 40×40, T=72, `include_real_wind: true`, observed `wind_kind: "real"`) route through exactly this fallback.
- **Failure scenario:** (i) it is a city-level scalar sequence, not the paper's gridded field (the gridded kernel path `gridded_new_delhi_wind_field` exists but is not what `make_wind("real")` uses); (ii) hours where WD/WS are entirely missing become **calm** (v=0), which stalls puffs at their release cells and systematically inflates same-cell sensor responses — "imputation by zero" is a transport artifact, not an imputation method; (iii) E4 rows label the wind axis simply `"real"`, so a summary CSV reader cannot distinguish imputed from fallback without opening wind metadata.
- **Suggested fix:** for paper-facing runs either generate the imputed product (`scripts/impute_new_delhi_wind.py`) or drive the response through `GriddedWindSampler(gridded_new_delhi_wind_field(...))`; propagate the wind provider label into experiment row records.

### F16 — Traffic is four separate spatial sources; the paper models it as one road-network source with slot temporal bases and a fixed-zero mask
- **Severity:** Medium (High for the observed mode, where it changes J and the diagnosed problem)
- **Layers:** (d)
- **Paper:** `7.evaluation.tex` §Inventories — "The study uses four source groups… Traffic is modeled as a single road-network source with a diurnal temporal structure rather than as separate inventories per time slot. The nearest-slot maps at 00/06/12/18 define traffic-specific temporal-basis components, so one road map carries a time-varying activity… the fixed-zero mask F₀ frees each group's coefficients only on its own admissible components."
- **Code:** `sim/pol_sources.py:10–18` loads 7 sources (`traffic_00/06/12/18` as separate spatial maps); `observed_new_delhi` fits all 7 with a single shared `constant` basis and an **empty** fixed-zero mask (experiments.py:928–956). The paper's K=4-group, per-group-admissible-basis design (with F₀ masking) is never instantiated. (`build_default_activity_profile` implements the slot-indicator activities for simulation, but the fitting side never uses per-group bases + mask.)
- **Failure scenario:** the observed-mode J, coherence structure, and report groups describe a different declared model than the paper's (e.g. four traffic columns that are near-coherent by construction under similar transport, inviting merges the paper's parameterization avoids by design). Also, paper's "congestion spatial pattern is time-invariant" assumption is not what the code encodes.
- **Suggested fix:** build the observed-mode design as the paper declares: 4 source groups; traffic uses one road map (or the 4 slot maps as the paper's *temporal* components with a shared spatial map — follow the paper text) with slot-indicator bases; other groups get their declared bases; encode admissibility via `fixed_zero_indices`.

### F17 — Even at native 40×40 resolution the platform re-normalizes each inventory map to unit max, overriding the declared own-p99 normalization
- **Severity:** Low-Medium (amendment to F5; applies to the live 40×40 runs too)
- **Layers:** (d)
- **Paper:** maps "divided by its own cropped 99th percentile" (`7.evaluation.tex`).
- **Code:** `nd_platform.py:_block_downsample` — even when `nx == h` (no downsampling) it divides each map by its max (`peaks` division, nd_platform.py:123–125), on top of the p99 normalization applied in `pol_sources.py`. Coefficients therefore live in "p99-then-max" units, not the declared p99 units; recorded normalization metadata (`per_source_cropped_p99`) no longer describes the maps actually used.
- **Suggested fix:** skip the peak renormalization at native resolution (or record it explicitly as a second normalization stage).

---

## 6. Method / logical-correctness findings (adversarial checklist)

### F11 — σ_J is not zeroed for a numerically rank-deficient response (J ≤ N case)
- **Severity:** Low-Medium
- **Layers:** (d)
- **Paper:** `5.theory.tex` eq. identifiability_score — "the singular spectrum is padded with zeros whenever J>N **or** H̃_Φ is rank deficient. A rank-deficient response therefore scores zero and is never reported as stable."
- **Code:** `model/iasa/diagnostics.py:196–204` pads zeros only when `J > N`; when J ≤ N and r_num < J, `sigma_J` is reported as the raw smallest singular value (~1e-16·σ₁ for an exact duplicate column), not 0. Mitigations verified: condition status is `infinite`, a `rank_deficient` warning is emitted, and `refine.py:_sigma_J_eff` (refine.py:198–203) applies the zeroing rule where it matters most (refinement acceptance, eq. refinement_smin_check). Exposure: experiment tables and JSON summaries report the raw tiny value (`_diag_summary`), and `perturbation_sensitivity.inverse_sigma_J` becomes a huge finite number instead of `infinite`.
- **Suggested fix:** in `diagnose_identifiability`, set `sigma_J = 0.0` when `numerical_rank < J` (keep `sigma_min_positive` as the raw value, which the paper explicitly allows as an additional report).

### F12 — Adequacy bootstrap refits ignore a nonzero prior mean
- **Severity:** Low
- **Layers:** (c)
- **Paper/roadmap:** the bootstrap must "refit each replicate with the same estimator".
- **Code:** `model/iasa/fit.py:892` — replicate refits use `prior = zeros` and the stored `lambda_reg`, even if the observed-data fit used `prior_mean=c0 ≠ 0`. Inactive at the default λ=0 (all current callers), but a latent inconsistency if the prior regularizer is ever used with adequacy.
- **Suggested fix:** thread `cfg.prior_mean` (reduced) into `residual_adequacy_check`.

### F13 — Provenance claims emitted as hardcoded literals
- **Severity:** Low (verified true today, but unenforced)
- **Layers:** (c)
- **Code:** `experiments.py:402` `"declared_before_fit": True` and `experiments.py:712` `"coefficients_used_for_selection": False` are constants, not checks. I verified both are *currently* true by reading the code (backgrounds are constructed from configs before Y is seen except the labeled stress basis, which necessarily reads H_lag; lag selection at experiments.py:678–690 uses only Frobenius deltas of H). But nothing would flip these flags if a future edit violated them.
- **Suggested fix:** derive the flags (e.g. assert the background config hash was registered before `forward()` generated Y; assert the lag-selection code path has no access to fit results).

### F14 — Plain single fits default to `ensemble_kind="inventory"`
- **Severity:** Info
- **Code:** `forward(..., ensemble_kind="inventory")` default (experiments.py:142) tags every ordinary controlled fit as an inventory-ensemble member. The aggregation guards still work (kinds must be homogeneous), but the tag was designed to mark *ensembles*; a neutral default (or requiring the caller to tag only real ensemble members) would keep provenance sharp. Not a correctness bug.

### Verified-correct items (adversarial checklist, no finding)
- **H_lag construction** (`response.py`): fingerprint `h_{t,kb} = Σ_{ℓ} φ_b(t-ℓ)·OG·s_k` implemented release-time-major with weight φ_b(τ) at the release time — matches eq. time_response. Lag window changes fingerprints only; row count `M·T` independent of `lag_window_steps` when `trim_initial_lag=False` (the default; E7 verifies `row_count_fixed`). Kernel normalization `2π√(var∥·var⊥)` = `2π|Σ|^{1/2}` exact. Anisotropy: covariance axes aligned to trajectory-mean wind, deterministic +x fallback below 1e-8 norm (matches paper's near-calm tie-breaker). Same-time release uses `effective_age = max(age, t_min)` — matches. Open boundary: center exit ⇒ whole remaining release removed (break over later observation times), never reflected/wrapped/clamped/renormalized; retained/dropped mass are quadrature diagnostics only (5×5 midpoint rule per cell, clipped to [0,1] with clip-count recorded) and never rescale H. Mass conservation `retained + dropped = emitted` holds per kernel by construction.
- **Background projection** (`projection.py`): thin-SVD `P_Q^⊥X = X − U_r(U_rᵀX)`, tolerance `max(shape)·eps·σ₁`, empty-Q no-op, idempotence/orthogonality residuals recorded, dependent columns identified by prefix rank, rank cap 8 enforced for `basis_mode="normal"`, stress bases must be labeled. Row-metadata equality between response and background enforced (`projection.py:167`). Normal constructors use only timestamps/day labels/sensor ids/coords (`background.py`) — matches 4.method §background_construction.
- **Diagnostics** (`diagnostics.py`): τ_num = max(N,J)·ε·σ₁ via true SVD (not Gram eigenvalues — the code comments correctly note the sqrt(eps) floor hazard); κ finite only at full numerical rank; τ_σ separate noise threshold with `noise_threshold_provided` flag; visibility = column norms; weak set `v_j ≤ τ_v`; coherence/ray distance only for eligible pairs, NaN→JSON null (never a fake numeric distance); absorption `‖H_removed_j‖/‖H_lag_j‖` with N/A when the raw norm is at tolerance — all match 5.theory. Fixed-zero mask removes columns pre-diagnostics with both index maps; small fitted values never alter the mask (no post-fit column removal path exists).
- **IASA estimator** (`fit.py`): projected FISTA, L = 2(σ₁²+λ), exact clamp projection, per-column monotone restart with t reset, correct NNLS KKT residual (`g_j` on the active set, `min(g_j,0)` at the boundary), fixed-zero restore, device/dtype/KKT/iterations recorded. Fit/diagnostics mask mismatch rejected (`fit.py:597–600`).
- **θ reconstruction:** `theta = Φ Cᵀ` with source-major/basis-minor mapping via column_index — matches eq. reconstructed_activity.
- **Merge** (`merge.py`): eligible cross-source pairs only (same-source ambiguous pairs excluded from edges), max-coherence trigger pair retained per edge (ties broken deterministically; min-ray tracked), deterministic union-find components ordered by smallest member, `is_conservative=true`, no `finest` claim, A–B–C chain retains both edges, weak flags never create edges, `global_unresolved` exactly when rank-deficient with no eligible cross-source edge, grouped activity/sensor contributions are member sums with **no refit** (`summarize_report_groups`, and `grouped_sensor_contribution = Σ_members H̃ ĉ` on member columns).
- **Footprints** (`footprints.py`): per-sensor decomposition is exact linear algebra on the fitted matrices (projected and raw forms); footprint = one-hot single-cell re-run of the same Task 5 builder (no new operator), fitted footprint per cell = `s_k(i)·Σ_b ĉ_kb·pull(i,b)` which sums over cells to the raw per-sensor contribution; nonnegativity enforced by assertion; inheritance σ_J(H̃^{(s)}) ≤ σ_J(H̃) verified via row-submatrix diagnostics (`per_sensor_identifiability`).
- **Adequacy** (`fit.py:residual_adequacy_check`): Z from SVD of I−U_rU_rᵀ with trace-rank; T_res = r̄ᵀΣ̄⁻¹r̄ on the raw source residual (background directions annihilated by Zᵀ — algebraically equal to the paper's coordinates); bootstrap mean = H ĉ + best-fit background component (with the empty-background case correctly NOT absorbing the whole residual — the comment documents the trap they avoided); refitted replicates via batched FISTA; add-one p; uncalibrated ⇒ null fields, no verdict; noise model must be external, calibrated, and not residual-derived.
- **Ensemble separation:** `validate_ensemble_kind` at construction; `_require_single_kind` rejects mixed pooling; E6 asserts the rejection actually fires (`transport_inventory_pooling_rejected`).
- **Refinement** (`refine.py`): objective terms match eq. refinement_objective (wind anchor uses the position-independent per-node proxy with the grid factor absorbed in λ_w — documented); ε_w enforced by projection AND re-checked post hoc; Ψ_phys box clamp; acceptance uses the σ_J-zeroing rule and eligible-only max coherence; deterministic pattern search (no RNG); fixed-response fit remains the default report. The λ_sm>0 fit-gate subtlety (total could drop while data term rises) is explicitly handled (refine.py:505–513).
- **Lag selection** (E7): smallest L with Frobenius ratio ≤ τ_L=1e-3, denominators max(‖H(L+Δ)‖_F, ε), row count invariance asserted, coefficients never consulted.

---

## 7. Intentional additions / not-in-paper (verified consistent)

| Item | Judgment |
|---|---|
| **Gate S7 adequacy calibration study** (`run_calibration_gate`) | Consistent and statistically sound. Uses an orthonormal synthetic design to isolate the test; judges the null rate by p-value (< α) rather than the finite-B interpolated-quantile decision, with the liberal-decision caveat documented in-code; KS uniformity check; monotone power with Monte-Carlo slack; deterministic seeding with non-colliding multipliers. A genuine strengthening of the paper's §model_adequacy, not a contradiction. |
| **S1–S6 gate harness** (`scripts/run_iasa_sanity.py`) | Consistent; implements the paper's Appendix §Sanity Gates faithfully (all thresholds from the roadmap present; the S3 wind-diversity inequality is asserted, not weakened; S4 includes the span-absorbed negative control; S6 runs the fit on a response-builder-derived H̃). |
| **Kernel coordinate-query imputer as wind default; FieldFormer trained but not adopted** (Task 9A/9D) | Internally sound (adoption gated on beating kernel AND city-mean on held-out stations; report json confirms FF 1.12 vs kernel 1.55 vs city-mean 1.06) — but it contradicts the paper text, so it is escalated as **F1** rather than accepted silently. The adapter refuses non-2-vector models and documents the upstream lowest-index neighbor caveat. |
| **ImputeFormer city-level v1 wind product** (`data/pol_weather.py`) | Not in the paper (paper names FieldFormer only). Acceptable as the roadmap's v1 adapter; the product file is absent in the repo, so nothing currently consumes it (see F15). Folded into F1's resolution. |
| **Task 9C constrained refinement as optional non-default** | Consistent. Objective/constraints/acceptance match eqs. refinement_objective/constraints/smin_check/coherence_check; σ_J-zeroing rule applied; fixed-response fit verified unmutated in the refine gate; deterministic pattern search avoids RNG. The position-independent wind-correction basis is a declared simplification of "wind-imputer parameters φ" — within the paper's "small physically constrained corrections" intent. |
| **FieldFormer training pipeline** (`scripts/train_fieldformer_wind.py`, `--gate fieldformer_train`) | Consistent scaffolding; smoke-only in gates; untrained model explicitly barred from being a scientific wind product. |
| **Edge-hold PDE generator clamping U ≥ 0 per substep** (`edge_hold_pde.py:93`) | Acceptable: a stabilizer on the *generator* side of a labeled structural-mismatch experiment; does not touch the fitted operator. Zero IC deviation recorded in F8. |
| **`_diverse_wind` two-direction synthetic regime** in experiments | Consistent with the gate S1/S3 two-direction construction; a controlled input, labeled by provider. |
| **Reduced-platform experiments gate** (12×12/T=12 in `run_experiments_gate`) | Fine as a gate; not paper-facing. |

---

---

## 8. Open questions / not fully verified

1. **Purely static review.** No code was executed (the analysis did not require it, and gates/tests were reported green on GPU at `a333d1f`). All numerical-behavior statements are derivations from the code text; e.g. the claim that mean-filled PM2.5 rows bias ĉ (F2) is analytic, not measured.
2. **Live run artifacts.** `evaluation/iasa_pol/runs/` contained exp01–exp05 at review time (generation in progress). I verified one resolved config (exp01: 40×40, T=72, cuda, git a333d1f) but did not audit the numerical contents of `result.json`/`arrays.npz`.
3. **Vendored internals not line-audited:** `baselines/fieldformer/model.py`, `model/imputation/imputeformer.py`. Their adapters (`fieldformer_adapter.py`, `pol_weather.py` training loop) were read; the training loop looked standard (masked-target reconstruction with early stopping on held-out RMSE, no leakage of held-out tuples into neighbors via `allowed_indices` in the FieldFormer path).
4. **Tests directory** (`tests/test_iasa_*.py`, 18 files) not read line-by-line; coverage inferred from names + the gate harness, which subsumes most contracts.
5. **Record-span assertions** (21,960 timestamps, 32 sensors after Pusa averaging, missingness percentages in `7.evaluation.tex`) not re-counted from `sim/govdata_1H_current.csv`; the loader logic (Pusa averaging by timestamp at mean coordinates, valid-value masks) matches the paper's description.
6. **Substep-boundary subtlety in `_advect`:** trajectories to observation times t and t+1 are recomputed independently; with the default `substep_dt=0.25` dividing `dt=1.0` they coincide on shared segments, but a non-divisor `substep_dt` makes the paths differ slightly near the endpoint (a discretization choice, not an error; exit-removal semantics remain paper-conformant either way). Not numerically probed.
7. **Kriging helper inconsistency (unused path):** `polsim.build_U0_from_govdata_kriging` drops `Pusa_IMD` (legacy) instead of averaging the Pusa pair, and hardcodes `ic_row_idx=745`. Currently dead code in the IASA path — but it is the natural ingredient for fixing F3's missing baseline, at which point this must be reconciled with the paper's Pusa-averaging rule.
8. **`sensor_layout` duplicate-cell dedup** can return fewer than `n` sensors for random/downwind layouts; harmless for one-factor comparisons (same observer on both sides) but the effective sensor count should be reported in E4 rows.
9. **Whether committed `summaries/` (currently only `.gitkeep`) will be regenerated from the paper-scale eval configs** — the summarizer appends to existing CSVs (`csv_path.open("a")`), so mixing pilot-scale and paper-scale rows in one summary file is possible if `summaries/` is not cleaned between config generations. Worth a guard.

---

---

## Coverage log

| Artifact | Status |
|---|---|
| paper/3.setup.tex, 4.method.tex, 5.theory.tex, 6.algorithm.tex, 7.evaluation.tex, 8.evaluation.tex, 9.appendix.tex | **read in full** (6.algorithm's commented-out legacy block noted as superseded by the active §Procedure) |
| paper/0.abstract.tex, 1.intro.tex, 2.related.tex, 8.conclusion.tex, -1.archive.tex | not read (no method/experiment definitions; archive explicitly legacy) |
| ROADMAP_STAMP_NEW.md | **read in full** (both halves) |
| execution_plans: task10 | **read in full** |
| execution_plans: task5_v3, task6_v2, task7, task8, task9, task9b, task9c, gateS7 | objectives/contract sections read |
| execution_plans: task3*, task4, task5_v1/v2/findings, task6, task6a, task7(6-19), task9a, task9d, paper_* | not read (code audited directly against roadmap+paper) |
| model/iasa/: response, projection, background, diagnostics, fit, activity, merge, footprints, refine, wind, backend, fieldformer_adapter | **read in full**, math re-derived |
| model/iasa/__init__.py | not read (exports only) |
| model/imputation/imputeformer.py, baselines/fieldformer/* | **not audited** (vendored; adapters read) |
| sim/pol_sources.py, sim/polsim.py | **read in full** |
| data/pol_weather.py | **read in full** |
| scripts/run_iasa_sanity.py (S1–S7 + parity + wind_field + footprints + refine + fieldformer_train + experiments gates) | **read in full** |
| scripts/impute_new_delhi_wind.py, train_fieldformer_wind.py, smoke_iasa_runtime.py | not read (invoked contracts checked via callers/gates) |
| experiments/iasa_pol/: nd_platform, experiments, edge_hold_pde, run_experiment, runio (header), summarize_results, configs/* | **read in full** |
| evaluation/iasa_pol: configs diffed vs committed; exp01 resolved provenance inspected; run result contents | configs verified; results **not** numerically audited (generation in progress) |
| tests/test_*.py (18 files) | not read line-by-line |
| checkpoints/fieldformer_wind_new_delhi.report.json | read (corroborates Task 9D numbers) |

No code was executed; the review is fully static. No temporary files were created; the only path added to the repository by this review is this report. (Pre-existing uncommitted state at review time, not introduced here: a `.gitignore` edit un-ignoring `evaluation/iasa_pol/**`, and the in-progress `evaluation/` result-generation tree.)
