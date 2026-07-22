# IASA workflow

Identifiability-aware source apportionment (IASA) on the New Delhi PM\(_{2.5}\)
platform. This is the detailed companion to the top-level `README.md`. All
commands run inside the container (see `README.md` for the runtime pattern);
GPU work goes through SLURM (`--account=torch_pr_633_general
--partition=l40s_public --gres=gpu:1`).

## Recommended pipeline

1. Build or load named inventories (four normalized proxy source groups:
   brick kilns, industries, population density, traffic).
2. Load and impute New Delhi `WD/WS` into a gridded wind field queried per
   response-grid cell (the adopted kernel coordinate-query imputer), or choose a
   synthetic wind provider; build transport wind-field ensembles as needed.
3. Build temporal activity bases (traffic diurnal slots, kiln block, industry
   day/night, population cooking) with the fixed-zero admissibility mask.
4. Declare the primary background \(Q\), the lag-candidate grid, the inventory
   version, and any fixed-zero coefficient mask -- all before fitting.
5. Generate (controlled) or load (observed, with the PM\(_{2.5}\) mask \(M_O\);
   PM\(_{2.5}\) is never imputed) the observations.
6. Build \(H^{\mathrm{lag}}\), select the lag by response convergence
   (`tau_L=1e-3`), and preserve the full sweep.
7. Build \(Q\) and project to \(\widetilde H\) via \(P_Q^\perp\).
8. Run realized-wind and, when requested, wind-distribution diagnostics
   (quantiles `0.05/0.50/0.95`).
9. Fit nonnegative source-basis coefficients (projected FISTA) and check residual
   adequacy **only** when a calibrated external noise model is available
   (`alpha=0.05`, 1000 bootstrap refits).
10. Optionally run constrained end-to-end refinement, accepting it only if it does
    not degrade identifiability.
11. Report source contributions, per-sensor contributions and footprints,
    observation/transport uncertainty, inventory robustness, and conservative
    merge recommendations.
12. Evaluate controlled Experiments 1--10 and the observed New Delhi apportionment.

## Command reference

Gates (deterministic, public-API):

```bash
python3 scripts/run_iasa_sanity.py --gate <name>      # single gate
python3 scripts/run_iasa_sanity.py --gate all         # light regression set
python3 scripts/run_iasa_sanity.py --gate all --strict-all   # + calibration, experiments, reporting
```

Wind product (optional; the kernel field is built on demand, so this is only for a
saved product):

```bash
python3 scripts/impute_new_delhi_wind.py --start '<ts>' --end '<ts>' --device cpu \
  --output data/new_delhi_wind_imputed.npz
```

Controlled experiments and observed windows (GPU):

```bash
python3 experiments/iasa_pol/run_experiment.py \
  --config evaluation/iasa_pol/configs/expNN.json --device cuda \
  --out evaluation/iasa_pol/runs
# observed weekly windows: configs observed_week1..4.json (T=168 each), out week<K>/
```

Roll-ups and paper tables:

```bash
python3 experiments/iasa_pol/summarize_results.py \
  --runs evaluation/iasa_pol/runs --summaries evaluation/iasa_pol/summaries
python3 experiments/iasa_pol/summarize_weeks.py \
  --runs evaluation/iasa_pol/runs/week1/observed_seed0 ... --out evaluation/iasa_pol/summaries
python3 evaluation/eval_pol_iasa.py \
  --runs evaluation/iasa_pol/runs --out evaluation/iasa_pol/reports
```

Paper build: see `README.md` (`TEXINPUTS=.:./icml2026//: pdflatex ...`).

## Artifacts and provenance schema

Each run writes `runs/<experiment>_seed<N>/` (observed: `week<K>/observed_seed0/`):

- `config.resolved.json` -- platform config (grid, T, lag rule), inventory version,
  device, dtype, git SHA, and runtime provenance, all recorded before/independent
  of fit results.
- `result.json` -- accuracy (when synthetic ground truth exists) and
  identifiability diagnostics; observed mode reports geometry/residuals/groups and
  never a synthetic recovery error.
- `arrays.npz` -- `H_tilde`, `Y`, `c_hat`, singular values (where produced).

Key result fields:

- Diagnostics: padded singular spectrum, `sigma_J` (zeroed when numerically
  rank-deficient), numerical/effective rank, condition number/status, per-coefficient
  visibility, background absorption, maximum eligible coherence and ray distance
  (JSON `null` for weak/ineligible pairs -- never NaN), weak set, ambiguous pairs.
- Merge: deterministic connected components (`report_components`), retained trigger
  edges (`source_edges`; A--B--C chains keep both), `is_conservative`.
- Ensemble kind: `transport`, `inventory`, or the neutral `single` for a plain fit;
  the aggregators refuse to pool mixed kinds.
- Adequacy: `calibration_status`; a run with no external calibrated noise model is
  `uncalibrated` and emits no pass/fail. Paper defaults `alpha=0.05`, 1000 refits.
- Wind: `wind_provider` (e.g. `gridded_kernel_new_delhi`), imputer metadata.
- Observed: `window_start/end/index`, `pm25_imputed=false`,
  `kriged_baseline_subtracted`, `sensor_signal_contribution_shares` (fractions of
  fitted sensor signal, with the explicit denominator), `per_monitor_group_contributions`.

## Defaults

- Lag convergence tolerance `tau_L = 1e-3`.
- Adequacy `alpha = 0.05`, 1000 bootstrap refits.
- Wind-distribution quantiles `0.05 / 0.50 / 0.95`.
- Device `cpu` default (`cuda` supported); `float64` inverse/diagnostics/fit,
  `float32` response.

## Scope and legacy

Identifiability results are conditional on the declared inventory, transport,
temporal basis, lag, background, mask, and noise assumptions. Report groups are
conservative merges, not a guaranteed finest partition. Inventory scenarios are
robustness rows, not confidence intervals. Archived legacy code lives only in the
git-ignored `archive/` and is out of the active contract.
