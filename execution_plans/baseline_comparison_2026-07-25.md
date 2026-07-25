# Plan — Baseline comparison for IASA (2026-07-25)

> **Status (2026-07-25): IMPLEMENTED.** `baselines/receptor.py` (B1/B2/B3),
> `experiment_11` + `_baseline_scenario` in `experiments/iasa_pol/experiments.py`,
> registered as `exp11`, config `configs/exp11.json` (40×40, T=72), sbatch runner at
> `/scratch/ab9738/tmp/exp11.sbatch` (seeds 0–2 on the l40s GPU node → `runs/` +
> `summaries/`). Receptor helpers unit-tested (B1/B3 recover c_true on well-posed
> data). Figure panel (f) + §7 paragraph remain for the figure-panel plan.

## Why
Reviewer + advisor feedback: **no empirical comparison against any alternative.** AAAI
reviewers will ask "what do existing methods get wrong on your data that IASA fixes?"
The strongest, cheapest answer (already half-suggested by our background-stress result):
show a standard receptor / least-squares method **confidently reporting a
non-identifiable split** on the exact same data where IASA flags or merges it.

This plan produces the numbers and one figure panel + short paragraph. It is the
**upstream dependency** for the figure-panel plan
(`figure_panel_space_2026-07-25.md`): panel (f) is drawn from this experiment's output.

## Baselines to implement
Ordered by evidentiary value; B1 + B2 are the required pair, B3 optional.

- **B1 — Plain NNLS (identifiability-layer ablation).** Solve the *same* projected
  system `min_{c>=0} ||Ỹ - H̃_Φ c||²` that IASA solves, but report coefficients with
  a naive covariance-only CI and **no** visibility / coherence gating, **no** merge,
  **no** flag. This is the cleanest apples-to-apples: IASA minus the identifiability
  layer. Reuse the existing projected solver in `model/iasa/fit.py`
  (`fit_projected(H_tilde, Y_tilde, ...)`, ~line 561) with diagnostics/merge skipped.
- **B2 — PMF / NMF (receptor model).** Factor the sensor×time observation matrix `Y`
  into `W·H` with `K` factors via `sklearn.decomposition.NMF` (a standard PMF proxy;
  no inventories, no transport). Associate each factor to an inventory group by best
  spatial/temporal correlation, then read off the K-way apportionment. Represents the
  dominant real-world SA family (PMF), which emits no identifiability warning.
- **B3 (optional) — CMB.** NNLS of receptor concentrations on inventory-derived source
  profiles **without** the background projection `P_Q^⊥` (classic chemical mass
  balance). Shows background absorbing even more signal than B1.

## Scenarios (reuse existing scaffolding) — IMPLEMENTED, with an evidence-based pivot
Both are geometries where IASA already does the right thing, so the contrast is sharp.

**Pivot note (2026-07-25):** the original S-A ("coherent pair" via a shifted-copy
offset, `experiment_2`) turns out **not** to be genuinely non-identifiable — the
paper's own exp02 shows the pair stays full-rank with ~0 recovery error even at
offset 1 (max coherence only ~0.958 < τ_ρ=0.99, never merged). So the offset
construction cannot make a baseline fail. S-A was replaced with the **wind/geometry
collapse** (exp04's single-wind + random-layout cell), which is a *true* collapse:
max coherence → 1.0, σ_J → 0, and even IASA's individual coefficient error is ~0.57.

- **S-A wind/geometry collapse** — single steady wind + seeded random sensor layout,
  two nearby sources. σ_J→0, coherence→1: the fine split is non-identifiable. IASA
  flags (coherence≥τ_ρ / σ_J≈0, and reports the identifiable *grouped* activity);
  B1/B2/B3 report the individual split (~0.57 coef error) with **no warning**. Headline
  metric is *coefficient* error (per-group shares are trivially determined here).
- **S-B background stress** — `experiment_3` source-like `stress` basis (with
  `beta_scale=2.0` so the source-like background genuinely bites). IASA detects
  visibility→0, absorption→1, σ_J→0 and flags; B1 (projected, unregularized) explodes,
  B3/CMB (no projection) silently absorbs the background into source coefficients, and
  none of the baselines emit a warning.

## Metrics (per method × scenario)
| metric | IASA | B1 | B2 | B3 |
|---|---|---|---|---|
| individual coef. relative error | | | | |
| grouped coef. relative error | | | | |
| reported CI width / "no warning" | qualified | naive | none | none |
| identifiability flag or merge raised? | **yes** | no | no | no |
| projected residual norm | | | | |

The headline: baseline **residual is as small as IASA's** (fits fine) yet **individual
error is large and no warning is emitted** — accuracy ≠ identifiability, made empirical.

## Steps
1. **`baselines/receptor.py`** — new module:
   - `plain_nnls(H_tilde, Y_tilde, lam, ...) -> dict` (calls the fit.py projected
     solver; returns c_hat, naive per-coef std from `(HᵀH)⁻¹σ²`, residual_norm; **no**
     diagnostics).
   - `pmf_nmf(Y, K, source_maps=None, ...) -> dict` (sklearn NMF; factor→group
     assignment by correlation; returns per-group shares + residual).
   - `cmb(H_lag, Y, ...) -> dict` (optional; unprojected NNLS on source profiles).
2. **`experiment_11_baselines(platform, cfg, seed)`** in `experiments/iasa_pol/experiments.py`:
   - Build S-A and S-B systems via the existing `forward()` helper (bundle already
     exposes `_projection.H_tilde`, `_Y`, `_fit`, `_diagnostics`, `_H_lag` — reuse them;
     add `Y_tilde` to the bundle if not already surfaced).
   - Run IASA (from the bundle) + B1 + B2 (+ B3) on the identical matrices.
   - Emit a `rows` table with the metrics above (JSON-safe), plus `arrays` for plotting.
   - Register as `"exp11"` in the `EXPERIMENTS` dict.
3. **`experiments/iasa_pol/configs/exp11_baselines.json`** — config (K, offset for S-A,
   stress mode for S-B, seed, λ).
4. **Run on a GPU node via sbatch** (per memory `run-tests-on-gpu-node` /
   `runtime-container-overlay-slurm`): `run_experiment.py --config configs/exp11_baselines.json`.
   Output → `evaluation/iasa_pol/runs/exp11_*` + a `summaries/` entry.
5. **Transcribe results** into (a) `paper/figures/make_figures.py` panel (f) and
   (b) a short §7 "Controlled Results" paragraph ("Baselines and Identifiability").
   These land via the figure-panel plan.
6. **Sanity checks** (verification in lieu of formal gates — see memory
   `skip-tests-gates-remaining-tasks`): baseline residual ≤ IASA residual (baselines fit
   at least as well nominally); baselines raise no flag; S-A individual error ≫ grouped
   error for baselines; numbers stable across 2–3 seeds.

## Risks / notes
- Keep baselines *charitable* (well-tuned, same λ where applicable) so the contrast is
  "even a fair baseline can't see the non-identifiability," not a strawman.
- PMF factor→group assignment is the fiddly part; if correlation assignment is
  ambiguous, report the *best-case* assignment for the baseline (again charitable).
- Scope: this is a **controlled** demonstration of a failure mode, not a full benchmark
  suite — frame it that way in the text to avoid inviting a "why not 5 datasets" ask.
