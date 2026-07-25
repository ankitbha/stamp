# Plan — Bring results into the main text: two figures + eval rewrite (2026-07-25)

Supersedes the reclamation half of `figure_panel_space_2026-07-25.md` (that plan
assumed one 6-panel figure; we now have ~1 freed page from the section merge and
enough results for two figures).

## Goal
Surface **most/all** results in the main text as figures: add the new baseline
comparison (exp11) and convert the appendix experiment tables into panels. Target
eval ≈ 2.5 pages. Keep the precise numeric tables in the appendix as backing.

## Figure grouping (decided)
Two full-width `figure*` floats, split by question:

### Figure 1 — `fig_controlled.pdf` "Controlled identifiability" (2 rows)
The synthetic experiments establishing that `H̃_Φ` geometry governs recovery/robustness.
- (a) **Exp1** conditioning sets the recovery ceiling (coef err vs noise, 2 geometries) — *existing*.
- (b) **Exp4** wind×layout (σ_J vs max coherence; single/random collapse) — *existing*.
- (c) **Exp3** background stress (visibility/absorption/σ_J bars) — *existing*.
- (d) **Exp5** transport error (coef err vs operator-error norm, by axis; structural-mismatch adequacy=1.0 as caption note) — *new, from* `tab:results_h5a`.
- (e) **Exp7** lag selection (σ_J ↑ and κ ↓ vs lag; selected L=16) — *new, from* `tab:results_lag`.
- (f) **Exp9** temporal-basis recovery (coef err vs activity err vs noise) — *new, from* `tab:results_temporal`.
- (g) **Exp6** inventory robustness (σ_J across 5 scenarios; recovery exact) — *new, from* `tab:results_h5b`.
Layout: 2×4 grid with (h) empty or used for a compact legend. Exp2 (coherence
sweep, `tab:results_h2`) stays **text-only** — it's the confirmatory "high
coherence but no merge, exact recovery" case; a panel would under-deliver.

### Figure 2 — `fig_baselines_observed.pdf` "Baselines and observed New Delhi" (1 row)
- (a) **Exp11 baselines** [headline new result]: IASA vs plain-NNLS/CMB/PMF on the
  two non-identifiable scenarios — bars of share-error (bounded, interpretable),
  annotated with "flag raised?" (IASA yes, all baselines no). From
  `evaluation/iasa_pol/runs/exp11_seed0/result.json` (share_l2_error is the clean
  metric; coef errors blow up and are unbounded, so use shares in the panel and
  cite the coef/flag contrast in text).
- (b) **Observed geometry**, weeks 1–4 (σ_1/σ_J bars + max coherence line) — *existing (d)*.
- (c) **Observed apportionment**, weeks 1–4 stacked shares — *existing (e)*.

Exp10 footprints (centroid 0.74 cells, 100% within radius, contribution-sum
error 0) stays a **text callout** — no array data for a spatial panel.

## Data sources (all in hand)
Existing panels: `make_figures.py`. New panels transcribed verbatim from the
appendix tables listed above (a faithful re-encoding, as the current figure
already is) and from the exp11 result.json. No new runs needed.

## `make_figures.py` changes
- Split `fig_results()` into `fig_controlled()` → `fig_controlled.pdf` and
  `fig_baselines_observed()` → `fig_baselines_observed.pdf`.
- Fig 1: `subplots(2,4)`, figsize ≈ (7.15, 3.8), legends outside axes, direct curve
  labels where possible (as (a) already does).
- Fig 2: `subplots(1,3)`, figsize ≈ (7.15, 1.95).
- Reuse the existing rcParams/style; keep every legend off the data face.

## Eval section (`7.evaluation.tex` + `8.evaluation.tex`) rewrite
- Intro + platform: keep.
- **Controlled Results**: reference Fig 1; one compact run-in paragraph per
  experiment — Conditioning (a), Wind×Geometry (b), Background Stress (c),
  Transport Error (d), Lag Selection (e), Temporal-Basis Recovery (f), Inventory
  Robustness (g); Coherence/Grouping (Exp2) and Footprints (Exp10) as one-liners;
  Missing-Source Adequacy (Exp8) stays.
- **Baselines**: new run-in paragraph (Exp11) → Fig 2(a). Headline: standard
  methods confidently report a non-identifiable split (large error, no warning);
  IASA flags it in every scenario and seed, and under background stress still
  recovers apportionment to <0.01 share error.
- **Observed New Delhi**: wind-imputation validation; identifiability/report
  groups → Fig 2(b); apportionment/uncertainty → Fig 2(c); footprints one-liner.
- Update all `fig:results` refs → `fig:controlled` / `fig:baselines_observed`
  (a–e remap). Verify no dangling refs.

## Consistency fix — restore tables for the already-figured experiments
Earlier compression (commit `91b1c28`) deleted the tables for the experiments that
became figure panels. For consistency (every figure panel should have a backing
table, as Exp5/6/7/9 do), restore from `b5901c8:paper/8.evaluation.tex` — values
verified to match the current panels:
`tab:results_h1` (Exp1), `tab:results_h3` (Exp3), `tab:results_h4` + `tab:results_h4_ens`
(Exp4), `tab:results_missing` (Exp8), `tab:results_nd_ident` (obs geometry),
`tab:results_nd_appt` (obs apportionment), `tab:results_nd_resid` (obs residual).

## Appendix changes (`9.appendix.tex`)
- `app:additional_experiments`: reword intro from "deferred from Section 5" to
  "full numeric tables backing the Section 5 figures"; **keep** tables
  `tab:results_h2/h5a/h5b/lag/temporal` (referenced from the new figure captions),
  trim the now-duplicated prose to 1–2 sentences each.
- Add a small **exp11 baseline table** (from the committed summary CSV) as numeric
  backing, referenced from Fig 2(a)'s caption.

## Space budget
Fig 1 (2 rows ≈ 3.8in) + Fig 2 (1 row ≈ 1.9in) full-width floats ≈ ~1.3 pages of
figure area; eval prose ≈ ~1.2 pages → eval ≈ 2.5 pages. Fits within the ~1 page
freed by the section merge; keep discussion/references/appendix page breaks intact.
No AAAI spacing hacks.

## Verify
Rebuild in-container; 0 undefined/duplicate refs; eval ≈ 2.5 pp; figures render with
all legends outside the data area; every appendix table still referenced.
