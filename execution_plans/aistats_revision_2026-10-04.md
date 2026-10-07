# Plan — AISTATS 2027 revision of IASA: learning at the identifiable resolution (2026-10-04)

> **Status (2026-10-04): full draft in `paper_aistats/`. T1, T2, T3, E1, E2, A1, W1, W2 and W3
> are implemented; D6 (`stamp/.venv`) is created.** Nothing is committed. T1 uses the signal
> target A_g c_g (decided 2026-10-04). It replaces the JASA plan, which is dropped. The framing
> comes from `execution_plans/aistatsfeedback.pdf`, pages 1–7. Deadline: full paper and all
> supplementary material by Tue 6 Oct 2026, 23:59 AoE.

## Update 2026-10-06 (later): refocused on the application; this plan is superseded

- **Superseded by `execution_plans/aistats_applications_plan_2026-10-06.md`.** Ankit could not
  verify Section 4.3 onwards or the synthetic study, and the paper now targets the AISTATS
  applications track (PM2.5 source attribution).
- **Archived.** Sections 4.3–4.5 (separation, reportable partitions, stability, uncertain
  fingerprints, the IASA procedure), Section 5 (synthetic study), their proofs, the synthetic
  appendix and the related Appendix B passages are in `paper_aistats/sections/archive.tex`, which
  `main.tex` does not input. The update below describes material that is now archived.
- **State.** Clean build, 30 pages, 0 undefined references; the main text ends at 4.77 pages.
  Appendix B states the interim merge rule (pairwise coherence above τ_ρ = 0.99, starting from
  𝒫⋆ ∨ 𝒟). Nine bibliography entries no longer print; 51 print.

## Update 2026-10-06: uncertain fingerprints, verification, 8-page cut

- **New theory (Section 4.4, proofs in Appendix A).** Theorem 4.10: under a margin on the
  latent operator, the thresholded grouping is unique and equals the latent grouping.
  Proposition 4.11: the rule "smallest largest-discarded singular value D(P)" returns the
  latent grouping when ||Delta||_2 < d*/2, with a Gaussian probability bound; D(P) has a
  Gaussian p-value under a stated margin hypothesis. Corollary 4.12: the reporting bound holds
  for every partition at once (also for N(0, Sigma) noise with sigma^2 -> ||Sigma||_2).
- **Code.** `model/iasa/grouping.py`: `latent_grouping`, `largest_discarded`,
  `gaussian_discard_p_value`; 3 new tests (14 grouping tests pass). Study part `e1f` in
  `experiments/iasa_synthetic/run_study.py`: 242 ambiguous instances, rule correct in all 242,
  the guarantee's condition met in 54.
- **Independent proof verification (subagent, three rounds).** Round 1 found 2 wrong
  sentences, a quantifier error in the p-value, missing hypotheses (lambda = 0; c >= 0 in
  Corollary 4.12; arcsin domain in Prop 4.9) and several gaps; all fixed and confirmed in
  round 2. Round 3 checked the text changed by the page cut: one wrong sentence (the greedy rule described as finding the best minimal partition; counterexample found) and four wording gaps, all fixed; round 4 confirms.
- **Page cut (layout subagent).** Plan based on AISTATS 2023-2025 papers; applied with the
  verifier fixes kept. Main text now ends on page 8, left column (7.45 pages); the layout agent passed its checks for page count, formatting rule, moved material and kept verifier fixes, and its three self-containedness findings (separation range, Figure 2 crop, greedy sentence) are fixed. Moved to the appendices:
  proof sketches, the dynamic-programming and greedy description, the latent-grouping window
  argument, Remark 4.4, the full controlled-experiments figure, extended related work.
- **Controlled results restored (2026-10-06).** The full Section 6.1 text (baselines,
  conditioning, the six other controlled axes, recorded groupings) is back in the main text with
  the two-panel figure; main text ends on page 8 at 7.86 pages. Restoring the eight-panel figure
  as well gives 8.28 pages (34 column-lines over), so that figure stays in Appendix F.
- **Section 6 restructured (2026-10-06).** The full eight-panel controlled figure is back in
  the main text (Figure 2) and its appendix copy removed; the observed-weeks analysis moved to
  Appendix G.9 ("Observed Weeks: Full Analysis"), and Section 6.2 is a one-paragraph summary
  with Figure 3. The main text ends at the bottom of page 8 (8.00 pages); the AI-use statement
  starts page 9.
- **Related work.** Subspace clustering (Elhamifar and Vidal 2013; Soltanolkotabi and Candes
  2012; Wang and Xu 2016) with the three differences stated.
- **For the authors to confirm.** The IASA expansion is now "Identifiability-Aware Source
  Attribution"; the earlier versions used "identifiability-aware apportionment".

## Implementation status (2026-10-04, full draft)

**Build.** `cd paper_aistats && latexmk -pdf main.tex`: 35 pages, no undefined references or
citations, no bibtex warnings, one 5.1 pt overfull box from the template's abstract box. The main
text runs to page 10, so it exceeds the 8-page limit by about two pages; the page limit was not
addressed, as instructed.

**New code (untracked).**
- `model/iasa/grouping.py`: exact matroid components, separations, reportable partitions (exact
  enumeration over up to 12 atoms, greedy beyond), declared-partition join. Tests:
  `tests/test_iasa_grouping.py` (11 tests; a broken-components control fails the main test).
- `experiments/iasa_synthetic/` (`generator.py`, `baselines.py`, `run_study.py`) →
  `evaluation/iasa_synthetic/results.json` (E1a–E1e, E2; about 22 s on CPU).
- `experiments/iasa_pol/signal_target_weeks.py` → `evaluation/iasa_pol/signal_target/weeks.json` (A1).
- `experiments/iasa_pol/experiments.py`: Exp 11 now also records signal-target scores
  (`_signal_target_scores`); re-run on CPU → `evaluation/iasa_pol/runs_signal_target/exp11_seed{0,1,2}`.
- `paper_aistats/figures/make_synthetic_figure.py`, `paper_aistats/figures/make_figures.py`
  (panel (a) of the controlled figure and the observed figure now read the result files).
- Tests run: `tests/test_iasa_grouping.py`, `tests/test_iasa_experiments.py`,
  `tests/test_iasa_reporting.py`: 30 passed.

**Theory written (Section 4, Appendix A), each statement checked numerically first.**
- Theorem 4.6 (T1): identifiable ⇔ direct sum ⇔ rank additivity; unique finest partition = matroid
  components via the fundamental-circuit graph; join with a declared partition. Coefficient-sum
  target: non-unique example and NP-completeness by reduction from Partition (Appendix A).
- Theorem 4.8 (T2): LS and NNLS group-signal error ≤ ‖P_A ε‖/s_g; Gaussian LS version
  E‖·‖² ≤ σ² r_g/s_g² (equality for r_g = 1) and a 1−δ bound. Proposition 4.3 sharpened to
  ‖ĉ−c‖ ≤ ‖P_A ε‖/σ_J for LS and NNLS (factor 2 removed).
- Proposition 4.10: reportable partitions are not closed under coarsening; minimal ones can be
  non-unique (3-column example). Proposition 4.11 (T3): angle-perturbation bound and margin
  condition under which the reportable and minimal reportable partitions do not change.

**Corrections found while implementing (all reflected in the draft).**
- Exp 11 baseline numbers in the AAAI paper do not reproduce: plain NNLS under background stress
  is the same estimator as IASA's fit; its committed GPU share error 1.231 came with a coefficient
  of 2.8×10¹⁴ on a column of norm 10⁻¹⁶ (CPU seed 0: 0.006, identical to IASA). PMF/NMF varies with
  initialization (seed-0 collapse: 0.085 on GPU, 1.33 on CPU). The draft reports projected
  (signal-target) shares and group-signal errors over three CPU seeds.
- In the Exp 11 collapse, seeds 0 and 1 have operator columns of norm below 10⁻¹⁰: no plume
  reaches the six sensors. In Exp 4, the random-layout "collapse" is source B being nearly
  invisible (column norm 5×10⁻⁷–1.7×10⁻⁴ against 0.06–0.44 for source A), not two coincident
  signatures. The draft describes both this way.
- The grouped share error was constant by construction for a two-source merge and is no longer
  reported.
- The repo's `experiments/iasa_pol/configs/*.json` are reduced-scale (20×20, T = 24); the
  committed runs used the paper-resolution configs stored in each run's `config.resolved.json`.
  Re-run on CPU with those configs (scratchpad outputs, not committed), Exp 1–10 reproduce every
  recorded scalar within 1% relative difference (0 differing values in each of the ten result
  files; run times 29–1,018 s per experiment).

**A1 result.** All four source groups have separation 0.72–0.97 in every week, so the reported
partition is the four groups. At σ_e = 30.7 µg/m³ (Pusa pair, full record; the pair shares no hours
in May 2018) the NNLS group-signal bound is 2.4–3.3 µg/m³ per reading; with the lag-1
autocorrelation 0.857 it is 8.8–12.0. Fitted group signals are 0–11.9. Bootstrap 95% intervals of
the largest share: week 1 0.75–1.00, week 2 0.71–0.85, week 3 0.00–0.94, week 4 0.33–1.00.

**Remaining before submission.**
- Cut the main text to 8 pages.
- Confirm the AI-use statement (drafted for this revision only) and the checklist answers.
- Produce the anonymized code archive for the supplementary material.
- Decide whether to commit; nothing is committed.

## Short version

The AAAI-27 reviewers and the framing document make the same point: the propositions are
correct, but full column rank and a 1/σ_min error bound are standard linear inverse-problem
facts. The revision changes the question the paper answers. It stops presenting IASA as a
source-apportionment method with a diagnostic layer. It becomes a general statistical
inverse problem:

> Observations y = A(ω)c + Qγ + ε come from nonnegative latent components c, a nuisance
> subspace Q, and an operator A(ω) that depends on observed context ω. **What is the finest
> grouping of the components whose sums can be reliably identified from finite, noisy
> data?**

Pollution source attribution becomes the main application: ω is the wind, A(ω) = H_Φ^lag,
and Q is the background. The paper's message has three levels:

1. **Can c be estimated?** Often it cannot.
2. **Which functions of c can be?** The paper characterizes them.
3. **What is the finest resolution that can be safely reported?** The paper computes it and
   shows it empirically.

Additions, in the framing document's priority order:
- **T1, identifiable aggregates.** A definition and a characterization: block sums are
  identifiable exactly when every block indicator lies in the row space of A. A verified
  example shows that the finest identifiable grouping need not be unique.
- **T2, a noise bound for the sums.** Its constant is ‖R_P A⁺‖₂, not 1/σ_min(A).
- **E1, an application-independent synthetic study.** Phase-transition diagrams for J up
  to 100.
- **E2, baselines that address the same statistical question.** Most importantly a
  truncated-SVD spectral baseline.
- **T3, a stability condition** under which the reported grouping is unchanged by operator
  error.
- **W1–W3, rewriting:** the introduction reframed, the transport machinery moved to the
  appendix, the AISTATS template.

All of this runs on matrices and does not need the pollution transport pipeline, which
takes about 24 hours per run. The New Delhi part therefore reuses the committed runs (A1).
Expanding the New Delhi evaluation is out of scope for this deadline.

## Decisions (agreed with Ankit, 2026-10-04)

| # | Decision | Chosen | Not chosen |
|---|---|---|---|
| D1 | Title | *Identifiable Resolution in Structured Inverse Problems: Source Attribution from Sparse Sensors*. It keeps the application visible. | *Learning at the Identifiable Resolution: Latent Attribution under Nuisance Subspaces and Operator Uncertainty* |
| D2 | What IASA reports when several minimal identifiable groupings exist (T1 shows this can happen) | Their common coarsening, which is always identifiable, in the main text. All minimal groupings listed in the appendix. | A rule declared in advance that picks one minimal grouping |
| D3 | Estimator covered by the T2 bound | Least squares / minimum-norm. NNLS, IASA's actual estimator, is checked empirically in E1 and stated as a limitation. | An aggregate bound for NNLS; none is established |
| D4 | Noise tolerance that defines a "stable" grouping | Treat singular directions with σ_i ≤ τ_σ as the null space and apply T1, with τ_σ declared before each experiment | Accept a block when σ‖e_gᵀR_P A⁺‖ is below a declared fraction of its estimate |
| D5 | Where the pollution controlled experiments (exp01–exp11) go | Appendix, with one main-text figure | Keep the current main-text figure |
| D6 | Environment for running new code | A local environment on this laptop, created before E1 runs (details below). **Recorded only; not yet created.** | The NYU cluster container described in `README.md` |

**D6 details, for when the environment is created:**
- **What is known.** On 2026-10-04 no stamp environment existed on this laptop: no venv in
  the repo, no conda environment, and the Homebrew and system Pythons lack torch, scipy and
  einops.
- **Location and Python.** A venv at `stamp/.venv` with Python 3.12. `.gitignore` does not
  currently list `.venv/`; add it.
- **Packages.** The modules the code imported on 2026-10-04: torch, numpy, pandas, scipy
  (`baselines/receptor.py`), scikit-learn (PMF baseline), einops
  (`model/imputation/imputeformer.py`, via `data/pol_weather.py`), and matplotlib.
- **Known invocation problem.** One attempt to run `experiments/iasa_pol/run_experiment.py`
  from the repo root failed with "'experiments' is not a package". The script's directory
  contains `experiments.py`, which can shadow the `experiments` package. Running with
  `python -P` got past that point. Re-check once the environment exists.
- **Runtime.** A run of the pollution pipeline takes about 24 hours on this laptop.

## Scope boundary

| In this plan | Not in this plan |
|---|---|
| T1–T3 theory, each with a proof checked numerically | New runs of the pollution pipeline (~24 h each) |
| Synthetic study (E1) and statistical baselines (E2) on matrices | Lockdown validation, gas tracers, longer New Delhi windows |
| New Delhi re-analysis using the saved H̃ in `evaluation/iasa_pol/runs/week*/` | An aggregate bound for NNLS (D3) |
| Paper rewritten in the AISTATS template, AI-use statement, reproducibility checklist | JASA submission |

## Global constraints

- **Page limit:** 8 pages of main text, using the template in `AISTATS2027PaperPack.zip`.
  References, the AI-use statement, the reproducibility checklist and appendices do not
  count toward it.
- **Proofs:** no theorem enters the paper until its proof is written out and checked
  numerically on random instances.
- **Thresholds:** declared before each experiment and never tuned on recovery error.
- **New Delhi numbers:** taken only from the committed runs. Shares are described as
  fractions of the fitted inventory-attributed signal. Identifiability is stated as exact
  identifiability under the declared model, unless noise thresholds are applied (A1).
- **Code reuse:** the synthetic study uses the repo's own `diagnose_identifiability`
  (`model/iasa/diagnostics.py:145`), `recommend_merges` (`model/iasa/merge.py:99`) and the
  projected fit (`model/iasa/fit.py`), applied directly to synthetic H̃. It does not build a
  separate implementation.

## Reviewer and framing-document concerns, and the item that answers each

| Concern | Source | Item |
|---|---|---|
| The theory is correct but elementary: rank and 1/σ_min | AAAI WfUB; framing document | T1, T2, T3 |
| Baselines are identifiability-blind, so the comparison is easy | AAAI UW4z (W4); framing document | E2 |
| Controlled experiments are mostly K = 2 | framing document | E1 (J up to 100) |
| Real data is four one-week windows with no ground truth | AAAI UW4z, WfUB; framing document | A1, plus a limitation in the text; expansion is out of scope |
| Pollution machinery obscures the statistical problem | framing document | W1, W2 |
| Novelty relative to inverse problems, data assimilation and system identification | AAAI UW4z (W1) | W1 related work, including estimability (see Risks) |
| Background-stress result: small σ_J with small error | AAAI WfUB | W1: Proposition 2 needs noise; reuse the exp01 noise sweep |

## Measured and verified facts this plan starts from

**Theory checks** (numerical, 2026-10-04):
1. **The finest identifiable grouping need not be unique.** Take A with row space spanned by
   (1,1,0,0), (0,0,1,1), (1,0,1,0) and (0,1,0,1), which has rank 3.
   - The identifiable groupings are {1,2,3,4}, {1,2 | 3,4} and {1,3 | 2,4}.
   - The last two are both minimal, and neither refines the other.
   - In general, coarsening an identifiable grouping keeps it identifiable, but the common
     refinement of two identifiable groupings need not be identifiable.
2. **The sums of an identifiable grouping are recovered exactly by least squares.** If the
   rows of R_P lie in the row space of A, then R_P A⁺A = R_P, and the minimum-norm
   least-squares estimate satisfies R_P(ĉ − c) = R_P A⁺ε.
3. **Coherence is an estimator correlation.** With two columns and independent noise, the
   correlation between the two least-squares estimates equals −cos(h₁, h₂), so coherence is
   its absolute value.
4. **Perturbation bound.** For ‖Δ‖ < ‖h‖, the angle between h and h + Δ is at most
   arcsin(‖Δ‖/‖h‖). There were no violations in 20,000 random draws.

**New Delhi run facts** (from `evaluation/iasa_pol/runs/week{1..4}/observed_seed0/`):
- **Thresholds used:** τ_ρ = 0.99, τ_v = 0, τ_σ unset. With τ_σ unset, effective rank
  equals numerical rank. Sources: `experiments/iasa_pol/experiments.py:1199`,
  `model/iasa/diagnostics.py:28-32`, `:227-228`.
- **Coefficients:** J = 7. Traffic has 4 time-slot coefficients; brick kilns, industries
  and population have 1 each (`experiments/iasa_pol/nd_platform.py:360-411`).
- **Noise bound per week.** ‖r‖ is the projected fit residual (`residual_norm`). The last
  column is the error along the least-determined direction, (‖r‖/√N)/σ_J, divided by the
  size of the fitted coefficient vector ‖ĉ‖:

| Week | N | σ_J | ‖ĉ‖ | 2‖r‖/σ_J ÷ ‖ĉ‖ | (‖r‖/√N)/σ_J ÷ ‖ĉ‖ |
|---|---|---|---|---|---|
| 1 | 4,561 | 3.715 | 7.73 | 262 | 1.9 |
| 2 | 4,278 | 5.671 | 14.69 | 143 | 1.1 |
| 3 | 4,315 | 10.154 | 1.10 | 546 | 4.2 |
| 4 | 4,236 | 7.646 | 15.28 | 64 | 0.5 |

- **Instrument noise:** the co-located Pusa_DPCC and Pusa_IMD monitors differ with a
  standard deviation of 43.4 µg/m³ over 17,949 shared hours. If their errors are
  independent and equal in size, each monitor's noise is about 30.7 µg/m³.

## Work items

Each item below gives an estimate of working time under the decisions above, not counting
the time to create the D6 environment. The total is about 20 hours. The deadline leaves about 2.5 days. With only one
working day, do T1, T2, E1 with the spectral baseline only, A1 and W1–W3, and drop T3 and
the remaining E2 baselines.

### T1 — Identifiable aggregates (priority 1; ~3 h)

1. **Definition.** A grouping P = {G₁, …, G_r} of the coefficients is identifiable if the
   block sums z = R_P c are determined by y for every c ≥ 0. Here A = P_Q⊥ A(ω) already
   includes the nuisance projection.
2. **Lemma.** The following are equivalent:
   - P is identifiable;
   - ker(A) ⊆ ker(R_P);
   - every block indicator 1_G lies in row(A).

   The proof follows Proposition 1: the interior point β·1 shows that nonnegativity does
   not help. Corollaries:
   - coarsening an identifiable grouping keeps it identifiable;
   - minimal identifiable groupings need not be unique (fact 1 above).
3. **Relation to the current ambiguity graph.**
   - Connected components of the pairwise-coherence graph are not guaranteed to form an
     identifiable grouping. Three columns can be linearly dependent while every pairwise
     coherence is below τ_ρ.
   - The existing "global unresolved" warning covers that case. T1 replaces the heuristic
     with an exact test and keeps the components as the fallback.
4. **Computing it.**
   - P is identifiable exactly when, with N a basis of ker(A), the rows of N within each
     block sum to zero.
   - For J ≤ 10, enumerate all groupings: Bell(10) = 115,975 candidates, each tested
     cheaply.
   - For larger J, use a greedy merge guided by the null space, and measure its accuracy
     against the exact enumeration for J ≤ 10.
   - The noise-level version follows decision D4.

### T2 — Recovery bound for the block sums (priority 2; ~2 h)

1. **Theorem candidate.** Suppose:
   - ε ~ N(0, σ²I);
   - ĉ = A⁺y;
   - P is identifiable;
   - M = R_P A⁺.

   Then with probability at least 1 − δ,

   ‖R_P(ĉ − c)‖₂ ≤ σ(‖M‖_F + ‖M‖₂ √(2 log(1/δ))) ≤ σ‖M‖₂(√r + √(2 log(1/δ))).

   Proof: R_P(ĉ − c) = Mε by fact 2, then Gaussian concentration for the ‖M‖₂-Lipschitz
   function ε ↦ ‖Mε‖₂. A sub-Gaussian version follows with different constants.
2. **The constant C_P = ‖R_P A⁺‖₂ is set by the singular directions the block sums use.**
   It can be small even when σ_min(A) is near zero. This is the formal statement of "the
   individual coefficients are unstable while the block sums are stable", and it justifies
   IASA's practice of reporting groups.
3. Check numerically on random instances before writing it up. NNLS is covered by D3.

### E1 — Synthetic phase-transition study (priority 3; ~4 h including code)

1. **Generator.**
   - A = U diag(s) Vᵀ with a prescribed spectrum, plus planted groups of nearly collinear
     columns, so the oracle identifiable grouping is known.
   - A random nuisance Q of dimension dim Q, noise level σ, and an operator perturbation
     ΔA of controlled size.
   - c is nonnegative.
2. **Grid.**
   - N ∈ {100, 1,000, 10,000}, J ∈ {5, 10, 25, 50, 100}.
   - Varied: σ_min(A), maximum coherence, dim Q, σ and ‖ΔA‖, with seeds.
   - Matrix sizes up to 10⁴ × 100, so a full grid runs in minutes, not hours.
3. **Metrics.**
   - coefficient error and block-sum error;
   - false-merge and false-separation rates;
   - rate of recovering the oracle grouping exactly;
   - interval coverage;
   - calibration of the identifiable / not-identifiable flag;
   - sensitivity to τ_ρ and τ_σ.
4. **Key figure:** a phase diagram of σ_min(A) against σ, coloured by the rate of
   recovering the correct resolution, with the boundary predicted by T2 overlaid.

### E2 — Statistical baselines (priority 4; ~2 h)

1. **Spectral baseline (required).** Write A = UΣVᵀ, drop directions whose σ_i is below
   the noise threshold, and report the projection of c onto the remaining right-singular
   subspace. Compare with IASA on two things:
   - **Interpretability:** how far the recovered subspace is from the nearest
     block-indicator subspace.
   - **Tightness:** interval width for the same block sums.
2. **Other baselines.**
   - ridge / Tikhonov regularization;
   - Gaussian-prior Bayesian posterior (closed form), with grouping from posterior
     correlation;
   - residual-bootstrap intervals;
   - NNLS, CMB and PMF from `baselines/receptor.py`, kept.
3. **If time allows:** profile likelihood and partial-identification bounds.

### T3 — Stability under operator error (priority 5; ~1.5 h)

1. **Condition for an unchanged grouping.** The weak set and every pair's
   ambiguous / not-ambiguous label stay the same under ΔA if both of these hold:
   - |v_j − τ_v| > ‖ΔA e_j‖ for every coefficient j;
   - for every eligible pair, the gap between that pair's angle and arccos(τ_ρ) exceeds
     arcsin(‖ΔA e_i‖/v_i) + arcsin(‖ΔA e_j‖/v_j).

   The proof uses fact 4. The existing error bound for a perturbed operator (Proposition in
   `paper/9.appendix.tex`, response-matrix perturbations) is reused.
2. **If time allows:** stability of the T1 grouping itself, which needs a singular-value
   gap condition (Wedin-type).
3. Check in E1 by varying ‖ΔA‖.

### A1 — New Delhi re-analysis from the saved H̃ (~1 h)

For each of the four weeks, load `H_tilde` and `c_hat` from `arrays.npz` and compute:
- the identifiable groupings, by exact enumeration (J = 7, Bell(7) = 877 candidates);
- the stable groupings at the instrument-noise level (σ_e ≈ 30.7 µg/m³, decision D4);
- the spectral baseline.

Report these next to the existing results. Write a script under `scripts/` that saves to
`evaluation/iasa_pol/summaries/`. Nothing in the pollution pipeline is re-run.

### W1 — Reframe the paper (priority 6; ~3 h)

1. Title (D1) and abstract. An introduction built on the three-level message. The general
   model y = A(ω)c + Qγ + ε first, then pollution as the main application.
2. **Related work.** Position against:
   - estimable functions in linear models: a linear function λᵀc is estimable exactly when
     λ lies in the row space of the design;
   - Henry's (1992) eligible space in chemical mass balance, which estimates linear
     combinations of collinear sources;
   - aggregation error in atmospheric inversion (Turner & Jacob 2015) and averaging kernels
     (Rodgers 2000);
   - statistical source-apportionment identifiability: Park, Guttorp & Henry (2001); Jin &
     Datta (arXiv:2510.03616);
   - partial identification.

   State what is new against these: groupings whose block sums are identifiable, the
   non-uniqueness of minimal groupings and how to compute them, a nuisance subspace,
   nonnegative components, stability at a stated noise level, and stability under operator
   error.
3. Explain the background-stress result with Proposition 2: a small σ_J causes error only
   when there is noise. Reuse the exp01 noise sweep.

### W2 — Move pollution machinery to the appendix (priority 7; ~1.5 h)

Move these to the appendix: Gaussian puff transport, wind imputation, lag selection,
inventories, observation masks, initial conditions. The main text keeps:
- one paragraph defining the application: ω = wind and A(ω) = H_Φ^lag;
- the New Delhi results;
- one figure of the pollution controlled experiments (D5).

### W3 — AISTATS packaging (~2 h)

- Port the paper to the AISTATS 2027 template.
- Write the AI-use statement.
- Complete the reproducibility checklist.
- Prepare the supplementary material: appendix and code.

## Not by this deadline

- Expanding the New Delhi evaluation, priority 8 in the framing document: new windows at
  about 24 hours per run.
- Lockdown validation against Manchanda et al. (2021), and the gas-tracer checks.
- An aggregate bound for NNLS (D3), and a Bayesian model with a nonnegativity constraint.

## Risks

- **The aggregate criterion is classical estimability.** Statistics reviewers will
  recognize that "λᵀc is identifiable exactly when λ ∈ row(A)" is estimability in linear
  models. Henry's eligible space is the same idea in chemical mass balance. The novelty
  claim must rest on what T1–T3 add on top: groupings, non-uniqueness and how to compute
  them, the nuisance subspace, nonnegativity, noise-level stability and operator stability.
  It must not rest on the criterion itself. Cite estimability explicitly.
- **The greedy grouping for large J may miss the exact answer.** E1 reports its accuracy
  against exact enumeration for J ≤ 10.
- **IASA may not beat the spectral baseline on interval width.** In that case the claim
  rests on interpretability, because the recovered quantities are sums of named sources.
  Report the comparison either way.
- **Time.** The estimates total about 20 hours. With one working day, the reduced set
  listed under Work items applies.

## Sources

- `execution_plans/aistatsfeedback.pdf` (framing; pages 1–7).
- AAAI-27 reviews (Program Committee UW4z, WfUB), decision 24 Sep 2026.
- AISTATS 2027 call for papers (deadlines, page limit, template, supplementary rule).
- Henry (1992), Atmospheric Environment 26A; Turner & Jacob (2015), ACP 15, 7039–7048;
  Park, Guttorp & Henry (2001), JASA 96(456), 1171–1183; Jin & Datta, arXiv:2510.03616;
  Manchanda et al. (2021), Environment International 153.
- Rodgers (2000); estimability in linear models (standard linear-models texts); and
  Wedin's perturbation theorem. These are not yet checked against the published versions.
