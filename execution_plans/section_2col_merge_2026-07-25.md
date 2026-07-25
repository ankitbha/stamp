# Plan — Two-section methods spine (merge §3+§4 and §5+§6) — 2026-07-25

> **Status (2026-07-25): IMPLEMENTED.** §3 (Problem Setup = old 3+4, aggressively
> compressed) and §4 (Identifiability-Aware Source Apportionment = old 5+6, kept
> largely intact) are live; `4.method.tex`/`6.algorithm.tex` deleted and dropped
> from `main.tex`. Both `sec:identifiability_theory` and
> `sec:identifiability_aware_method` labels sit on the merged §4 header so all 17
> external refs resolve unchanged; `sec:wind_conditioned_source_apportionment`
> repointed to `sec:problem_setup`. Build clean (19 pp, 0 undefined/duplicate).
> Net: the main text now ends on **p6** (was p7) — ~1 page freed for the eval.
> Methods are now **2 sections** (were 4).

## Goal
Collapse the four methods/theory sections into **two**, freeing ~1 column (almost
all of it from the §3+§4 merge) to give the evaluation room to grow to ~2.5 pages
(baselines + additional experiments). New spine:

1. **§3 "Problem Setup"** = current §3 (Problem Setup) + §4 (Wind-Conditioned
   Response), **aggressively compressed** (~1 column removed): one background
   treatment, one response treatment, mechanism detail relocated to the appendix.
2. **§4 "Identifiability-Aware Source Apportionment"** = current §5
   (Theory/Identifiability) + §6 (IASA), **combined but keeping most of the text**
   — mainly drop the §6 header, flow the algorithm after the diagnostics, fix the
   now-internal cross-references. Light compression only.

## File moves
- Merged §3 text → `3.setup.tex`. Delete `4.method.tex`, remove its `\input` in `main.tex`.
- Merged §4 text → `5.theory.tex` (retitle). Delete `6.algorithm.tex`, remove its `\input`.
- Resulting `\input` order: `3.setup, 5.theory, 7.evaluation, 8.evaluation, 8.conclusion`.

## Label plan (verified reference map)
Keep as the two section anchors: `sec:problem_setup` (§3), `sec:identifiability_aware_method` (§4).

Repoint:
- `sec:wind_conditioned_source_apportionment` → `sec:problem_setup`
  (2 refs: `6.algorithm.tex`, `9.appendix.tex`).
- `sec:identifiability_theory` → `sec:identifiability_aware_method`
  (10 refs: `6.algorithm.tex`×3 become internal, `8.evaluation.tex`×1, `9.appendix.tex`×6).

Preserve (referenced externally from eval/appendix; keep the labels on surviving
paragraphs of merged §3): `subsec:wind_field_estimation`, `subsec:plume_response`,
`subsec:background_construction`. Drop `subsec:response_and_fit` (0 refs).

Fix now-internal self-references (a section can't `\ref` itself): §4-origin text that
said "Section~\ref{sec:problem_setup}" and §6-origin text referring to
`sec:identifiability_theory`/`sec:identifiability_aware_method` — reword to "above"/
"the diagnostics of this section" etc.

## Equations — keep numbered only if referenced
Must stay numbered (referenced elsewhere): `eq:constructed_response`,
`eq:lag_convergence`, `eq:observation_selection`, `eq:projected_model`,
`eq:wind_direction_conversion`, `eq:nnls_estimator`.
Safe to un-number/inline to save space (0 refs): `eq:model_with_background`
(in §3). Leave §5's `eq:rank_condition`/`eq:nnls_noise_bound` (0 refs but §5 is
kept largely intact; un-numbering optional, low priority).

## §3 compression strategy (the ~1 column)
Merge the two sections into one flow and cut duplication:
1. **Transport + sparse observation** (from §3): the inverse problem — tighten.
2. **Inventory-based source–basis parameterization** (from §3): keep.
3. **Wind-conditioned lagged response** — merge §3's abstract "Lagged
   wind-conditioned response" with §4's "Masked wind preparation" +
   "Open-boundary lagged plume response": define the fingerprint
   (`eq:constructed_response`, `eq:wind_direction_conversion`), state the puff
   operator in 2–3 sentences, **relocate** puff dynamics / dispersion covariance /
   exit bookkeeping to `app:transport_response` (mostly already there).
4. **Background + projection** — collapse §3 "Background and temporal correction"
   and §4 "Background construction and projection" into **one** paragraph: define
   `P_Q^\perp`, `eq:projected_model`, arrive at `\widetilde H_\Phi`; relocate
   column-family list / basis-rank specifics to the appendix pointer.
5. **NNLS fit** (`eq:nnls_estimator`) → `\widehat c`; relocate regularizer/
   refinement variants to `app:transport_response`.
Un-number `eq:model_with_background`. Net target: ~1 column shorter than §3+§4 today.

## §4 merge strategy (keep most text)
- Drop the `\section{Identifiability-Aware Apportionment}` header; retitle
  `5.theory.tex`'s section to **"Identifiability-Aware Source Apportionment"**.
- Order: intro/identifiability object → Noise-Robust Identifiability (props) →
  Identifiability Diagnostics → Identifiable Source Resolution → Per-sensor
  footprints → **IASA algorithm float** → step-order discipline → Model Adequacy,
  Uncertainty, Reporting.
- Rewrite the IASA intro sentence ("combine the estimator of
  Section~\ref{sec:wind_conditioned_source_apportionment} with the diagnostics of
  Section~\ref{sec:identifiability_theory}") since both now live in the paper's
  §3 and this section: "combine the estimator of Section~\ref{sec:problem_setup}
  with the diagnostics above."
- Keep `alg:iasa`, both propositions, all referenced equations. Trim only obvious
  connective redundancy; do **not** gut the diagnostics.

## Steps
1. Read current `3.setup.tex` + `5.theory.tex` in full (have §4/§6 in context).
2. Author merged `3.setup.tex` (Problem Setup).
3. Author merged `5.theory.tex` (Identifiability-Aware Source Apportionment).
4. Update `main.tex` inputs; delete `4.method.tex`, `6.algorithm.tex`.
5. Repoint the two section labels across `6.algorithm`(gone)/`8.evaluation`/`9.appendix`
   and fix internal self-refs.
6. Rebuild in-container; **verify**: 0 undefined/duplicate refs, main text page count,
   ~1 column freed vs before.

## Risk
Heavy ref-repointing (12 section refs + equation renumbers). Mitigation: Python
aux-parse + grep for dangling refs after the build, as done earlier this session.
Do not touch AAAI template spacing; all savings from merge + appendix relocation.
