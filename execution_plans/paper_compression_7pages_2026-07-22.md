# Paper compression to 7 pages of main text (AAAI 2027)

**Date:** 2026-07-22
**Goal:** Compress the main text (§1–§9, everything before `\bibliography`) from its
current ~17 two-column pages to **7 pages**, self-contained (minimal forward-pointing
to the appendix), with the **appendix following the references**. Nothing scientific
is lost — material either becomes denser (lossless) or moves to the post-reference
appendix.

---

## 1. Current state (audit)

Total compiled length: **22 pages** (`Output written on main.pdf (22 pages)`).

Page map (from `main.aux`):

| Block | Sections | Pages | Approx words |
|---|---|---|---|
| Intro | §1 | 1–2 | 766 |
| Related work | §2 | 2–3 | 686 |
| Problem setup | §3 | 3–4 | 1076 |
| Method (response + fit) | §4 | 4–6 | 1704 |
| Identifiability theory | §5 | 6–9 | 2170 |
| IASA method | §6 | 9–11 | ~1600 live (+1600 dead comments) |
| New Delhi platform | §7 | 12–13 | 1273 |
| Evaluation | §8 | 13–17 | 3540 |
| Discussion/conclusion | §9 | 17–18 | 906 |
| **References** | — | ~18 | — |
| **Appendix A–G** | — | 19–21 | 1735 |

Main text ≈ **17 pages**; we must cut ≈ 10.

**Inventory of compressible artifacts:**

- **Dead code:** `6.algorithm.tex` lines 1–402 are one giant commented-out block
  (an earlier draft of §6). Deleting it is free (0 page impact but removes a large
  maintenance/confusion hazard and any risk of accidental re-enable).
- **Display equations:** ~68 numbered display equations in the main text. A reference
  scan (`\ref`/`\eqref`) shows **only 7 are ever cross-referenced**:
  `eq:lag_convergence` (×2), `eq:observation_selection` (×2),
  `eq:wind_direction_conversion`, `eq:nnls_estimator`,
  `eq:ray_distance_coherence_relation`, `eq:operator_error_bound`,
  `eq:active_covariance`. **The other ~60 are display-only and never referenced** →
  candidates to inline or delete per user rule #6.
- **Tables:** 15 tables. Two are glossary/reference (Table 1 diagnostics, Table 2
  thresholds); one is small setup (Table 3 inventories); **twelve are result tables**
  (Tables 4–15) — the target of the tables→figures consolidation (user rule #1).

---

## 2. Target page budget (7 pages)

| Section | Now | Target | Lever |
|---|---|---|---|
| Title + abstract + §1 intro | ~1.7 pp | **1.0 pp** | tighten prose, shorten 7-item contributions to 4 |
| §2 related work | ~1.2 pp | **0.4 pp** (1 col) | 4 paragraphs → 1 dense paragraph; extended version → appendix |
| §3 problem setup | ~1.5 pp | **1.0 pp** | inline 15 of 17 eqs; keep model + projected `H̃` |
| §4 method | ~2.5 pp | **0.6 pp** | keep NNLS fit + response concept; puff internals + refinement → appendix |
| §5 theory | ~3 pp | **1.3 pp** | keep 2 propositions + diagnostics defs; proofs already in appendix; op-error + footprints → appendix |
| §6 IASA method | ~2.5 pp | **0.7 pp** | keep procedure + threshold summary; uncertainty/adequacy/wind-dist/refinement → appendix |
| §7 platform | ~1.5 pp | **0.5 pp** | keep sensor/inventory/wind essentials; forward-model + kriging detail → appendix |
| §8 evaluation | ~4 pp | **1.2 pp** | 12 tables → 2 figures; ~5 experiments → appendix |
| §9 discussion/conclusion | ~1.2 pp | **0.35 pp** | half column; limitations compressed to a few sentences |
| **Total** | ~17 pp | **≈ 7.05 pp** | trim to fit |

---

## 3. Execution phases (ordered as the user requested)

### Phase 0 — free deletions (no content change)

- **P0.1** Delete `6.algorithm.tex` lines 1–402 (commented dead draft).
- **P0.2** Delete the ICML-carryover `tightalign`/`tightalign*` env if unused after
  Phase 1 (verify no live use first).

### Phase 1 — lossless transforms (denser, same information)

**1A. Tables → figures (user rule #1).** Convert the 12 result tables into **2
consolidated multi-panel figures** (small numbers → plotted, where a panel carries
more signal per cm² than a table). Figures are generated as vector PDFs from the
existing `evaluation/iasa_pol` result JSON/CSV (matplotlib in-container), saved to
`paper/figures/`, and embedded with `\includegraphics`. No re-run of experiments —
plots read committed results.

- **Figure 1 — "Identifiability geometry governs recovery" (main text, ~full column
  width, 3 panels).** Consolidates the central-claim controlled experiments:
  - (a) Exp 1: coefficient error vs noise for *close* vs *separated* geometry,
    annotated with σ_J/κ — conditioning sets the recovery ceiling.
  - (b) Exp 4: σ_J and max eligible coherence across wind provider × layout (grouped
    bars) — wind diversity and sensor geometry.
  - (c) Exp 3: min-visibility / background-absorption / σ_J across the four
    predeclared backgrounds — a source-like background can erase the signal.
- **Figure 2 — "Observed New Delhi, weeks 1–4" (main text, 2 panels).**
  Consolidates Tables 14–15:
  - (a) per-week σ₁, σ_J, max coherence (identifiability geometry, all full rank).
  - (b) stacked bar of proxy apportionment shares per week — the large week-to-week
    swing (population-dominated wk 1–3 → brick-kiln-dominated wk 4).
- **Appendix figures/tables:** the remaining experiments (2 coherence/merge, 5
  transport error, 6 inventory robustness, 7 lag selection, 9 temporal-basis, 10
  footprints) move to an appendix "Additional experiments" section, kept as compact
  tables or small figures (their current tables are already small).

**1B. Equations → inline (user rule #6).** Convert the ~60 unreferenced display
equations to inline `\(...\)` math or fold into prose. Keep as *display* only the 7
cross-referenced ones (above) plus the 2 headline propositions' bound equations.
Concrete high-value inlines:
- §3: inline `eq:continuous_transport, sensor_obs, stacked_observations,
  inventory_matrix, temporal_activity_basis, source_activity, lagged_operator,
  lag_set, time_response, stacked_response, background_basis, model_with_background,
  background_projection, projected_model, source_fingerprints` (keep
  `lag_convergence`, `observation_selection` as display).
- §4: inline `wind_unit_conversion, mass_nonincreasing, puff_ode, puff_discrete,
  gaussian_kernel, dispersion_covariance, sensor_response,
  implicit_background_projection, constructed_response,
  constructed_projected_response, reconstructed_activity, ridge_regularizer,
  prior_regularizer, joint_fit` — most of these leave the main text entirely (→
  appendix, Phase 3).
- §5: inline `theory_projected_model, exact_identifiability_definition,
  numerical_rank, identifiability_score, condition_number, effective_rank,
  visibility, pairwise_coherence, ambiguous_pair_set, ray_distance,
  background_absorption, response_error_decomposition, operator_error_as_noise,
  source_ambiguity_graph, grouped_reports` (keep `rank_condition`, the two noise
  bounds, and `ray_distance_coherence_relation` display).
- §6: inline `projected_residual, residual_variance, adequacy_coordinates,
  residual_adequacy_statistic, refinement_smin_check, refinement_coherence_check,
  reported_quantities` (most → appendix, Phase 3).

**1C. Theorem environments.** Keep Proposition statements (rank identifiability,
noise-robust bound) in main text; proofs are *already* in Appendix B — delete the
inline proof of `prop:rank_identifiability` and `prop:noise_robust` from §5 (they are
duplicated in the appendix) and keep only the one-line intuition after each.

### Phase 2 — cut non-core prose (no appendix section needed)

- **2A. Discussion/conclusion (§9)** → half column (~0.35 pp). Keep: one paragraph
  restating the identifiable-resolution thesis + the `H̃` object; 3–4 sentence
  limitations; one sentence of future work. Cut the design-implications and
  sensing-system paragraphs (their content is implicit in the theory/eval).
- **2B. Related work (§2)** → 1 dense paragraph (~0.4 pp). Keep one sentence per
  strand (receptor/source-oriented apportionment; inverse problems/PINN; sensor-space
  ST learning; observability) + the "we are complementary: identifiability of declared
  inventories" positioning. Move the extended discussion to Appendix "Extended related
  work" (user rule #4 — expanded points allowed in appendix).
- **2C. Concise pass (user rule #3)** across §1, §3, §7, §8 intros: remove restating
  sentences, collapse enumerations, delete "we now…/this section…" scaffolding.
  Intro contributions list 7 → 4 bullets.

### Phase 3 — move non-core *method* to appendix (needs appendix homes)

- **3A. §4 puff internals** (ODE, discretization, Gaussian kernel, dispersion
  covariance, exit/boundary/truncation-loss bookkeeping, PyTorch differentiability) →
  new/extended Appendix "Transport response construction". Main §4 keeps: wind→transport
  conversion (1 inline eq), a 2–3 sentence description of the open-boundary puff
  response + mass-non-increasing property, background projection concept, and the NNLS
  fit `eq:nnls_estimator`.
- **3B. §4.5 constrained refinement** (objective + constraints) → appendix. Main text:
  one sentence that an optional constrained refinement exists and is accepted only if
  it does not degrade identifiability (with acceptance criteria in appendix).
- **3C. §4.1 wind ensembles** (transport-uncertainty ensemble construction) → appendix
  (referenced from the uncertainty paragraph).
- **3D. §5.5 response-matrix perturbations** (Δ decomposition, perturbation bound
  Prop) → appendix; main §5 keeps a one-sentence statement "response error acts as
  additional observation error amplified by 1/σ_J" pointing to Appendix B.
- **3E. §5.7 per-sensor footprints** → appendix; main text keeps 2 sentences (footprints
  inherit the global resolution; identifiability is not created per-sensor).
- **3F. §6 uncertainty reporting, residual model-adequacy, wind-distribution
  diagnostics, refinement acceptance** → appendix (contracts already partly in App F).
  Main §6 keeps: inputs/outputs (prose), the 10-step procedure compressed to a tight
  list, and the one-line adequacy statement (test exists; one-sided; uncalibrated ⇒ no
  verdict). Threshold Table 2 → appendix (referenced).

### Phase 4 — move non-core *evaluation* to appendix

- **4A.** Main §8 keeps Figure 1 (Exp 1/3/4) + Figure 2 (observed) + ~1 short
  paragraph per figure + the metrics/reporting paragraph compressed to 3 sentences.
- **4B.** Move Experiments 2, 5, 6, 7, 9, 10 to Appendix "Additional experiments"
  with their (small) tables and 1–2 sentence readings each. These support but are not
  required for the central claim.
- **4C.** §7 platform: keep sensor network (32 sensors, missingness), inventories
  (Table 3, compressed), and "kernel wind, PM₂.₅ never imputed, windowed observed"
  in ~0.5 pp. Move forward-model/kriging-baseline detail and controlled-vs-observed
  mode long-form to appendix.

---

## 4. What ends up ONLY in the appendix (explicit)

New/expanded appendix sections after the references (order):
1. **Extended related work** (from §2 cuts).
2. **Transport response construction** (§4 puff internals, wind ensembles,
   refinement objective + constraints).
3. **Identifiability proofs** (existing App B) + **response-matrix perturbation** (§5.5)
   + **per-sensor footprints** (§5.7).
4. **IASA reporting internals**: uncertainty covariance, residual model-adequacy test,
   wind-distribution diagnostics, refinement-acceptance criteria, threshold table
   (§6 detail + existing App F contracts).
5. **Additional experiments**: Exp 2, 5, 6, 7, 9, 10 (tables + short readings).
6. Existing App A (notation, incl. diagnostics Table 1 + thresholds Table 2), C
   (invariants), D (gates), E (solver), G (reproducibility) — retained, lightly merged.

The 7-page main text should read end-to-end without requiring the appendix; appendix
pointers are "full derivation/…in Appendix X" only.

---

## 5. Mechanics & verification

- **Figures:** matplotlib script `paper/figures/make_figures.py` reads committed
  `evaluation/iasa_pol` results → `fig1_identifiability.pdf`, `fig2_observed.pdf`.
  Run in-container (no GPU, no experiment re-run). Commit the PDFs + script.
- **Build:** rebuild in-container via the `.vscode/texmf` recipe (per README);
  confirm main text ends at **≤ 7.0 pages** before `\bibliography`, references follow,
  appendix after. Use `\clearpage` before `\appendix` if needed to measure the split.
- **Per memory:** paper folder is editable and is the authoritative spec — keep code
  docs consistent if any claim changes (none should; this is compression only).
  Skip tests/gates (docs/paper-only work).
- **No numbers change.** Every reported value stays; only its container (table→figure,
  display→inline, main→appendix) changes. Cross-refs updated so no `??` in the build.
- **Commit** after each phase; then run an independent subagent audit (page count,
  no lost numbers, no broken refs, self-containment) and fix findings.

---

## 6. Risks

- **Figure legibility** at column width with 3 panels — mitigate with per-panel small
  multiples and a shared legend; fall back to 2 panels + move Exp 3 to appendix if
  cramped.
- **Over-cutting theory** — the two propositions + the diagnostics definitions are the
  paper's contribution and must stay in the main text even if it pushes the budget;
  trade against §4/§6 method detail first.
- **7 pages is a hard target** — if still over after Phases 0–4, the next cut is Exp 4
  from Figure 1 (→ appendix) and folding §7 platform into §8, not touching theory.
