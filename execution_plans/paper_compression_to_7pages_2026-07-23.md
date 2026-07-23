# Aggressive compression plan: 7-page main text (2026-07-23)

## Goal & assumptions
- **Target:** main text (§1–§8, incl. all in-text figures/tables) fits in **7 pages**;
  references and then the appendix follow. References do **not** count toward the 7.
- **Current:** body ends ~p11 (refs start ~p11); appendix A at p12; total 21 pp.
  Body ≈ **10 pages** ⇒ must remove **~3 pages (~30%)**.
- **Mode:** aggressive. Two kinds of edit, tagged per item below:
  - **[RELOCATE]** = move verbatim/near-verbatim to appendix → lossless, recoverable.
  - **[CUT]** = genuine deletion / hard compression → some detail is lost. These are
    flagged explicitly so they can be vetoed.
- Two-column AAAI ≈ ~1000–1100 words of body per page (less where equations/floats sit).

## Where the 3 pages come from (priority order)

### Phase A — §5 Theory: 2.3 pg → ~1.25 pg (save ~1.0 pg) — biggest lever
Keep the **theoretical core** in main text, push definitional machinery to the appendix.
- **KEEP (compact):** Def. exact identifiability; Prop. rank(H̃_Φ)=J (statement only,
  proof already in app); Prop. noise-robust σ_J bound + Eq. (nnls_noise_bound)
  (statement only, proof already in app). These are contribution #2 and must stay.
- **[RELOCATE] §5.3 "Identifiability Diagnostics" (lines 86–162).** This is ~1 full
  column of definitions (τ_num, padded σ_J, κ, σ_min+, effective rank τ_σ, visibility
  & weak set, coherence, ambiguous-pair set, ray distance + its derivation). Move the
  full definitions to a new appendix subsection "Identifiability Diagnostics
  (definitions)". Leave in main text **one compact paragraph** naming the diagnostics
  with one-line intuition each (σ_J = worst-case robustness; effective rank at the
  noise threshold; visibility = post-projection fingerprint magnitude; coherence/ray
  distance = pairwise separability; absorption = signal lost to background) + pointer
  to the appendix and to Table (tab:diagnostics, already in app).
- **[RELOCATE] Eq. (ray_distance_coherence_relation) + its paragraph (145–162).** Ray
  distance is explicitly "geometrically equivalent to coherence, not independent" — the
  equivalence identity and the signed-ray discussion go to the appendix; main text keeps
  one clause ("a scale-invariant ray distance equivalent to coherence").
- **[RELOCATE] "Background absorption" paragraph (164–175).** One sentence stays
  (absorption fraction a_j flags signal lost to Q; why identifiability is assessed after
  projection); the P_Q definition and N/A bookkeeping go to app.
- **[RELOCATE] "Response-matrix perturbations" paragraph (177–192).** Already has
  app:operator_error_details. Reduce main text to one sentence (transport vs inventory
  error both act like observation error, worst when H̃_Φ is ill-conditioned; transport →
  intervals, inventory → named scenarios) and move the Δ_H decomposition + inflated bound
  to that appendix section.
- **§5.4 "Identifiable Source Resolution" (194–224):** KEEP (conservative merging is a
  contribution) but tighten — **[CUT]** the transitive over-merging worked example
  ({A,B,C}) to one clause; keep the ambiguity-graph definition and the group-activity
  sums. Save ~0.2 pg.
- **[RELOCATE] "Per-sensor footprints" paragraph (226–242).** Secondary (spatial overlay
  of the global fit). Compress to 2 sentences (footprints = response rows read backward;
  identifiability inherited, σ_J^(s) ≤ σ_J) + pointer to app:footprint_details; move the
  inequality statement and detail to appendix.

### Phase B — §6 Algorithm: 1 pg → ~0.6 pg (save ~0.4 pg)
The Algorithm float is the value; the surrounding prose largely restates it.
- **[CUT] §6.1 "Inputs and Outputs" (10–31).** The float already has full Input/Output
  blocks. Delete the two prose paragraphs that duplicate them; keep only the one sentence
  pointing thresholds to Table (tab:thresholds) in the appendix.
- **[RELOCATE] FISTA solver paragraph (101–110).** Implementation detail (projected
  FISTA, Lipschitz step, KKT residual, batching) → appendix. Keep one clause: "solved by
  an accelerated projected-gradient method."
- **§6.3 "Model Adequacy, Uncertainty, Reporting" (112–148):** **[RELOCATE]** most to
  app:reporting_details (already exists). Keep a compact paragraph: uncertainty from
  active-set/ridge covariance + transport ensembles (transport vs inventory never
  pooled); one-sided calibrated residual test (rejection ⇒ mismatch; non-rejection ≠
  completeness); uncalibrated fallback. Drop the per-group reliability-label enumeration
  (it repeats §5.4).

### Phase C — §7 Evaluation: 3 pg → ~1.75 pg (save ~1.25 pg) — second lever
- **[RELOCATE] §7.1 "New Delhi Experimental Platform" (949 w, five \paragraph blocks).**
  app:platform_details already holds the deep version. Collapse the five paragraphs
  (government obs, wind prep, inventories, forward models, study modes) to **one dense
  paragraph**: 32 regulatory PM2.5 sensors, hourly 2018–2020 govt records, four proxy
  source groups (brick kilns, industry, population, traffic) with declared temporal
  bases, kernel-imputed gridded wind, open-boundary puff forward model + auxiliary
  advection–diffusion mismatch, controlled vs observed modes. Push column names, coverage
  %, Pusa averaging, crop windows, kriged-baseline/simulator settings to the appendix
  (most already there). Save ~0.8 pg.
- **§7.2 "Metrics and Reporting" (1–21):** **[CUT]** the exhaustive metric enumeration to
  ~half — name the metric families (recovery/coverage; singular spectrum & derived
  diagnostics; provenance) and point to appendix for the full list. Save ~0.3 pg.
- **§7.3 Controlled Results (4 \paragraph experiments):** figures already carry the
  numbers. **[CUT]** the in-prose re-recitation of values duplicated in the Fig. 1
  caption; keep the setup sentence + the interpretive claim per experiment. Save ~0.3 pg.
- **§7.4 Observed Results (3 \paragraph):** same tightening; the per-week numbers live in
  Fig. 2. Save ~0.2 pg.

### Phase D — §3 Setup + §4 Method + polish: save ~0.4 pg
- **§3 [CUT/RELOCATE]:** tighten the temporal-dictionary discussion (44–58) and the
  lag-window η_L prose (84–102, keep Eq., trim justification → app). Save ~0.3 pg.
- **§4 [CUT]:** trim wind-imputer and background-construction prose (detail already
  points to app:transport_response). Save ~0.15 pg.
- **§8 Discussion [CUT]:** shorten the limitations run-on (one long sentence) by ~2
  lines. Intro/§2 left as-is (already tight).

## Projected 7-page budget
| Block | Now | Target |
|---|---|---|
| §1 Intro + §2 Related | 1.25 | 1.0 |
| §3 Setup | 1.25 | 1.0 |
| §4 Method | 0.75 | 0.6 |
| §5 Theory | 2.3 | 1.25 |
| §6 Algorithm (+ float) | 1.0 | 0.65 |
| §7 Evaluation (+ 2 figs) | 3.0 | 1.75 |
| §8 Discussion | 0.7 | 0.5 |
| **Total body** | **~10** | **~6.75** |

Leaves ~0.25 pg slack against the 7-page line. If it overshoots, the next cuts are:
Fig. 1/Fig. 2 to single-column (currently figure*), or fold §6.3 entirely into the
algorithm caption.

## Appendix impact
All [RELOCATE] items land in existing or new appendix subsections
(app:identifiability_proofs neighbourhood gets a new "Diagnostics (definitions)";
app:operator_error_details, app:footprint_details, app:reporting_details,
app:platform_details already exist and absorb the rest). Appendix grows ~1.5–2 pp; total
page count roughly unchanged, but the **7-page main text** target is met.

## Validation
- Per phase: rebuild in-container, confirm 0 undefined/duplicate refs, capture the body
  end-page (target: §8 ends ≤ p7, refs begin p7/8).
- **Relocation check (lossless):** an independent validator confirms every [RELOCATE]
  item's content is present in the appendix (byte/number-level for tables & equations).
- **Cut disclosure (lossy):** the validator lists what each [CUT] removed that is *not*
  recoverable anywhere, so the loss is explicit and signed off — the core claims,
  propositions, algorithm, and headline results must remain fully in the main text.

## Suggested execution order
A (theory) → C (evaluation) → B (algorithm) → D (setup/polish), rebuilding after each and
checking the body end-page. A and C alone should recover ~2.25 pg; B+D close the rest.
