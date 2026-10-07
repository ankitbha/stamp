# Plan — AISTATS 2027 applications paper: identifiability-aware PM2.5 source attribution (2026-10-06)

> **Status (2026-10-06, later): D1–D7 taken as recommended; full draft written.** Title
> "Identifiability-Aware Source Attribution of Urban PM2.5 from Sparse Regulatory Sensors".
> Main text ends at 8.00 pages (measured with the end-of-main-text marker); clean build,
> 0 undefined references. New in the main text: Section 3 data and transport model with a
> platform figure (proxy maps + 32 sensor cells) and an inventory table; Section 4.3 IASA
> procedure with Algorithm 1 and the pairwise merge rule; Section 5 with the baseline table
> (full width), a sensor-layout × wind table (Experiment 4), missing-source and transport-error
> paragraphs; Section 6 observed weeks with Figure 3 (σ_J and max coherence; group-signal norms
> against √7·σ_e = 81.2) and a shares table with intervals at 30.7 and at the AR(1) upper level
> 110.7. Two read-only verifiers (math/numbers/code claims; reader-level review) are checking
> the draft. Nothing is committed.
>
> **Restructure (2026-10-06, night; Ankit's request).** Order is now Introduction, Related
> Work, Problem Setup (data, maps + Figure 1, transport, model, the attribution problem with
> requirements R1 identifiability and R2 reportability), Method (IASA, Algorithm 1),
> Theory (Assumption 5.1 declared model, 5.2 Gaussian noise; Props 5.3–5.4, Thm 5.7
> groupings, Thm 5.9 group-signal error with a new part (c): uniform high-probability bound
> σ(√rank Ã + √(2 log 1/δ))/s_g for LS and NNLS, and the N(0,Σ) version; reportable
> partitions; stability under operator error; each result has an "Implication" remark tied
> to New Delhi and to an Algorithm 1 step), Experiments (6.1 controlled, 6.2 observed),
> Conclusion. The archived Section 4.3 is back. IASA's grouping is now the separation rule
> at τ_θ = 0.141 (equal to coherence 0.99 for single columns); on every saved operator and
> observed week it gives the same partition as the recorded coherence rule. The observed
> weeks now use the Theorem 5.9(c) bound as the criterion: b_g/‖signal‖ for the largest
> group is 0.49, 0.24, 2.96, 1.33 at σ_e = 30.7 (1.78, 0.88, 10.67, 4.81 at 110.7).
> Figure 2 keeps four panels in the main text; four moved to an appendix figure. Main text
> 8.00 pages. Open: the inventory crop is a fixed index window whose alignment with the
> sensor bounding box is not verified in the code (stated in Appendix D.6).
>
> **Verification (2026-10-06, evening).** Two read-only verifiers (math/numbers/code; reader)
> reported 55 findings; every main-text number they checked matched the result files. Confirmed
> and fixed, among others: L = 12 h was declared in every run (the 10^-3 lag rule never selected
> L; Experiment 7's grid did not converge); Q is rank 4 (constant, linear, daily sin/cos), no
> sensor offsets; τ_v = 0 in every run; the flag thresholds (visibility ratio ≤ 0.1, absorption
> ≥ 0.9, σ_J ≤ 1e-6, coherence ≥ 0.99) are now stated; controlled experiments mostly use
> synthetic emitters and winds (now stated); the proportional pair is the six-sensor
> steady-wind case; shares are concentration-signal fractions (µg/m³), not proxy units; plain
> NNLS matches IASA's errors (now stated); "determined in two of four weeks" replaced by
> interval facts; East Arjun Nagar has no readings (coverage over 31 locations: 93.4/77.3/75.1%);
> the Exp 4 downwind coherences are 0.9897 and 0.9901, on either side of τ_ρ. A third verifier
> checks the fixes.
>
> Facts found while writing: the recorded Experiment 4 run has 18 rows, the appendix table had
> 15 (three downwind rows added); the regulatory layout has 32 sensors and the random and
> downwind layouts 8; no run declares τ_σ, so r_eff equals the numerical rank; the projected
> residual is 46–91 µg/m³ RMS per reading against 1.05–13.24 for the total fitted signal; the
> 06:00 traffic map in the inventory is all zeros.

## Short version

- **Target.** The AISTATS 2027 call lists "Applications of machine learning and statistics
  (… sustainability and climate …)". The paper becomes an application of statistics to urban
  PM2.5 source attribution from sparse regulatory sensors, not a general theory paper.
- **What stays as theory.** Only results the authors can check and that an independent verifier
  marked correct: Proposition 4.1 (exact identifiability), Proposition 4.2 (noise robustness),
  Theorem 4.4 (identifiable groupings, (a)–(e)) and Proposition D.1 (operator perturbation).
- **What was archived.** Sections 4.3–4.5 (separation, reportable partitions, stability, uncertain
  fingerprints, the IASA procedure section), Section 5 (synthetic study and statistical
  baselines), their proofs, the synthetic appendix, and the Appendix B passages tied to them, are
  in `paper_aistats/sections/archive.tex`, which `main.tex` does not input.
- **Current state.** The paper builds with no undefined references (30 pages). The main text ends
  at 4.77 pages, so about 3.2 pages (388 column-lines) are available.
- **How to reach 8 pages.** Add application content that already exists in the appendices or the
  archive: the data and transport model (from Appendix C), a rewritten procedure section with an
  algorithm box (from the archive, without the archived theory), the full observed-weeks analysis
  (from Appendix G.9), and a sensor-network design result (Experiment 4).
- **Decisions needed before writing (D1–D7 below).** The main one is D1: which rule merges sources
  at the noise level now that the separation rule's theory is archived.
- **Deadline.** Full paper due today, 6 October 2026, 23:59 AoE. Estimated work after the
  decisions: about 5 hours.

## Scope boundary

- No new theory and no new statements that are not checked by the independent verifier.
- No pipeline run longer than a few minutes; every number comes from committed results
  (`evaluation/iasa_pol/runs`, `runs_signal_target`, `signal_target/weeks.json`).
- The archived material stays in `archive.tex` and is not cited.

## What changed today to keep the build consistent (already done)

| Place | Change |
|---|---|
| Abstract | Removed the error-bound and synthetic-experiment sentences. |
| Introduction | Removed the contribution bullet on group-error bounds, noise-level groupings and latent groupings; the IASA bullet now points to Section 4 and Appendix B. |
| Related work (Section 2, Appendix I) | Removed principal-angle and oblique-projector sentences and the archived theorem references. |
| Section 5 (was 6), controlled | Separation defined in one sentence at first use; the groupings paragraph now states the pairwise rule at τ_ρ = 0.99 and its agreement with 𝒫⋆ ∨ 𝒟 for Experiments 3 and 9. |
| Section 5 (was 6), observed | Removed the Corollary-based error-bound sentence; the summary keeps the fitted signal size and the bootstrap intervals. Figure 3 caption defines the dashed line as √(1−0.99²). |
| Conclusion | Removed claims about principal-angle bounds, noise-level non-uniqueness and the synthetic study. |
| Appendix B | Reporting protocol now: start from 𝒫⋆ ∨ 𝒟, merge blocks with a column pair of coherence above τ_ρ. τ_θ and separation rows removed; τ_ρ is the merge threshold. |
| Appendix G.9 and Table 17 | Bound columns and bound sentences removed; separation, fitted signal, share and bootstrap interval kept. |
| Appendix F, checklist, AI statement | References to the algorithm box, synthetic study and archived results removed. |
| Bibliography | One `refs.bib`; 51 entries print. No longer printed: beck2009fast, bjorck1973numerical, boucheron2013concentration, efron1993introduction, golub1979generalized, hansen1987truncated, hoerl1970ridge, ipsen1995angle, vershynin2012introduction. |

## Decisions (each with a recommendation)

| # | Question | Options | Recommendation |
|---|---|---|---|
| D1 | Rule that merges sources at the noise level | (a) pairwise coherence above τ_ρ = 0.99, starting from 𝒫⋆ ∨ 𝒟; (b) 𝒫⋆ ∨ 𝒟 only (exact dependencies); (c) the archived separation rule as a heuristic without its theorem | **(a).** The recorded controlled experiments used it, it needs no archived theory, and Theorem 4.4 supplies the exact part. State its limitation: it misses a near-dependency among three or more columns with no high pairwise coherence. The observed weeks give the same result under (a) and under the separation rule recorded in `weeks.json` (τ_θ = 0.141): in all four weeks the exact components are the seven singletons, the reported grouping equals the declared partition {0},{1},{2},{3,4,5,6}, and the maximum coherence (0.32–0.50) is below 0.99, so (a) merges nothing. |
| D2 | Figure 3(a) for the observed weeks | (a) keep the per-group separation as a descriptive diagnostic (one-sentence definition, no bound); (b) replace with σ_J (from the singular values recorded in `weeks.json`) and the maximum pairwise coherence per week | **(b)** if D1 = (a), because coherence is then the quantity the rule uses; (a) needs no change to the figure if time is short. |
| D3 | Noise statement for the observed shares | (a) parametric-bootstrap intervals only; (b) also restore the analytic bound | **(a).** The bound rests on archived theory. |
| D4 | Theorem 4.4(e) (coefficient sums, NP-completeness) | keep in the main text; move to the appendix with one sentence on units | **Keep.** It is verified, short, and justifies reporting group signals. |
| D5 | Title | e.g. "Which Sources Can Sparse Sensors Separate? Identifiability-Aware Attribution of Urban PM2.5"; "Identifiability-Aware Source Attribution of Urban PM2.5 from Sparse Regulatory Sensors" | Second option: it names the application and the method. |
| D6 | Related work emphasis | keep the matroid and subspace-clustering paragraph; reduce it to one sentence and expand air-quality source apportionment and atmospheric inversion | **Reduce and expand.** Applications reviewers judge the positioning against receptor models (CMB, PMF) and inverse modelling. |
| D7 | Which application content fills the 3.2 pages | (i) data, inventories, wind and transport model; (ii) procedure section with algorithm box; (iii) full observed-weeks analysis; (iv) sensor-network design from Experiment 4 (regulatory vs random vs downwind layouts, six wind providers); (v) per-sensor footprints (Experiment 10) | **(i)–(iv).** (v) only if space remains. |

## Target structure and page budget (8.0 pages before the AI statement)

| Section | Pages | Source of the text |
|---|---|---|
| 1 Introduction (application framing, contributions) | 1.0 | rewrite of current Section 1 |
| 2 Related work (source apportionment, atmospheric inversion, estimability) | 0.6 | current Section 2 and Appendix I |
| 3 Data and transport model (network, inventories, wind field, lagged puff response, nuisance basis, projection) | 1.0 | current Section 3 and Appendix C |
| 4 Identifiability and the IASA procedure (Props 4.1–4.2, Theorem 4.4, group signals, diagnostics, merge rule per D1, algorithm box, uncertainty and adequacy) | 1.6 | current Section 4, archived Section 4.5 without the separation and latent-grouping parts |
| 5 Controlled experiments (full Figure 2, baselines, conditioning, sensor-network design, absorption, operator error, lag, temporal basis, inventory) | 2.0 | current Section 5.1 |
| 6 Observed New Delhi weeks (data, noise level, identifiable resolution, shares with bootstrap intervals, Figure 3, week-by-week table) | 1.3 | current Section 5.2 and Appendix G.9 |
| 7 Discussion and limitations | 0.5 | current Section 6 |

## Work items (after the decisions)

1. **Procedure section (45 min).** Restore Diagnostics, the algorithm box and Uncertainty from
   `archive.tex`; replace step 5 with the D1 rule; remove every mention of τ_θ, separation bounds and
   latent grouping; restore the FISTA citation (beck2009fast) if the solver is named.
2. **Data and transport model (45 min).** Move the network, inventory, wind-field and lagged-response
   description from Appendix C into Section 3; leave derivations in Appendix C.
3. **Observed weeks (30 min).** Move Appendix G.9 back into Section 6 with bootstrap-only noise
   statements (D3) and Table 17 in the main text; make Figure 3(a) per D2.
4. **Sensor-network design (20 min).** Turn the Experiment 4 bullet into a short subsection with the
   wind × layout table summary (σ_J and coherence by layout).
5. **Framing (60 min).** Title (D5), abstract, introduction and conclusion for the applications track:
   the problem, why receptor models over-report, what IASA reports, what the New Delhi data support.
6. **Related work (30 min)** per D6.
7. **Length (30 min).** Measure with the end-of-main-text marker; trim or add to reach 8.0 pages;
   the layout agent re-checks the formatting rule and self-containedness.
8. **Verification (30 min).** The proof verifier re-checks any sentence that states a mathematical
   fact about the procedure; numbers are re-checked against the result files.
9. **Submission items.** Checklist answers, AI statement confirmation, anonymized code archive.

## Risks

- **No ground truth on the observed weeks.** Applications reviewers will ask how the attribution is
  validated. The controlled experiments use the real network, wind and inventories with known
  contributions; the observed analysis reports only what the noise level supports. External
  validation (lockdown periods, co-located speciation) is not possible by today.
- **The pairwise merge rule is a heuristic.** It is the rule the recorded experiments used; the paper
  must state that Theorem 4.4 covers only exact dependencies and that near-dependencies among three or
  more columns are not detected by pairs.
- **Time.** About 5 hours of work after the decisions, on the deadline day.
