# Plan — Make space for a 6th figure panel + caption (2026-07-25)

## Why
Advisor: "we need more results in the main text" and "too much text in the main text is
viewed as a negative." Plan: add **panel (f) = baseline comparison** (from
`baseline_comparison_2026-07-25.md`) to the consolidated results figure, plus ~2 lines of
caption and a short results paragraph, **without pushing main text past 7 pages.**

## Current state
- `paper/figures/make_figures.py` → `fig_results.pdf`: single row, **5 panels**,
  `figsize=(7.15, 1.95)`, `\includegraphics[width=\textwidth]` inside a `figure*`
  (double-column float). Caption ≈ 9 lines.
- Main text = **7 pages exactly** (discussion/conclusion on p7); `\clearpage` sends
  references to a fresh p8, appendix to a fresh p10. There is some slack at the bottom
  of p7 that can absorb part of the new height.

## Layout decision
Reshape to a **2×3 grid**: top row (a,b,c) controlled; bottom row (d,e,f) where
d,e are the observed panels and **f is the new baseline-comparison panel**. Panels stay
legible (each is larger than in the cramped 1×5). Rejected: 1×6 single row — panels get
too narrow at `\textwidth/6`.

- Figure height goes ~1.95in → ~3.7in. A double-column float of added height `Δh` removes
  `Δh` from the page and, because the body is two-column, costs ≈ `2 × Δh` worth of text
  lines. `Δh ≈ 1.75in` ⇒ reclaim ≈ **25–30 lines of main-text body** (minus the p7 slack
  already available) to stay at 7 pages. Budget for reclaiming **~30 lines**.

## Panel (f) content
Grouped bars (or a small 2×2 of IASA vs plain-NNLS vs PMF) on the S-A coherent-pair
scenario: baseline **individual coefficient error** (large) with a **tight naive CI /
"no flag"** annotation, next to IASA's merged/qualified report — the visual punchline
"accuracy ≠ identifiability." Exact encoding finalized once exp11 numbers exist.

## Space-reclamation targets (relocate > compress > cut), ~30 lines
1. **§7 "Metrics and Reporting" para** (`8.evaluation.tex`, ll.1–15) — largely
   enumerative; relocate the full list to `app:reporting_details`, keep a 2-sentence
   pointer. **~9 lines.**
2. **§7.1 "New Delhi Experimental Platform"** (`7.evaluation.tex`) — detailed and already
   has an appendix pointer; compress. **~5 lines.**
3. **§7 evaluation intro** (`7.evaluation.tex`, ll.4–15) — tighten. **~3 lines.**
4. **§6 algorithm prose** (`6.algorithm.tex`, step-order + "Model Adequacy…" run-in) —
   relocate secondary detail to `app:reporting_details`. **~5 lines.**
5. **Individual controlled-results paragraphs** (Conditioning, Background Stress, Wind
   Diversity, Missing-Source) — trim one line each. **~4 lines.**
6. **Caption**: tighten existing (a–e) wording by ~1 line to offset the +2–3 lines panel
   (f) adds.

Total reclaimable ≈ 26 lines + p7 slack ≈ enough headroom; #4/#5 are the buffer if the
first pass overshoots.

## Steps
1. *(after exp11 numbers exist)* Rewrite `make_figures.py`: `subplots(2, 3)`,
   `figsize≈(7.15, 3.7)`, move a,b,c to top / d,e,f to bottom, draw panel (f). Regenerate
   `fig_results.pdf` in-container (`source /ext3/env.sh && cd paper/figures && python3 make_figures.py`).
2. Extend the `fig:results` caption in `8.evaluation.tex` with the (f) description.
3. Apply reclamation targets #1–#5; add a short "Baselines and Identifiability"
   paragraph in §7 Controlled Results referencing panel (f).
4. Rebuild (container build.sh). **Verify:** main text ≤ 7 pages (discussion label still
   p7, references still start on their fresh page), 0 undefined/duplicate refs.
5. If over 7 pages, apply buffer trims (#4/#5) or move one more controlled paragraph's
   detail to the appendix.

## Dependency & ordering
- Panel (f) needs exp11 output ⇒ run `baseline_comparison_2026-07-25.md` **first**.
- Reclamation (#1–#5) is independent and can be done in parallel/first to bank the space.
- Keep the `figure*` double-column float and `width=\textwidth`; do **not** use `\vspace`
  or shrink AAAI template spacing to gain room (disallowed) — all savings come from
  content relocation.
