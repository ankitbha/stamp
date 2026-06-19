# Task 5 Completion Plan: Address Five Verification Findings

## Summary

Complete the Gaussian puff response implementation by enforcing physical mass bounds, expanding Gate S1 to the complete roadmap experiment matrix, making diagnostic truncation explicit, filling test gaps, and synchronizing the roadmap with the four-source sanity configuration. Preserve the existing `H_lag` API and metadata keys; all changes are additive except for correcting retained-mass calculations.

## Public Contracts

- Add `ResponseConfig.max_kernel_diagnostic_records: int | None = 500`; `None` stores every record and a nonnegative integer stores a deterministic prefix.
- Add metadata fields:
  - `kernel_emitted_mass_by_column`
  - `kernel_observation_count_by_column`
  - `kernel_diagnostic_total_count`
  - `kernel_diagnostic_stored_count`
  - `kernel_diagnostics_truncated`
  - `kernel_quadrature_clip_count_by_column`
  - `max_raw_retained_fraction_by_column`
- Preserve existing metadata keys and duplicate `row_index`, `column_index`, and `baseline` in `ResponseMatrixResult`.
- Export `WindSampler` from `model.iasa` alongside `CityWindSampler`.

## Implementation Steps

1. **Enforce mass conservation**
   - Treat `_retained_kernel_mass()` output as a raw retained fraction.
   - Compute `retained_fraction = clip(raw_retained_fraction, 0, 1)`.
   - Leave values below one unchanged so real boundary and truncation loss is never renormalized.
   - Compute `retained_mass = release_mass * retained_fraction` and `dropped_mass = release_mass - retained_mass`.
   - Accumulate release mass once per evaluated kernel into `kernel_emitted_mass_by_column`.
   - Track quadrature clipping count and maximum raw retained fraction.
   - Require `max(abs(retained + dropped - emitted)) <= 1e-5` in Gate S1.

2. **Make diagnostic truncation explicit**
   - Increment total diagnostic count for every evaluated release kernel.
   - Store detailed records only below `max_kernel_diagnostic_records`, preserving deterministic loop order.
   - Record stored count and truncation state.
   - Keep complete per-column emitted, retained, dropped, exit, and kernel-count aggregates even when detailed records are truncated.
   - Validate that the configured limit is `None` or nonnegative.

3. **Implement the full Gate S1 experiment matrix**
   - Keep the eastward case and four sources, including required `interior_source`.
   - Add constant northward wind and require the south-source impulse to reach the north sensor more strongly than the south sensor at positive ages.
   - Add a two-direction sequence: eastward during timesteps `0..29`, northward during `30..59`. Compare a constant-basis source column to the eastward-only fingerprint and require a maximum difference above `1e-6`.
   - Retain matched east-edge/interior mass comparison, positive dropped mass, and no opposite-edge signal.
   - Build a dedicated single-cell interior source and dense `16x16` observer for dispersion moments.
   - Require:
     - anisotropic along/cross moment ratio above `1.25`;
     - increasing `sigma_parallel` primarily increases the along-wind moment;
     - increasing `sigma_perp` increases the crosswind moment;
     - isotropic moments agree within `0.25` relative difference.
   - Include every metric and threshold in Gate S1 JSON output.

4. **Expand unit coverage**
   - Test a narrow same-time kernel to prove retained mass never exceeds emitted mass and retained plus dropped equals emitted within `1e-5`.
   - Test zero source and zero basis, requiring zero `H_lag` and diagnostics.
   - Test a recording sampler with `dt=1` and `substep_dt=0.3` to verify the fractional final step without time overshoot.
   - Test invalid dispersion, lag window, source shape, nonfinite basis/wind, and diagnostic limits.
   - Test diagnostic counts, truncation state, complete aggregate lengths, JSON serialization, and the package-level `WindSampler` export.

5. **Synchronize the roadmap**
   - Change the Task 5 sanity source count from three to four.
   - Make `interior_source` required as the matched boundary-loss control.
   - Document bounded retained fractions, per-kernel emitted mass, and the conservation equation.
   - Document capped diagnostic samples and required count/truncation metadata.
   - Keep northward, two-direction, anisotropy, and conservation checks mandatory for Gate S1.

## Test Plan

- Run syntax compilation, response tests, and the complete pollution/activity/wind/response suite in the repository overlay.
- Run `scripts/run_iasa_sanity.py --gate response`, `--gate all`, and `scripts/smoke_iasa_runtime.py`.
- Require all tests to pass, `git diff --check` to be clean, Gate S1 to report all wind/dispersion metrics, and no conservation error above `1e-5`.

## Assumptions

- Deterministic quadrature remains acceptable for Task 5; clipping only corrects numerical overshoot and never restores genuinely lost mass.
- Detailed records remain capped by default while complete aggregates are never truncated.
- Sanity experiments run on CPU in the existing overlay without Slurm or New Delhi data.
- Projection, diagnostics, fitting, and merge behavior remain outside Task 5.
