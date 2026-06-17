# Roadmap for Implementing `STAMP_new.pdf`

## 1. Paper Delta

### Old paper: limits of sparse-sensing inverse problems

`STAMP_old.pdf` frames the project around fundamental limits. The central claims are:

- Sparse sensor observations induce a low-dimensional observation space for a high-dimensional physical state.
- Under finite horizons, multiple source configurations can produce indistinguishable sensor trajectories.
- Distinct PDE models can collapse in observation space; in particular, advection-diffusion can become observationally close to pure advection under sparse and spatially isolated sensing.
- The main empirical goal is to demonstrate non-identifiability and model indistinguishability, even when the forward simulator and optimizer are accurate.

The old codebase reflects this framing. The pollution path simulates advection-diffusion dynamics with an aggregate known source plus a learnable coarse unknown source field. Calibration is done by optimizing that field through the differentiable simulator and then evaluating whether the inferred field matches ground truth.

### New paper: identifiability-aware source apportionment

`STAMP_new.pdf` changes the problem from arbitrary source recovery to inventory-based source apportionment. The new target is not to recover a free spatial source field. Instead, the system assumes a finite set of known or proxy source maps and estimates nonnegative source activities over those groups. The practical target also includes a real New Delhi apportionment experiment that uses observed `pm25`, imputed government wind observations, named source inventories, and time-varying activity patterns.

The new conceptual object is the projected lagged source-response matrix:

```text
H_tilde = P_Q_perp H_lag
```

where:

- `S = [s_1, ..., s_K]` is a matrix of candidate source maps.
- `theta >= 0` is a source activity vector; for the New Delhi experiment this generalizes to nonnegative time-varying activity coefficients over low-dimensional temporal bases.
- `H_lag` stacks wind-conditioned, finite-lag sensor-space fingerprints for each source group.
- `Q` is a low-dimensional background or temporal basis.
- `P_Q_perp` projects away background components.
- `H_tilde` determines which source groups can be separated after transport, sparse sensing, lag, and background correction.

The goal becomes: estimate source activities and report the finest source grouping that is defensible under the sensor layout, wind regime, lag window, background basis, and noise level. For the real New Delhi case, the roadmap must also produce actual apportionment numbers, uncertainty intervals, and merge recommendations.

### Required new capabilities

The implementation needs these new capabilities:

- Separate named source inventories instead of one aggregate `S_known`.
- Government `WD`/`WS` ingestion, missing-data imputation, and conversion to simulator-ready wind vectors.
- A wind-provider interface that supports real imputed New Delhi wind and controlled synthetic wind regimes.
- Source activity parameterization over inventory groups and temporal activity bases.
- Construction of `H_lag` from a dedicated open-boundary source-response operator under real or synthetic wind sequences.
- Background basis construction and projection.
- Identifiability diagnostics: rank, singular values, condition number, effective rank, visibility, background absorption, pairwise coherence, ray distance where feasible, and perturbation sensitivity.
- Nonnegative source-activity fitting.
- Uncertainty intervals and ambiguity reporting.
- Merge recommendations for indistinguishable source groups.
- Controlled experiments matching the new paper hypotheses, including the claim that wind diversity improves source separability.
- Real New Delhi apportionment reporting from observed `pm25`.

## 2. Current Codebase Map

### Pollution simulator

`sim/polsim.py` is the main differentiable pollution simulator. It currently:

- Loads seven pollution intensity maps.
- Aggregates them into one normalized `S_known` field:
  - brick kilns
  - industries
  - population density
  - traffic at 00, 06, 12, and 18 hours, averaged into one traffic field
- Crops the 80x80 source maps to the 40x40 simulation domain.
- Builds an initial condition from government sensor data through kriging.
- Generates a monsoon-like synthetic wind series.
- Evolves a 2D advection-diffusion-source PDE.
- Accepts `S_unknown` as a 10x10 coarse field, smooths and upsamples it to the simulator grid, and adds it to `S_known`.

The simulator currently ignores the `WD` and `WS` meteorological fields already present in `sim/govdata_1H_current.csv`. It must be extended to accept actual wind sequences, not only internally generated synthetic winds.

The current simulator uses an edge-hold boundary condition. This is useful for legacy rollouts, but it is not the paper-faithful open-boundary response operator used to define `H_lag`. The IASA response-matrix path should therefore use a dedicated open-boundary response implementation, or explicitly label any PDE rollout response as `edge_hold_pde` until an open-boundary PDE mode exists.

There are also `sim/polsim_adv_only.py` and `sim/polsim_diff_only.py` variants for ablation-style model comparisons. These match the old paper more than the new one.

### Government weather data

`sim/govdata_1H_current.csv` contains hourly government station observations with columns:

```text
monitor_id, timestamp_round, AT, RH, WD, WS, pm10, pm25
```

The local file has 32 stations and 21,960 timestamps. `WD` and `WS` have substantial missingness, so real New Delhi apportionment needs an imputation stage before the wind can drive the transport model. The roadmap should treat this CSV as the authoritative local source for both observed `pm25` and weather fields.

There are references to FieldFormer-style data generation in the repository, but no local ImputeFormer implementation is currently visible in the tracked code paths. The implementation should therefore either locate the existing FieldFormer/ImputeFormer baseline used by the project or vendor/adapt it into a documented local path before claiming full New Delhi wind-imputation support.

### Data generation

`data/poldata.py` currently generates a dataset by:

- Creating a 40x40 pollution grid.
- Loading the aggregate known source.
- Generating one synthetic 10x10 unknown source.
- Scaling the unknown source relative to the aggregate known source.
- Running the simulator.
- Sampling fixed government sensor locations.
- Saving `sensor_clean`, `sensor_noisy`, `S_known`, `S_unknown_coarse`, and `S_unknown_fine`.

This dataset supports free-field source recovery, not source apportionment over named inventories.

### Calibration

`model/calibrator/tuner_pol_simgrad.py` currently:

- Loads the pollution dataset.
- Builds the simulator and sensor observer.
- Creates a learnable `S_unknown` field.
- Optimizes the unknown field through the differentiable simulator.
- Optionally adds a frozen MPRNN/STAMP dynamics loss.
- Saves best calibration outputs.

This is a gradient-through-simulator inverse solver. It is not the linear or constrained source-activity fitting pipeline described in the new paper.

### Evaluation

`evaluation/eval_pol.py` currently:

- Loads a predicted unknown source field.
- Runs the simulator using that predicted field.
- Compares predicted and observed sensor trajectories.
- Computes spatial error and correlation metrics against `S_unknown_coarse`.

The new paper needs evaluation over source activities, merged groups, response-matrix diagnostics, and identifiability predictions.

### Prior model

`model/prior/*` implements MPRNN/STAMP prior training and graph utilities. This machinery is useful for old STAMP-style dynamics regularization, but it is mostly orthogonal to the new IASA formulation. It can remain as a legacy or optional comparison path.

### Main mismatch

The current codebase optimizes a spatial unknown source field using synthetic wind. The new paper requires estimating nonnegative activity coefficients over named inventories, using real or controlled wind sequences, and auditing whether those source groups are identifiable. The implementation should therefore add an IASA path rather than trying to force the existing `S_unknown` calibrator to serve as the main method.

## 3. Sequential Implementation Tasks

Each task below is intended to be implemented in order. Later tasks should not assume future infrastructure exists.

### Task 1: Runtime and dependency baseline

**Objective**

Make the expected runtime environment explicit and add minimal smoke checks for the pollution workflow.

**Likely files**

- `README.md`
- Optional: `scripts/smoke_iasa_runtime.py`
- Optional: `requirements.txt` or `environment.yml` if the project should support host Python later

**Implementation details**

- Document that the repository is currently expected to run inside the existing Singularity image:

```bash
/share/apps/apptainer/bin/singularity exec cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif python3 ...
```

- Confirm availability of required packages inside the image: `numpy`, `torch`, `pandas`, `pykrige`, and optionally `scipy`.
- Add a smoke script that imports the pollution simulator, loads source maps, creates a grid, and prints source and sensor shapes without running a long simulation.
- Keep host Python limitations explicit; in the current environment, host Python does not provide `numpy`.

**Outputs and artifacts**

- Runtime instructions in `README.md`.
- A quick smoke command that future tasks can use before deeper tests.

**Acceptance checks**

- Smoke command runs in the Singularity image.
- It imports `sim.polsim`, builds a 40x40 grid, loads source maps, and exits successfully.
- No tracked data artifacts are rewritten by the smoke check.

### Task 2: Weather data and wind imputation

**Objective**

Build the real New Delhi wind pipeline from government `WD` and `WS` observations, including missing-data imputation and conversion to simulator-ready wind vectors.

**Likely files**

- New `data/pol_weather.py` or `model/iasa/wind.py`
- Optional: `baselines/imputeformer/` if the FieldFormer/ImputeFormer implementation is vendored into this repo
- Optional: `scripts/impute_new_delhi_wind.py`
- Optional: `scripts/smoke_iasa_runtime.py`

**Implementation details**

- Load `WD` and `WS` from `sim/govdata_1H_current.csv`, preserving:
  - station ids
  - hourly timestamps
  - missingness masks
  - units and source column names
  - alignment with `pm25` observations
- Treat `sim/govdata_1H_current.csv` as the authoritative local New Delhi weather and air-quality source.
- Use the FieldFormer/ImputeFormer baseline implementation for wind imputation. If that implementation is not present in this repository, explicitly add or adapt it under a documented path such as `baselines/imputeformer/` or `model/imputation/`.
- Impute `WD` and `WS` over the selected experiment window before simulator use. Preserve both raw and imputed arrays for auditability.
- Convert imputed meteorological direction and speed into simulator coordinates:

```text
WD/WS -> Vx(t), Vy(t)
```

- Make the meteorological direction convention explicit. `WD` is normally the direction wind comes from; the simulator needs the direction pollutants are transported toward.
- Save imputed wind products with metadata:

```text
timestamps
station_ids
raw_WD
raw_WS
imputed_WD
imputed_WS
missingness_masks
Vx
Vy
imputation_config
direction_conversion_convention
```

- Support both station-level wind fields and a city-level aggregate wind sequence. The v1 New Delhi apportionment path should use a city-level sequence unless spatially varying wind is implemented.

**Outputs and artifacts**

- A reusable wind preprocessing API.
- Saved imputed wind product for New Delhi experiments.
- A smoke script that verifies the imputed wind sequence has no missing values over the selected window.

**Acceptance checks**

- Loader reports 32 stations and the expected timestamp range for the local government CSV.
- Raw `WD`/`WS` missingness masks are preserved.
- Imputed `WD`/`WS` have no missing values over the selected experiment window.
- Cardinal-direction conversion tests pass for known synthetic `WD`/`WS` cases.
- `Vx` and `Vy` are aligned to the same timestamps used for `pm25` observations.

### Task 3: Inventory source refactor

**Objective**

Expose separate named source inventories without directly aggregating them into a
single `S_known` field. The proxy maps should remain distinct because their raw
scales are not reliably comparable across source categories.

**Likely files**

- `sim/polsim.py`
- `data/poldata.py`
- Optional: new `sim/pol_sources.py`

**Implementation details**

- Add a loader that returns individual cropped source maps with source-specific
  normalization or scale metadata:

```text
brick_kilns
industries
population_density
traffic_00
traffic_06
traffic_12
traffic_18
```

- Do not sum source categories into one aggregate source in the new IASA path.
- Preserve the four time-of-day traffic maps (`traffic_00`, `traffic_06`, `traffic_12`, `traffic_18`) for the traffic diurnal activity basis.
- Preserve the existing crop `(21:61, 16:56)`.
- Do not apply the old shared aggregate percentile-99 normalization to the
  combined source field. Instead, keep per-source normalization choices explicit
  and record the scale used for each source map in metadata.
- Return:

```text
source_names: list[str]
source_maps: [K, Nx, Ny]
source_time_profiles or source_activity_defaults
source_matrix: [Nx * Ny, K]
raw_metadata: dict
```

- Add optional source splitting hooks, but do not make spatial splitting mandatory in this task. The default source groups should include brick kilns, industries, population density, and traffic time-of-day maps for later temporal-basis construction.

**Outputs and artifacts**

- Named source maps are available from simulator utilities.
- Each source map carries enough metadata to reconstruct its crop, raw file,
  normalization convention, and scale factor.

**Acceptance checks**

- New source loader returns exactly aligned 40x40 source maps.
- Source maps are not collapsed into an aggregate `S_known` for the new IASA path.
- Traffic time-of-day maps are available for constructing a diurnal traffic basis.

### Task 4: Simulator support for inventory activities and wind providers

**Objective**

Allow the simulator to build emissions from inventory activities and run under real or synthetic wind providers:

```text
S_total(t) = sum_k source_map_k * theta_k(t)
```

**Likely files**

- `sim/polsim.py`
- Optional: `sim/pol_sources.py`
- Optional: `model/iasa/wind.py`
- Optional: `model/iasa/activity.py`
- Optional: `data/poldata.py`

**Implementation details**

- Extend `PolGrid` to optionally store:

```text
source_names
source_maps
source_matrix
```

- Add a helper:

```text
combine_inventory_sources(source_maps, theta, nonnegative=True)
```

- Generalize from constant `theta` to time-varying nonnegative activity:

```text
theta_k(t) >= 0
```

- Implement v1 time variation through low-dimensional temporal bases rather than an unconstrained activity value at every timestamp.
- Add source-specific activity defaults:
  - traffic: diurnal basis or fixed hourly profiles informed by `traffic_00`, `traffic_06`, `traffic_12`, and `traffic_18`
  - brick kilns: seasonal/intermittent basis with sparse or blocky activation
  - industry: day-to-day or slowly varying activity basis
  - population-related activity: slowly varying or constant baseline unless a better proxy is available
- Add a simulator path that accepts `source_theta`, `source_activities`, or temporal-basis coefficients.
- If inventory activities are provided, use them as the source term at each simulator step.
- Do not model residual or unknown spatial mass as a learned `S_unknown` field in
  the IASA path. Unexplained broad components should be handled through the
  background basis and residual diagnostics.
- Clamp or validate nonnegative activities.
- Add a `WindProvider` interface or equivalent input contract supporting:
  - `real_imputed_new_delhi`
  - `constant_direction`
  - `single_direction_synthetic`
  - `diurnal_synthetic`
  - `ar1_synthetic`
  - `multi_direction_synthetic`
- For the real New Delhi workflow, use imputed government wind as the default. Synthetic wind providers should be used for controlled identifiability experiments.
- Save source names, activity time series, wind source, and exact `Vx/Vy` sequences in simulator outputs where useful.

**Outputs and artifacts**

- Simulator can run with named source activities.
- Residual/background components remain separate from named inventory activities.

**Acceptance checks**

- A zero activity sequence produces no inventory emissions beyond any explicitly provided background or initial condition.
- A one-hot source activity activates exactly one source group.
- A one-basis temporal coefficient produces the expected `theta_k(t)` profile.
- Real imputed wind can be passed into the simulator without invoking synthetic `monsoon_wind_series`.
- Inventory-activity runs do not require a learned `S_unknown` field.

### Task 5: Open-boundary lagged response matrix builder

**Objective**

Build the central object required by the new paper: `H_lag`, using a paper-faithful open-boundary transport response rather than the legacy edge-hold simulator boundary.

**Likely files**

- New `model/iasa/response.py`
- New `model/iasa/__init__.py`
- Optional: `scripts/build_pol_response_matrix.py`

**Implementation details**

- Implement `model/iasa/response.py` as the owner of `H_lag` construction. Its primary public API should be:

```text
build_lagged_response_matrix(...) -> H_lag, metadata
```

- The builder should accept:

```text
source_maps or source_matrix
source_names
activity_bases
source_basis_column_metadata
grid
params
observer
dt
times
steps
save_every
lag_window
wind_sequence_or_provider
response_config
initial_condition policy
baseline/background policy
```

- The v1 response implementation should use the open-boundary puff/plume approximation described by the paper. It must be the default response implementation for paper-facing IASA results.
- Open-boundary behavior is required:
  - Emissions leaving the modeled domain are removed.
  - Boundary exits are never reflected, wrapped, clamped, or renormalized.
  - Unit source releases are transported through the supplied wind sequence and accumulated into sensor fingerprints over `lag_window`.
  - Kernel or plume mass outside the domain is dropped, so in-domain mass is non-increasing under pure transport.
- For each source group and temporal basis component, generate an open-boundary unit response and record its sensor trajectory.
- Stack each `(source group, temporal basis component)` fingerprint into one column. Constant source activities are represented as the special case with one constant temporal basis per source.
- Support response-matrix construction under:
  - city-level imputed New Delhi wind from the required ImputeFormer pipeline
  - single-direction synthetic wind
  - progressively more diverse synthetic wind regimes
  - fixed supplied `Vx/Vy` arrays
  - later spatially varying wind without changing downstream diagnostics or fitting APIs
- Produce:

```text
H_lag: [m * T_effective, K_basis]
row_index metadata: sensor id, time index, lag policy
source_names
source_basis metadata
wind metadata
boundary_mode
response_implementation
dropped_mass summaries
```

- Make baseline subtraction explicit. Recommended default:
  - Run a zero-source or background-only baseline.
  - Subtract that baseline from each unit-source response.
- Record the exact wind sequence used to build each `H_lag`.
- For New Delhi, default to imputed government `WD/WS` converted to `Vx/Vy`.
- For wind-diversity experiments, make the wind regime an explicit saved config dimension.
- Do not silently reuse the current `sim/polsim.py` edge-hold rollout as the paper-facing `H_lag` generator. If PDE unit rollouts are retained as an auxiliary response mode, metadata must label them as `edge_hold_pde`; if an open-boundary PDE mode is added later, it must expose `boundary_mode="open"`.

**Outputs and artifacts**

- A reusable open-boundary response-matrix construction API.
- Optional script to save `H_lag` and metadata to `data/` or `logs/iasa_*`.

**Acceptance checks**

- `H_lag.shape == (num_sensors * num_times, num_source_basis_columns)`.
- A source or puff leaving the domain stops contributing after exit.
- Total in-domain response mass is non-increasing under pure transport with no source reinjection.
- No boundary reflection or wraparound signal appears at opposite edges.
- One-hot source-basis coefficient prediction matches the corresponding open-boundary response column after baseline subtraction.
- Repeated runs with the same real wind file or synthetic wind seed produce identical matrices.
- The saved metadata is sufficient to map each fitted coefficient back to source name and temporal basis, and to prove which boundary mode generated the matrix.

**Sanity experiment requirements**

Task 5 should also create the reusable tiny sanity runner that later tasks extend. Put it under `experiments/iasa_pol/sanity.py` or `scripts/run_iasa_sanity.py`, and support:

```text
--gate response
--gate projection
--gate diagnostics
--gate fit
--gate merge
--gate all
```

The default sanity setup should be fully synthetic and should not require New Delhi data or ImputeFormer training:

```text
grid size: 16x16
timesteps: 40 to 80
lag window: 8 to 16
source count: 3
sensor count: 4
seed: 123
```

Use grid coordinates consistent with the response builder. If `x` increases east and `y` increases north, use:

```text
sources:
  west_source: compact one-cell or Gaussian source centered near (3, 8)
  east_edge_source: compact one-cell or Gaussian source centered near (14, 8)
  north_source: compact one-cell or Gaussian source centered near (8, 12)

sensors:
  west_sensor/upwind: near (1, 8)
  east_sensor/downwind: near (12, 8)
  north_sensor: near (8, 14)
  south_sensor: near (8, 3)

wind cases:
  eastward: Vx > 0, Vy = 0 for all timesteps
  northward: Vx = 0, Vy > 0 for all timesteps
  two_direction: eastward first half, northward second half
```

The exact units and speeds should match the response implementation, but the puffs must visibly move across the small domain within the configured horizon. The runner should save compact JSON summaries and optional `.npz` arrays under `logs/iasa_sanity/` or `/tmp/iasa_sanity/`. Every gate should print `PASS` or raise a clear error naming the failed metric and expected range. Use the real public APIs rather than private helper-only shortcuts.

Gate S1, completed as part of Task 5, builds `H_lag` for the shared toy geometry with constant temporal bases and open-boundary transport. It must verify:

- `H_lag` is nonzero and has shape `(num_sensors * num_times, num_sources)`.
- Metadata records `boundary_mode="open"`, wind sequence, lag policy, source columns, sensor/time row index, and dropped-mass summaries.
- Under eastward wind, `west_source` produces a larger response norm at `east_sensor` than at `west_sensor`.
- Under eastward wind, `east_edge_source` exits quickly and has less total in-domain response than `west_source`.
- After `east_edge_source` exits, later contribution does not reappear at the west edge or opposite boundary.
- Under northward wind, `north_sensor` response increases relative to the eastward case for at least one source placed south or interior.
- `two_direction` fingerprints are not identical to single-direction fingerprints.
- If pure transport with no source reinjection is exposed in metadata, total in-domain mass is non-increasing after each puff release leaves the source time.

Suggested pass/fail tolerances:

```text
norm(H_lag) > 0
downwind_norm > 2 * upwind_norm for the designed source/wind pair
east_edge_total_mass < west_source_total_mass
opposite_edge_late_signal <= 1e-6 or <= 1e-4 * source_column_norm
max_abs(H_two_direction - H_eastward) > 1e-6
```

If the response approximation is intentionally diffuse, adjust only the ratio thresholds, not the qualitative checks.

### Task 6: Background basis and projection

**Objective**

Implement background correction and projected inverse problem:

```text
H_tilde = P_Q_perp H_lag
Y_tilde = P_Q_perp Y
```

**Likely files**

- New `model/iasa/background.py`
- New `model/iasa/projection.py`
- Optional: `scripts/build_pol_response_matrix.py`

**Implementation details**

- Implement background basis constructors:
  - constant offset
  - per-sensor intercepts
  - global temporal trend
  - low-order temporal polynomial
  - optional user-provided basis
- Represent `Q` as a matrix with rows aligned to the stacked sensor-time observation vector.
- Implement stable orthogonal projection using QR or SVD:

```text
P_Q_perp X = X - Q (Q^+ X)
```

- Apply the same projection to `H_lag` and stacked observations.
- Save projection metadata: basis type, rank of `Q`, and removed dimensions.

**Outputs and artifacts**

- `H_tilde`
- `Y_tilde`
- `Q` metadata

**Acceptance checks**

- Projected columns are numerically orthogonal to `Q`.
- With empty `Q`, projected outputs equal inputs.
- With constant `Q`, a constant observation vector projects close to zero.

**Sanity experiment requirements**

Gate S2, completed as part of Task 6, extends the Task 5 sanity runner with `--gate projection`. Use the Task 5 toy `H_lag` and create:

```text
Y = H_lag c_true + Q beta
c_true = [1.0, 0.5, 0.0]
Q = per-sensor intercepts or one global constant plus one linear time trend
```

Project both `H_lag` and `Y` to produce `H_tilde` and `Y_tilde`. The gate must verify:

- Empty `Q` returns unchanged `H_lag` and `Y`.
- With nonempty `Q`, projected columns and projected observations are numerically orthogonal to `Q`.
- The known background component `Q beta` is removed from `Y_tilde`.
- Source columns are not accidentally erased in the normal background case.
- In a deliberately over-flexible `Q` case that includes a source-like column, the affected source visibility drops and a warning or high absorption score is produced.

Suggested pass/fail tolerances:

```text
||Q.T @ H_tilde||_max <= 1e-6 * max(1, ||H_lag||)
||Q.T @ Y_tilde||_max <= 1e-6 * max(1, ||Y||)
normal_projection_visibility_ratio >= 0.2 for every designed visible source
overflexible_projection_visibility_ratio < normal_projection_visibility_ratio
```

### Task 7: Identifiability diagnostics

**Objective**

Compute diagnostics that explain whether source activities or source-basis coefficients are identifiable.

**Likely files**

- New `model/iasa/diagnostics.py`
- Optional: `scripts/diagnose_pol_sources.py`

**Implementation details**

For `H_tilde`, compute:

- Matrix rank at configurable tolerance.
- Singular values.
- Smallest nonzero singular value.
- Condition number.
- Effective rank.
- Per-source visibility, using projected column norms.
- Pairwise source-fingerprint coherence:

```text
coherence(i, j) = abs(<h_i, h_j>) / (||h_i|| ||h_j||)
```

- Background absorption:

```text
absorption_k = ||P_Q h_k_lag|| / ||h_k_lag||
```

or the complementary removed fraction, with naming made explicit.

- Ray distance where feasible, using normalized nonnegative source fingerprints.
- Perturbation sensitivity proxies based on singular values and condition number.

**Outputs and artifacts**

- Diagnostics dictionary.
- Human-readable table.
- Machine-readable `.npz` or `.json` output.

**Acceptance checks**

- Duplicate source columns yield high coherence and rank deficiency.
- Orthogonal synthetic columns yield low coherence and stable rank.
- Zero or near-zero source columns are flagged as weakly visible.

**Sanity experiment requirements**

Gate S3, completed as part of Task 7, extends the sanity runner with `--gate diagnostics`. Run diagnostics on:

```text
orthogonal_case: nearly orthogonal source columns
duplicate_case: column 2 exactly equals column 1
weak_case: one column has near-zero norm
```

Also run diagnostics on the response matrices from Gate S1 for eastward and two-direction wind. The gate must verify:

- `orthogonal_case` has full rank, low maximum coherence, and finite condition number.
- `duplicate_case` has rank deficiency and coherence close to 1 for the duplicated pair.
- `weak_case` flags the near-zero source as weakly visible.
- For the designed toy geometry, adding the second wind direction does not make identifiability worse by the reported metrics unless the summary explicitly explains the exception.
- Diagnostics output includes rank, singular values, condition number, effective rank, visibility, pairwise coherence, background absorption if projection was used, and warning flags.

Suggested pass/fail tolerances:

```text
duplicate_pair_coherence >= 0.999
weak_column_visibility <= 1e-8 or weak_visibility_flag == true
orthogonal_max_coherence <= 0.1
two_direction_sigma_min >= eastward_sigma_min - 1e-8
two_direction_max_coherence <= eastward_max_coherence + 1e-8
```

If the toy geometry does not satisfy the wind-diversity inequalities, redesign the sanity setup; do not silently weaken this gate.

### Task 8: IASA nonnegative fitting

**Objective**

Estimate nonnegative source-basis coefficients:

```text
c_hat = argmin_{c >= 0} ||Y_tilde - H_tilde c||^2
```

**Likely files**

- New `model/iasa/fit.py`
- Optional: `model/iasa/results.py`
- Optional: `scripts/run_pol_iasa.py`

**Implementation details**

- Prefer `scipy.optimize.nnls` or `scipy.optimize.lsq_linear` if SciPy is available in the runtime environment.
- Provide a fallback PyTorch optimizer:
  - optimize unconstrained raw parameters
  - transform through `softplus` or clamp projected values
  - stop by tolerance or fixed iteration cap
- Return:

```text
c_hat
theta_hat(t)
source_contribution_summaries
fitted_sensor_vector
residual_vector
residual_norm
source_names
source_basis_metadata
solver metadata
```

- Aggregate fitted coefficients back into interpretable source activity and contribution summaries:
  - total contribution over the fitted window
  - diurnal or hourly contribution summaries for traffic
  - intermittent or active-period summaries for brick kilns
  - day-to-day summaries for industry
  - merged-source summaries when merge recommendations apply
- Estimate uncertainty using one or both of:
  - local linear covariance approximation on active variables
  - bootstrap or wind-ensemble refits when ensembles are available
- Include warning flags for ill-conditioned matrices or unstable estimates.

**Outputs and artifacts**

- Source-basis coefficient estimates.
- Time-varying source activity estimates.
- Aggregated source apportionment numbers.
- Fitted sensor trajectory.
- Residual metrics.
- Optional uncertainty intervals.

**Acceptance checks**

- Synthetic orthogonal case recovers known source-basis coefficients within tolerance.
- Synthetic temporal case recovers known diurnal traffic and intermittent brick-kiln activity within tolerance.
- Duplicate-column case reports unstable or non-unique activity split.
- Nonnegativity is enforced in all solver paths.

**Sanity experiment requirements**

Gate S4, completed as part of Task 8, extends the sanity runner with `--gate fit`. Generate synthetic observations from known coefficients:

```text
Y = H_tilde c_true
c_true = [1.5, 0.7, 0.0]
```

Run the nonnegative fitter. The gate must verify:

- In the well-conditioned noiseless case, recovered coefficients match `c_true` within tight tolerance.
- With small Gaussian noise, recovered coefficients remain close and residual norm is lower than the zero-coefficient baseline.
- Nonnegativity is enforced for every solver path.
- In a duplicate-column case, individual duplicate coefficients may be unstable, but their sum matches the true merged contribution.
- In an ill-conditioned case, fit metadata includes a warning rather than presenting the result as fully stable.

Suggested pass/fail tolerances:

```text
noiseless_relative_coefficient_error <= 1e-4
noisy_relative_coefficient_error <= 0.1
residual_norm < zero_model_residual_norm
min(c_hat) >= -1e-8
duplicate_pair_sum_error <= 1e-4 in noiseless duplicate case
```

### Task 9: Merge recommendation system

**Objective**

Report the finest defensible source grouping by merging or flagging indistinguishable sources.

**Likely files**

- New `model/iasa/merge.py`
- Update `model/iasa/diagnostics.py`
- Optional: `scripts/run_pol_iasa.py`

**Implementation details**

- Define configurable thresholds:
  - high coherence threshold
  - low visibility threshold
  - minimum singular value or condition threshold
  - uncertainty width threshold
- Build a graph where source groups are connected if they are indistinguishable under current diagnostics.
- Connected components become merge candidates.
- Report:

```text
source-level activity summaries
source-basis coefficient groups
merged-group activity summaries
merge reason
diagnostic values that triggered the merge
```

- Keep recommendations deterministic for identical inputs.
- Make threshold defaults conservative and paper-aligned, but easy to override.

**Outputs and artifacts**

- Merge recommendations.
- Merged activity estimates.
- Per-source flags.

**Acceptance checks**

- Identical source fingerprints are recommended for merge.
- Clearly separated synthetic sources are not merged.
- Weakly visible sources are flagged even if they do not form a high-coherence pair.

**Sanity experiment requirements**

Gate S5, completed as part of Task 9, extends the sanity runner with `--gate merge`. Use fingerprints and fits from the duplicate and separated cases. The gate must verify:

- Duplicate fingerprints produce one merge recommendation containing exactly the duplicate source pair.
- The merge reason includes the triggering diagnostic, such as high coherence or rank deficiency.
- Clearly separated synthetic sources are not recommended for merge.
- A weakly visible source is flagged even if it is not connected to a high-coherence pair.
- Merged contribution summaries are computed and match the true total contribution in the duplicate synthetic case.

Suggested pass/fail tolerances:

```text
duplicate_pair_in_same_merge_component == true
separated_sources_merged == false
weak_source_flagged == true
merged_duplicate_total_error <= 1e-4 in noiseless case
```

Gate S6, also completed as part of Task 9 before Task 10 begins, is the minimal end-to-end IASA sanity run. It should be available as `--gate all` and run:

```text
build toy sources
choose synthetic wind
build H_lag
build Q and project
generate synthetic Y from known coefficients plus background
run diagnostics
fit coefficients
recommend merges
write summary
```

The end-to-end gate must verify:

- The full toy pipeline runs from public APIs with one command.
- The summary includes source names, `c_true`, `c_hat`, coefficient error, residual norm, rank, singular values, condition number, visibility, coherence, background absorption, merge recommendations, and response boundary metadata.
- The well-conditioned case recovers source coefficients.
- The duplicate-source case recommends a merge and reports stable merged contribution.
- Generated artifacts are small and go to `logs/` or `/tmp`, not tracked source paths.

Task 10 should not begin until Gates S1 through S6 pass, except when a gate is explicitly marked blocked with a documented implementation reason.

### Task 10: Controlled experiment suite

**Objective**

Rework experiments around the hypotheses in the new paper.

**Likely files**

- New `experiments/iasa_pol/`
- New scripts under `scripts/` or `experiments/iasa_pol/`
- Updates to `data/poldata.py` or new `data/poldata_iasa.py`
- Updates to `evaluation/eval_pol.py` or new `evaluation/eval_pol_iasa.py`

**Implementation details**

Add reproducible experiment configs for:

- Noise levels.
- Wind regimes:
  - constant wind
  - single-direction synthetic wind
  - diurnal wind
  - AR(1)-perturbed wind
  - multi-directional episodes
  - real imputed New Delhi wind
- Sensor layouts:
  - regulatory layout
  - random layouts
  - downwind-focused layouts
  - layouts optimized for larger `sigma_min(H_tilde)` or lower maximum coherence
- Background bases:
  - none
  - constant
  - per-sensor intercept
  - temporal trend
  - intentionally over-flexible basis
- Source group variants:
  - base named inventories
  - spatial splits such as north/south, near/far, or upwind/downwind
  - deliberately coherent source pairs for controlled ambiguity tests
- Activity-basis variants:
  - constant source activities
  - traffic diurnal profiles
  - brick-kiln intermittent profiles
  - industry day-to-day profiles
  - mixed temporal bases with known synthetic coefficients

Map the experiment suite to the new paper hypotheses:

- H1: Singular values predict attribution stability.
- H2: High-coherence groups should be merged.
- H3: Background correction can help or hurt.
- H4: Wind diversity and sensor geometry change resolution. Include a dedicated wind-diversity sweep where all else is fixed and wind changes from single-direction transport to increasingly diverse directions.
- H5: Response-matrix error amplifies attribution error.

For the wind-diversity sweep, report how `rank(H_tilde)`, `sigma_min`, effective rank, condition number, maximum coherence, merge recommendations, and source-activity error change as directional diversity increases. The expected claim is that one-direction transport can leave source fingerprints coherent or invisible, while multi-direction wind exposes sources from different angles and improves identifiability.

**Outputs and artifacts**

- Configured experiment runs.
- Saved `H_lag`, `H_tilde`, diagnostics, fits, and evaluation outputs.
- Summary CSV or JSON tables for analysis.

**Acceptance checks**

- Each hypothesis has at least one runnable experiment config.
- Results include both attribution accuracy and identifiability diagnostics.
- Runs are reproducible from saved config and seed.

### Task 11: Evaluation and reporting

**Objective**

Evaluate the new method using source-apportionment metrics instead of free-field source recovery metrics, and produce real New Delhi apportionment tables.

**Likely files**

- New `evaluation/eval_pol_iasa.py`
- New `model/iasa/reporting.py`
- Optional plotting scripts

**Implementation details**

Compute:

- Absolute and relative source activity error when synthetic ground truth is available.
- Per-source absolute error.
- Merged-group error.
- Fitted sensor trajectory error.
- Projected residual error:

```text
||Y_tilde - H_tilde c_hat||_2
```

- Rank, singular values, condition number, effective rank.
- Visibility, coherence, absorption, merge recommendations.
- Uncertainty interval coverage when synthetic ground truth is known.
- Correlation between diagnostics and attribution error across experiment sweeps.
- Real New Delhi apportionment summaries from observed `pm25`, imputed `WD/WS`, named inventories, and temporal activity bases.
- Source-level and merged-source contribution tables with uncertainty intervals.

Produce:

- Human-readable run summaries.
- Machine-readable result files.
- Tables suitable for paper figures or appendices.
- New Delhi tables that clearly separate identifiable source-level contributions from merged or ambiguous groups.

**Outputs and artifacts**

- IASA evaluation script.
- Result tables for each experiment family.

**Acceptance checks**

- Evaluation works on a single saved IASA run.
- Evaluation aggregates multiple runs into one summary table.
- Merged-group metrics are reported whenever merge recommendations exist.
- New Delhi smoke run emits an apportionment report without requiring ground-truth source activities.

### Task 12: Documentation and cleanup

**Objective**

Make the new workflow discoverable and separate it clearly from legacy STAMP/SimGrad workflows.

**Likely files**

- `README.md`
- `ROADMAP_STAMP_NEW.md`
- Optional: `docs/iasa_workflow.md`
- Optional: docstrings in `model/iasa/*`

**Implementation details**

- Document the recommended workflow:

```text
1. Build or load named inventories.
2. Load and impute New Delhi `WD/WS`, or choose a synthetic wind provider.
3. Build temporal activity bases.
4. Generate or load observations.
5. Build H_lag.
6. Build Q and project to H_tilde.
7. Run diagnostics.
8. Fit nonnegative source-basis coefficients.
9. Report source contributions, uncertainty, and merge recommendations.
10. Evaluate controlled experiments and real New Delhi apportionment.
```

- Identify legacy modules:
  - `model/calibrator/tuner_pol_simgrad.py`
  - `archive/tuner_pol_stamp.py`
  - MPRNN prior code where used only for optional comparison
- Include commands for a minimal end-to-end IASA run.
- Keep old pollution calibration instructions if backwards compatibility remains.

**Outputs and artifacts**

- Updated project documentation.
- Clear distinction between old STAMP/SimGrad and new IASA paths.

**Acceptance checks**

- A new contributor can follow documentation to run the minimal IASA pipeline.
- Documentation names the expected runtime environment.
- Legacy code is not presented as the main implementation of the new paper.

## 4. Test Plan

The implementation should add tests or smoke checks as the IASA modules are introduced.

In addition to code-level tests, Tasks 5 through 9 must run the tiny sanity experiment gates defined above. These gates are deliberately smaller than paper experiments: they should use synthetic sources, synthetic wind, short horizons, and known coefficients so failures identify conceptual or algorithmic mistakes early.

Required sanity-gate ordering:

```text
Task 5 complete -> Gate S1 response sanity passes
Task 6 complete -> Gate S2 projection sanity passes
Task 7 complete -> Gate S3 diagnostic sanity passes
Task 8 complete -> Gate S4 fitting sanity passes
Task 9 complete -> Gate S5 merge sanity passes
Before Task 10 -> Gate S6 minimal end-to-end IASA sanity passes
```

These sanity gates are not replacements for the unit tests below. Unit tests check local contracts; sanity gates check whether the assembled method behaves as intended on toy scientific cases.

### Smoke tests

- Load source inventories and verify names, shapes, crop metadata, and
  per-source normalization metadata.
- Load government `WD`/`WS` and verify timestamps, station ids, missingness masks, and units are preserved.
- Run wind imputation on a short window and verify imputed `WD`/`WS` contain no missing values.
- Build a simulator grid and run a short inventory-activity rollout.
- Construct a small open-boundary `H_lag` with a reduced time horizon.

### Open-boundary response tests

- With constant eastward wind, a source near the east outflow boundary should exit quickly and then produce no additional contribution.
- With constant northward wind, an interior source should move toward the expected downwind sensors before leaving the domain.
- Domain-exit mass should be reported and should never be renormalized back into the grid.
- No reflected or wrapped signal should appear at the opposite domain edge after a puff exits.
- Metadata for every saved response matrix should record `boundary_mode="open"` for the paper-facing response implementation, or `response_implementation="edge_hold_pde"` for any legacy PDE-response comparison.

### Shape and consistency tests

- Verify:

```text
H_lag.shape == (num_sensors * num_times, num_source_basis_columns)
H_tilde.shape == H_lag.shape
c_hat.shape == (num_source_basis_columns,)
theta_hat_t.shape == (num_sources, num_times)
Y_tilde.shape == (num_sensors * num_times,)
```

- One-hot source-basis activity should match the corresponding open-boundary response column after projection and baseline subtraction.
- Empty background basis should be a no-op.
- `WD/WS -> Vx/Vy` conversion should pass cardinal-direction cases with the documented meteorological convention.

### Numerical diagnostics tests

- Duplicate columns should produce rank deficiency and high coherence.
- Orthogonal synthetic columns should produce low coherence.
- Near-zero columns should produce low visibility flags.
- Increasing background basis flexibility should not increase projected source visibility.
- Synthetic one-direction wind should produce lower or equal identifiability scores than a matched multi-direction wind case in a controlled layout where wind diversity is designed to reveal source separation.

### Fitting tests

- Recover known nonnegative activities in a well-conditioned synthetic case.
- Recover known diurnal traffic activity and intermittent brick-kiln activity in a synthetic temporal-basis case.
- Preserve nonnegative estimates in both SciPy and PyTorch fallback solvers.
- Flag instability in a high-coherence or rank-deficient case.

### Regression tests

- Existing simulator APIs should not break without a deprecation note.
- Existing edge-hold simulator output should never be mislabeled as the open-boundary response used for paper-facing IASA claims.

### End-to-end tests

- Run a minimal IASA pipeline:

```text
load inventories -> impute/load wind -> build temporal bases -> build H_lag -> project -> diagnose -> fit coefficients -> recommend merges -> evaluate
```

- Save diagnostics and fit outputs.
- Verify summary report contains source activities, residuals, singular values, visibility, coherence, merge recommendations, and response boundary metadata.
- Run a New Delhi smoke test that uses observed `pm25`, imputed `WD/WS`, named inventories, and temporal bases to emit an apportionment report without requiring ground-truth source activities.

## 5. Suggested Package Layout

Add a focused IASA package instead of expanding the old calibrator scripts.

```text
model/iasa/
  __init__.py
  sources.py        # optional if not placed under sim/
  wind.py           # WD/WS loading, imputed wind products, WindProvider
  activity.py       # source-specific temporal bases and theta_k(t)
  response.py       # open-boundary H_lag construction
  background.py     # Q basis construction
  projection.py     # P_Q_perp application
  diagnostics.py    # rank, singular values, coherence, visibility, absorption
  fit.py            # NNLS / nonnegative fitting
  merge.py          # merge recommendations
  reporting.py      # summaries and tables
```

Recommended scripts:

```text
scripts/smoke_iasa_runtime.py
scripts/impute_new_delhi_wind.py
scripts/build_pol_response_matrix.py
scripts/run_pol_iasa.py
scripts/diagnose_pol_sources.py
```

Recommended experiment area:

```text
experiments/iasa_pol/
  configs/
  run_experiment.py
  summarize_results.py
```

Optional baseline or imputation area:

```text
baselines/imputeformer/
```

Use this path only if the FieldFormer/ImputeFormer implementation is not already available elsewhere in the project. The roadmap should document whether the code is vendored, referenced as an external dependency, or reimplemented locally.

## 6. Defaults and Assumptions

- The implementation target is the pollution/source-apportionment path.
- Heat and shallow-water modules should remain untouched unless a shared utility naturally benefits them.
- The current free-field unknown-source workflow is legacy and should not shape
  the IASA source-inventory design.
- Treat `sim/govdata_1H_current.csv` as the authoritative local New Delhi source for observed `pm25`, `WD`, and `WS`.
- Use imputed real `WD/WS` as the default wind input for the New Delhi apportionment workflow.
- Use synthetic wind providers for controlled identifiability experiments.
- The first IASA version should use temporal-basis coefficients, not unconstrained `theta_k(t)` at every timestamp.
- Traffic should have a diurnal activity basis, brick kilns should support seasonal or intermittent activity, and industry should support day-to-day or slowly varying activity.
- The response-matrix builder should support fixed supplied `Vx/Vy` sequences, imputed New Delhi wind, and synthetic wind providers.
- Paper-facing `H_lag` construction should prefer the dedicated open-boundary puff/plume response implementation over minimal reuse of the current edge-hold PDE simulator.
- Spatially varying wind can be added after v1; a city-level imputed wind sequence is acceptable for the first open-boundary response path if the metadata documents that choice.
- Background projection should be implemented before diagnostics and fitting, because the new paper treats `H_tilde` as the central object.
- Merge recommendations should be reported as recommendations, not silently applied to source-level estimates.
- Any generated large matrices or experiment outputs should go under `logs/` or a clearly named generated-data path, not be committed by default.
- If the FieldFormer/ImputeFormer code is not present in this repo, adding or adapting it is required before the real New Delhi apportionment experiment can be considered complete.

## 7. Implementation Milestones

### Milestone A: Minimal IASA core

Complete Tasks 1 through 8 and sanity Gates S1 through S4.

Deliverable:

- Named inventories load.
- New Delhi `WD/WS` can be loaded, imputed, and converted to `Vx/Vy`.
- Inventory activity simulator path works.
- Temporal activity bases can be generated.
- `H_lag` and `H_tilde` can be constructed.
- Diagnostics run.
- Nonnegative source-basis coefficients can be fit for one dataset.
- Toy response, projection, diagnostics, and fitting sanity gates pass with saved summaries.

### Milestone B: Identifiable-resolution reporting

Complete Task 9, sanity Gates S5 and S6, and the single-run parts of Task 11.

Deliverable:

- Fit output includes diagnostics, uncertainty where available, source flags, and merge recommendations.
- One command can run a minimal source-apportionment report.
- New Delhi smoke run produces an apportionment report using imputed wind.
- Toy merge and end-to-end IASA sanity gates pass before broad controlled experiments begin.

### Milestone C: Paper-style experiments

Complete Tasks 10 and 11.

Deliverable:

- Controlled sweeps validate H1-H5, including the wind-diversity claim.
- Results tables connect attribution error to response-matrix geometry.
- New Delhi tables report source-level and merged-source apportionment numbers.

### Milestone D: Documentation and cleanup

Complete Task 12.

Deliverable:

- The README and docs identify the new IASA workflow as the main implementation of `STAMP_new.pdf`.
- Legacy STAMP/SimGrad paths are documented as baselines or prior work.
