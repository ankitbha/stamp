# Roadmap for Implementing `STAMP_new.pdf`

## 1. Paper Delta

### Old paper: limits of sparse-sensing inverse problems

`STAMP_old.pdf` frames the project around fundamental limits. The central claims are:

- Sparse sensor observations induce a low-dimensional observation space for a high-dimensional physical state.
- Under finite horizons, multiple source configurations can produce indistinguishable sensor trajectories.
- Distinct PDE models can collapse in observation space; in particular, advection-diffusion can become observationally close to pure advection under sparse and spatially isolated sensing.
- The main empirical goal is to demonstrate non-identifiability and model indistinguishability, even when the forward simulator and optimizer are accurate.

The old codebase reflected this framing through free spatial-field recovery, but
the active repository target is now IASA-only. Legacy files may be kept locally
under `archive/` for reference, but they are not maintained, tested, or part of
the repository contract.

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

- Separate named source inventories instead of one aggregate source field.
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

The active tree should contain only code needed for this IASA path. Old STAMP,
SimGrad, heat, shallow-water, notebook, free-field pollution, and ablation files
should be moved out of tracked active paths and into `archive/` if they are kept
at all.

## 2. Current Codebase Map

### Pollution simulator

`sim/polsim.py` is the active pollution utility module. It currently:

- Loads seven pollution intensity maps as separate named inventories:
  - brick kilns
  - industries
  - population density
  - traffic at 00, 06, 12, and 18 hours
- Crops the 80x80 source maps to the 40x40 simulation domain.
- Records per-source normalization metadata without aggregating the maps.
- Builds an initial condition from government sensor data through kriging.
- Generates a monsoon-like synthetic wind series.

The simulator currently ignores the `WD` and `WS` meteorological fields already present in `sim/govdata_1H_current.csv`. It must be extended to accept actual wind sequences, not only internally generated synthetic winds.

The current simulator uses an edge-hold boundary condition. This is not the
paper-faithful open-boundary response operator used to define `H_lag`. The IASA
response-matrix path should therefore use a dedicated open-boundary response
implementation, or explicitly label any diagnostic-only PDE rollout response as
`edge_hold_pde` until an open-boundary PDE mode exists.

Old advection-only and diffusion-only pollution variants belong under
`archive/` if retained locally. They are not part of the active IASA
implementation path and must not be imported by active code.

### Archive policy

Legacy files removed from active tracked paths should be moved under
`archive/`. This directory is ignored, untracked, and non-contract: it may be
absent from clean checkouts, and no active code, tests, roadmap tasks, README
commands, or experiment scripts may import from or rely on it.

### Government weather data

`sim/govdata_1H_current.csv` contains hourly government station observations with columns:

```text
monitor_id, timestamp_round, AT, RH, WD, WS, pm10, pm25
```

The local file has 32 stations and 21,960 timestamps. `WD` and `WS` have substantial missingness, so real New Delhi apportionment needs an imputation stage before the wind can drive the transport model. The roadmap should treat this CSV as the authoritative local source for both observed `pm25` and weather fields.

There are references to FieldFormer-style data generation in the repository, but no local ImputeFormer implementation is currently visible in the tracked code paths. The implementation should therefore either locate the existing FieldFormer/ImputeFormer baseline used by the project or vendor/adapt it into a documented local path before claiming full New Delhi wind-imputation support.

### Data generation

The old pollution free-field dataset generator has been removed from the active
tree. New data generation should be inventory-activity based.

### Calibration

The old pollution SimGrad tuner has been removed from the active tree. New
calibration should estimate nonnegative source-activity coefficients over named
inventories.

### Evaluation

The old pollution free-field evaluator has been removed from the active tree.
The new paper needs evaluation over source activities, merged groups,
response-matrix diagnostics, and identifiability predictions.

### Prior model

The old dynamics-prior training machinery is orthogonal to the new IASA
formulation and belongs outside the tracked active tree unless a specific
utility is reused through an explicit IASA module.

### Main mismatch

The active codebase is being redirected toward estimating nonnegative activity
coefficients over named inventories, using real or controlled wind sequences,
and auditing whether those source groups are identifiable. A clean active tree
means tracked imports pass, legacy references are gone, and a minimal IASA
sanity path runs without depending on `archive/`.

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

Expose separate named source inventories without directly aggregating them into
a single source field. The proxy maps should remain distinct because their raw
scales are not reliably comparable across source categories.

**Likely files**

- `sim/polsim.py`
- `sim/pol_sources.py`

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

- Do not sum source categories into one aggregate source in any source-loading
  path.
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
- Source maps are not collapsed into an aggregate field, and no aggregate source
  loader is maintained.
- Traffic time-of-day maps are available for constructing a diurnal traffic basis.

### Task 3A: Clean slate active tree

**Objective**

Make the tracked repository active tree IASA-only. Legacy files should be moved
to ignored `archive/` if they are kept locally, and all tracked code should be
free of dependencies on archived files.

**Likely files**

- `ROADMAP_STAMP_NEW.md`
- `.gitignore`
- Active package `__init__.py` files and tests
- Optional: local-only files under `archive/`

**Implementation details**

- Move old heat/SWE modules, old pollution free-field generators, old
  evaluators, old SimGrad tuners, old notebooks, old prior/calibrator code, and
  old advection/diffusion ablation simulators out of active tracked paths and
  into `archive/` if local reference copies are desired.
- Treat `archive/` as ignored, untracked, non-contract storage. It may be absent
  from a clean checkout.
- Remove stale imports and active references to archived modules.
- Active code must not import from `archive/`.
- Active tests, README commands, roadmap tasks, and experiment scripts must not
  require files under `archive/`.
- Keep only IASA-relevant tracked modules plus the source inventories, government
  data inputs, runtime scripts, and tests needed for the IASA workflow.
- Add `scripts/run_iasa_sanity.py` as the minimal Task 3A IASA sanity runner.
  At this stage it should validate the active inventory, weather, and simple
  nonnegative activity source-term path. The later `H_lag`, projection,
  diagnostics, fit, and merge gates are added by Tasks 5 through 9.

**Outputs and artifacts**

- Active tracked tree contains only IASA-relevant code and data inputs.
- Optional local archive copies exist only under ignored `archive/`.
- `scripts/run_iasa_sanity.py` provides a concrete minimal IASA sanity command
  for the current active-tree stage.

**Acceptance checks**

- All tracked Python modules import in the supported container runtime.
- Repository searches show no active references to retired free-field source
  APIs, old non-pollution simulator APIs, old SimGrad entrypoints, or imports
  from local archive storage.
- Source/weather tests, `scripts/smoke_iasa_runtime.py`, and
  `scripts/run_iasa_sanity.py` pass.

### Task 4: Simulator support for inventory activities and wind providers

**Objective**

Build the first IASA source-activity and wind-provider primitives, plus a
minimal inventory-driven rollout or response input path:

```text
S_total(t) = sum_k source_map_k * theta_k(t)
```

**Likely files**

- `sim/polsim.py`
- `sim/pol_sources.py`
- New `model/iasa/wind.py`
- New `model/iasa/activity.py`
- Optional: `model/iasa/sources.py` if source helpers move out of `sim/`

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

- Add a source activity API that produces `theta_k(t)` from constant activities
  or low-dimensional temporal bases.
- Add a temporal-basis coefficient API where a basis matrix `[T, B]` and
  source coefficients `[K, B]` produce `theta: [T, K]`.
- Generalize from constant `theta` to time-varying nonnegative activity:

```text
theta_k(t) >= 0
```

- Implement v1 time variation through low-dimensional temporal bases rather than an unconstrained activity value at every timestamp.
- Add source-specific activity defaults:
  - traffic: diurnal basis or fixed hourly profiles informed by `traffic_00`, `traffic_06`, `traffic_12`, and `traffic_18`
  - brick kilns: seasonal/intermittent basis with sparse or blocky activation
  - industry: day-heavy diurnal profile with optional spatial operating-fraction
    metadata; some fraction may represent 24/7 industries while the remainder
    follows daytime-only operation
  - population-related activity: baseline plus morning, afternoon, and evening
    cooking-related peaks
- For Task 4 v1, industry spatial sampling is represented as metadata and a
  mixed temporal profile, not as mandatory source-map splitting. True spatial
  splitting remains optional and should not block this task.
- Add a minimal IASA path that accepts `source_theta`, `source_activities`, or
  temporal-basis coefficients and emits the source term needed by rollout or
  response construction.
- If inventory activities are provided, use them as the only named-inventory
  source term at each step.
- Do not model residual or unmodeled spatial mass as a learned free field.
  Unexplained broad components should be handled through the background basis
  and residual diagnostics.
- Clamp or validate nonnegative activities.
- Add a `WindProvider` interface or equivalent input contract supporting:
  - `real_imputed_new_delhi`
  - `constant_direction`
  - `single_direction_synthetic`
  - `diurnal_synthetic`
  - `ar1_synthetic`
  - `multi_direction_synthetic`
- For the real New Delhi workflow, use imputed government wind as the default. Synthetic wind providers should be used for controlled identifiability experiments.
- The `real_imputed_new_delhi` provider must consume the saved imputed wind
  product from Task 2 (`Vx`/`Vy` in the `.npz`). If observed-only fallback is
  allowed for smoke checks, it must use a different provider label.
- Save source names, activity time series, wind source, and exact `Vx/Vy` sequences in simulator outputs where useful.

**Outputs and artifacts**

- IASA source-activity and wind-provider primitives exist under `model/iasa/`,
  including `ActivityProfile`, `WindSequence`, and inventory source-term
  construction.
- A minimal inventory-driven rollout or response-input path can construct
  `S_total(t)` from named source activities.
- Residual/background components remain separate from named inventory activities.

**Acceptance checks**

- A zero activity sequence produces no inventory emissions beyond any explicitly provided background or initial condition.
- A one-hot source activity activates exactly one source group.
- A one-basis temporal coefficient produces the expected `theta_k(t)` profile.
- Real imputed wind can be passed into the simulator without invoking synthetic `monsoon_wind_series`.
- Inventory-activity runs do not require a learned free spatial field.
- Industry default activity has higher daytime than nighttime activity.
- Population default activity has local peaks near morning, afternoon, and
  evening cooking windows.
- Activity metadata records source-default assumptions, the industry operating
  fraction, and the seed used for deterministic proxy profiles.

### Task 5: Open-boundary Gaussian puff response matrix builder

**Objective**

Build the central object required by the new paper: `H_lag`, using the
paper-faithful open-boundary Gaussian puff response rather than the edge-hold
simulator boundary.

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
dispersion_config
initial_condition policy
baseline/background policy
```

- Wind input should be mediated through a small sampler/provider interface:

```text
sample(t_index_or_float, position_xy) -> [Vx, Vy]
```

  The current city-level `Vx/Vy` sequence is one adapter. Later spatially
  varying wind should plug into the same sampler interface without changing
  downstream response, diagnostics, or fitting APIs.

- The v1 response implementation should be the paper's open-boundary
  differentiable Gaussian puff approximation. Record
  `response_implementation="open_boundary_gaussian_puff"` and make it the
  default for paper-facing IASA results.
- For a unit release from source cell `i` at location `r_i` and release time
  `tau`, initialize a puff center at `z_i(tau)=r_i` and advect it through the
  supplied or interpolated wind sequence:

```text
z_i(a + substep_dt) = z_i(a) + substep_dt * w_a(z_i(a))
```

- Use grid-index coordinates with integer cell centers. The physical open
  domain extents are `[-0.5, Nx - 0.5] x [-0.5, Ny - 0.5]`. If the puff center
  exits those extents, remove its remaining in-domain mass. Do not reflect,
  wrap, clamp, or reinsert it. The final advection substep should be fractional
  when needed so the puff lands exactly at the requested observation time.
- If the puff remains active, spread its contribution with an anisotropic
  Gaussian kernel centered at the advected puff location:

```text
K_phi(x, z_i, Sigma_i) =
  exp(-0.5 * (x - z_i)^T Sigma_i^{-1} (x - z_i))
  / (2*pi*sqrt(|Sigma_i|))
```

- Use a simple covariance model aligned to the mean wind direction along the
  puff trajectory:

```text
age = (t - tau) * dt
effective_age = max(age, min_dispersion_time)

Sigma_i(t, tau) =
  R_i(t, tau) diag(
    sigma_parallel^2 * effective_age,
    sigma_perp^2 * effective_age
  ) R_i(t, tau)^T
```

- `R_i` should align the first covariance axis with the mean downwind direction
  when wind speed is nonzero; use a deterministic fallback orientation when the
  wind norm is near zero.
- Same-time rows where `t == tau` are included. They use
  `effective_age = min_dispersion_time`, so releases have a finite initial
  Gaussian kernel rather than being dropped.
- Kernel support may be truncated for efficiency, but mass outside the modeled
  domain or outside the truncated support must be reported and must not be
  renormalized back into the grid.
- Do not infer mass by summing `H_lag` rows, because the same puff is observed
  at multiple times. Retained and dropped mass summaries should be defined per
  release kernel or per `(source, basis, release_time, observation_time)`, and
  exit loss should be recorded separately per release event.
- The response config should expose at least:

```text
substep_dt
sigma_parallel
sigma_perp
min_dispersion_time
kernel_truncation_radius
wind_interpolation
zero_wind_orientation
```

- Open-boundary behavior is required:
  - Emissions leaving the modeled domain are removed.
  - Boundary exits are never reflected, wrapped, clamped, or renormalized.
  - Unit source releases are transported through the supplied wind sequence and accumulated into sensor fingerprints over `lag_window`.
  - Gaussian kernel mass outside the domain is dropped, so in-domain mass is
    non-increasing under pure transport.
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
H_lag: [m * T, K_basis] by default
row_index metadata: sensor id, time index, lag policy
source_names
source_basis metadata
wind metadata
boundary_mode
response_implementation
dispersion_parameters
substep_dt
kernel_truncation_radius
kernel_mass_retained summaries
dropped_mass summaries
exit_time summaries
```

  Optional initial-lag trimming may produce `T_effective`; if used, the row
  metadata must record the trim policy. The default Task 5 builder should keep
  all `T` observation times.
- Persist `row_index`, `column_index`, `baseline_policy`, and the row-aligned
  baseline vector inside the response metadata as well as exposing convenient
  result fields. Saving `H_lag` with metadata alone must preserve enough
  provenance to reconstruct every row and fitted coefficient.

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
- Downwind spread increases with lag under nonzero wind.
- Crosswind spread follows `sigma_perp`, along-wind spread follows
  `sigma_parallel`, and changing these parameters changes the response
  anisotropy in the expected direction.
- Kernel mass outside the domain is dropped rather than renormalized, so a puff
  near an outflow boundary has lower retained mass than the same puff in the
  interior.
- One-hot source-basis coefficient prediction matches the corresponding open-boundary response column after baseline subtraction.
- Repeated runs with the same real wind file or synthetic wind seed produce identical matrices.
- The saved metadata is sufficient to map each fitted coefficient back to source
  name and temporal basis, and to prove which boundary mode, response
  implementation, wind sequence, and dispersion parameters generated the
  matrix.

**Sanity experiment requirements**

Task 5 should also create the reusable tiny sanity runner that later tasks extend. Put it under `experiments/iasa_pol/sanity.py` or `scripts/run_iasa_sanity.py`, and support:

```text
--gate task3a
--gate response
--gate projection
--gate diagnostics
--gate fit
--gate merge
--gate all
--strict-all
```

The default sanity setup should be fully synthetic and should not require New Delhi data or ImputeFormer training:

```text
grid size: 16x16
timesteps: 40 to 80
lag window: 8 to 16
source count: 4
sensor count: 4
seed: 123
response_implementation: open_boundary_gaussian_puff
substep_dt: chosen so puffs move smoothly across grid cells
sigma_parallel: larger than sigma_perp for anisotropic downwind spread
sigma_perp: positive crosswind dispersion
kernel_truncation_radius: at least 3 Gaussian standard deviations
```

Use grid coordinates consistent with the response builder. If `x` increases east and `y` increases north, use:

```text
sources:
  west_source: compact Gaussian inventory centered near (3, 8)
  east_edge_source: compact Gaussian inventory centered near (14, 8)
  south_source: compact Gaussian inventory centered near (8, 3)
  interior_source: required compact Gaussian inventory centered near (8, 8),
    used as the matched control for boundary-loss accounting

sensors:
  west_sensor/upwind: near (1, 8)
  east_sensor/downwind: near (12, 8)
  north_sensor: near (8, 14)
  south_sensor: near (8, 3)

wind cases:
  eastward: Vx > 0, Vy = 0 for all timesteps
  northward: Vx = 0, Vy > 0 for all timesteps
  two_direction: eastward first half, northward second half

dispersion cases:
  isotropic: sigma_parallel == sigma_perp
  anisotropic: sigma_parallel > sigma_perp
  boundary_loss: same puff/kernel as interior case, but near outflow boundary
```

The exact units and speeds should match the response implementation, but the
puffs must visibly move across the small domain within the configured horizon.
The runner should save compact JSON summaries and optional `.npz` arrays under
`logs/iasa_sanity/` or `/tmp/iasa_sanity/`. Every gate should print `PASS` or
raise a clear error naming the failed metric and expected range. Use the real
public APIs rather than private helper-only shortcuts.

Gate S1, completed as part of Task 5, builds `H_lag` for the shared toy
geometry with impulse and constant temporal bases and open-boundary Gaussian
puff transport.
It must verify:

- `H_lag` is nonzero and has shape
  `(num_sensors * num_times, num_source_basis_columns)`.
- Metadata records `boundary_mode="open"`,
  `response_implementation="open_boundary_gaussian_puff"`, wind sequence, lag
  policy, source columns, sensor/time row index, dispersion parameters,
  retained-kernel-mass summaries, dropped-mass summaries, and exit-time
  summaries.
- Under eastward wind, `west_source` produces a larger response norm at `east_sensor` than at `west_sensor`.
- Under eastward wind, `east_edge_source` exits quickly and has less total in-domain response than `west_source`.
- After `east_edge_source` exits, later contribution does not reappear at the west edge or opposite boundary.
- Under northward wind, `south_source` produces a larger response norm at
  `north_sensor` than at `south_sensor`.
- `two_direction` fingerprints are not identical to single-direction fingerprints.
- For a fixed interior source and eastward wind, increasing `sigma_parallel`
  while holding `sigma_perp` fixed increases the along-wind second moment more
  than the crosswind second moment.
- For a fixed interior source and eastward wind, increasing `sigma_perp` while
  holding `sigma_parallel` fixed increases the crosswind second moment.
- The isotropic dispersion case has approximately equal along-wind and crosswind
  second moments after orientation into wind-aligned coordinates.
- The boundary-loss case reports lower retained kernel mass than the matched
  interior case and does not renormalize the retained mass to one.
- Every evaluated release kernel clips only numerical quadrature overshoot:
  `retained_fraction = clip(raw_retained_fraction, 0, 1)`, then records
  `retained_mass + dropped_mass = emitted_mass` within `1e-5`.
- Detailed per-kernel diagnostic records may be capped by
  `max_kernel_diagnostic_records`, but metadata must record total count, stored
  count, and whether truncation occurred. Per-column emitted, retained, dropped,
  exit, clipping, and kernel-count aggregates must remain complete.
- If pure transport with no source reinjection is exposed in metadata, total
  in-domain mass is non-increasing after each puff release leaves the source
  time.

Suggested pass/fail tolerances:

```text
norm(H_lag) > 0
downwind_norm > 2 * upwind_norm for the designed source/wind pair
east_edge_total_mass < west_source_total_mass
opposite_edge_late_signal <= 1e-6 or <= 1e-4 * source_column_norm
max_abs(H_two_direction - H_eastward) > 1e-6
anisotropic_along_moment / anisotropic_cross_moment > 1.25 when sigma_parallel > sigma_perp
crosswind_moment_large_sigma_perp > crosswind_moment_small_sigma_perp
abs(isotropic_along_moment - isotropic_cross_moment) / max(isotropic_total_moment, 1e-12) <= 0.25
boundary_retained_mass < matched_interior_retained_mass
max_abs(retained_mass + dropped_mass - emitted_mass) <= 1e-5
```

If the response approximation is intentionally diffuse, adjust only the ratio
thresholds, not the qualitative checks.

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
- New IASA data loading utilities if experiment-specific loaders are needed
- New IASA evaluation utilities if experiment-specific metrics are needed

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

Make the IASA workflow discoverable and make clear that archived legacy files
are outside the active repository contract.

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

- Document that legacy files live only in ignored `archive/` if kept locally,
  and active code must not depend on them.
- Include commands for a minimal end-to-end IASA run.
- Remove old pollution calibration, heat/SWE, SimGrad, and free-field recovery
  instructions from active workflow documentation.

**Outputs and artifacts**

- Updated project documentation.
- Clear IASA-only active workflow documentation.

**Acceptance checks**

- A new contributor can follow documentation to run the minimal IASA pipeline.
- Documentation names the expected runtime environment.
- Documentation does not present archived legacy code as maintained, required,
  or available in clean checkouts.

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

- Verify all tracked Python modules import in the supported container runtime.
- Load source inventories and verify names, shapes, crop metadata, and
  per-source normalization metadata.
- Load government `WD`/`WS` and verify timestamps, station ids, missingness masks, and units are preserved.
- Run wind imputation on a short window and verify imputed `WD`/`WS` contain no missing values.
- Build a simulator grid and construct a short inventory-activity source term.
- Construct a small open-boundary `H_lag` with a reduced time horizon.
- Search active tracked files and verify there are no imports from `archive/`.
- Search active tracked files and verify stale legacy free-field source APIs,
  old non-pollution simulator APIs, and old SimGrad entrypoints are absent.

### Open-boundary response tests

- With constant eastward wind, a source near the east outflow boundary should exit quickly and then produce no additional contribution.
- With constant northward wind, an interior source should move toward the expected downwind sensors before leaving the domain.
- Domain-exit mass should be reported and should never be renormalized back into the grid.
- No reflected or wrapped signal should appear at the opposite domain edge after a puff exits.
- Metadata for every saved response matrix should record `boundary_mode="open"` for the paper-facing response implementation, or `response_implementation="edge_hold_pde"` for any diagnostic-only non-open PDE response.

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

### Clean-slate regression tests

- The active tree does not depend on ignored `archive/` files.
- No tracked active module imports deleted legacy modules.
- The minimal IASA sanity runner passes from a clean checkout where `archive/`
  is absent.
- Any diagnostic-only edge-hold PDE response must be explicitly labeled
  `edge_hold_pde`; paper-facing response matrices must record
  `boundary_mode="open"`.

### End-to-end tests

- Run a minimal IASA pipeline:

```text
load inventories -> impute/load wind -> build temporal bases -> build H_lag -> project -> diagnose -> fit coefficients -> recommend merges -> evaluate
```

- Save diagnostics and fit outputs.
- Verify summary report contains source activities, residuals, singular values, visibility, coherence, merge recommendations, and response boundary metadata.
- Run a New Delhi smoke test that uses observed `pm25`, imputed `WD/WS`, named inventories, and temporal bases to emit an apportionment report without requiring ground-truth source activities.

## 5. Suggested Package Layout

Make `model/iasa/` the core implementation area instead of expanding old
calibrator or simulator scripts.

```text
model/iasa/
  __init__.py
  sources.py        # optional bridge from sim/pol_sources.py
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

`data/pol_weather.py` may remain as a thin data-loading bridge if already
present, but new wind-provider contracts should live in `model/iasa/wind.py`.

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

- The implementation target is an IASA-only pollution/source-apportionment
  active tree.
- Legacy STAMP, SimGrad, heat, shallow-water, free-field pollution, old ablation,
  and notebook workflows are out of scope for active tracked code.
- Removed legacy workflows should not shape the IASA source-inventory design.
- `archive/` is ignored, untracked, non-contract, and may be absent from clean
  checkouts.
- Active code, tests, docs, and scripts must not import from or rely on
  `archive/`.
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

Complete Tasks 1 through 8, including Task 3A, and sanity Gates S1 through S4.

Deliverable:

- Active tracked tree is IASA-only and does not depend on `archive/`.
- Named inventories load.
- New Delhi `WD/WS` can be loaded, imputed, and converted to `Vx/Vy`.
- Coherent `model/iasa/` skeleton exists for wind, activity, response,
  projection, diagnostics, fitting, merging, and reporting.
- Inventory activity source-term construction works.
- Temporal activity bases can be generated.
- `H_lag` and `H_tilde` can be constructed.
- Diagnostics run.
- Nonnegative source-basis coefficients can be fit for one dataset.
- One tiny end-to-end IASA sanity run passes before broader experiments expand.
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
- Archive policy is documented as non-contract local reference storage only.
- No README or roadmap command depends on archived legacy files.
