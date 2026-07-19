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

The goal becomes: estimate source activities and report a conservative source
grouping that is defensible under the sensor layout, wind regime, lag window,
background basis, and noise level, conditional on the declared inventory. For
the real New Delhi case, the roadmap must also produce actual proxy
apportionment numbers, separated observation/transport uncertainty, inventory
robustness scenarios, calibrated residual adequacy when a noise model is
available, and merge recommendations.

### Required new capabilities

The implementation needs these new capabilities:

- Separate named source inventories instead of one aggregate source field.
- Government `WD`/`WS` ingestion, missing-data imputation, and conversion to simulator-ready wind vectors.
- A wind-provider interface that supports real imputed New Delhi wind and controlled synthetic wind regimes.
- A gridded FieldFormer wind field: a coordinate-query imputer queried on every
  response-grid cell and hour to produce a per-cell field `W in R^(T x n x 2)`,
  plus wind-field ensembles (held-out-calibrated error, station bootstrap,
  checkpoint ensembles) tagged as transport uncertainty. The earlier city-level
  sequence is the v1 adapter; the paper-facing New Delhi response uses the
  gridded field.
- Source activity parameterization over inventory groups and temporal activity bases.
- Construction of `H_lag` from a dedicated open-boundary source-response operator under real or synthetic wind sequences.
- Background basis construction and projection.
- Identifiability diagnostics: rank, singular values, condition number, effective rank, visibility, background absorption, pairwise coherence, ray distance where feasible, and perturbation sensitivity.
- Nonnegative source-activity fitting.
- Uncertainty intervals and ambiguity reporting.
- Merge recommendations for indistinguishable source groups.
- Per-sensor source footprints and spatial attribution: a per-sensor
  contribution decomposition and a nonnegative footprint field that pulls each
  sensor's response row back onto the source grid, aggregated to the identifiable
  report groups so no per-sensor separation exceeds the global resolution.
- An optional constrained end-to-end refinement stage that jointly adjusts wind,
  dispersion, source, and background coefficients under physical constraints and
  is accepted only if it does not degrade identifiability.
- Controlled experiments matching the new paper's Experiment 1--10 matrix,
  including the claim that wind diversity improves source separability, and
  separate observed-New-Delhi reporting from observed `pm25`.
- An auxiliary edge-hold advection--diffusion simulator retained as the labeled
  structural forward-model mismatch generator (Experiment 5), never silently
  substituted for the open-boundary puff response.

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

There are references to FieldFormer-style data generation in the repository, but no local FieldFormer implementation is currently visible in the tracked code paths. The paper's wind imputer is FieldFormer, a coordinate-query model. The implementation should therefore either locate the existing FieldFormer baseline used by the project or vendor/adapt it into a documented local path before claiming full New Delhi wind-imputation support.

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

### ~~Task 1: Runtime and dependency baseline~~

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

- Confirm availability of required packages inside the image: `numpy`, `torch`,
  `pandas`, and `pykrige`. SciPy is not part of the required solver path.
- Add a smoke script that imports the pollution simulator, loads source maps, creates a grid, and prints source and sensor shapes without running a long simulation.
- Keep host Python limitations explicit; in the current environment, host Python does not provide `numpy`.

**Outputs and artifacts**

- Runtime instructions in `README.md`.
- A quick smoke command that future tasks can use before deeper tests.

**Acceptance checks**

- Smoke command runs in the Singularity image.
- It imports `sim.polsim`, builds a 40x40 grid, loads source maps, and exits successfully.
- No tracked data artifacts are rewritten by the smoke check.

### ~~Task 2: Weather data and wind imputation~~

**Objective**

Build the real New Delhi wind pipeline from government `WD` and `WS` observations, including missing-data imputation and conversion to simulator-ready wind vectors.

**Likely files**

- New `data/pol_weather.py` or `model/iasa/wind.py`
- Optional: `baselines/fieldformer/` if the FieldFormer implementation is vendored into this repo
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
- Use the FieldFormer coordinate-query baseline implementation for wind imputation. If that implementation is not present in this repository, explicitly add or adapt it under a documented path such as `baselines/fieldformer/` or `model/imputation/`.
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

- Support both station-level wind fields and a city-level aggregate wind sequence. The v1 New Delhi apportionment path uses a city-level sequence; the paper-facing New Delhi response is later upgraded to the gridded FieldFormer field in Task 9A. This task supplies the loaded/imputed station data and conversion that both paths share.

**Outputs and artifacts**

- A reusable wind preprocessing API.
- Saved imputed wind product for New Delhi experiments.
- A smoke script that verifies the imputed wind sequence has no missing values over the selected window.

**Acceptance checks**

- Loader reports 32 stations and the expected timestamp range for the local
  government CSV. Per the paper, the record spans 21,960 hourly timestamps from
  1 May 2018 through 31 October 2020 (IST), and the two Pusa monitors
  (`Pusa_IMD`, `Pusa_DPCC`) are averaged by timestamp into `Pusa_averaged` at the
  mean of their coordinates to yield the 32-sensor layout.
- Raw `WD`/`WS` missingness masks are preserved.
- Imputed `WD`/`WS` have no missing values over the selected experiment window.
- Cardinal-direction conversion tests pass for known synthetic `WD`/`WS` cases.
- `Vx` and `Vy` are aligned to the same timestamps used for `pm25` observations.

### ~~Task 3: Inventory source refactor~~

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

### ~~Task 3A: Clean slate active tree~~

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

### ~~Task 4: Simulator support for inventory activities and wind providers~~

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

### ~~Task 5: Open-boundary Gaussian puff response matrix builder~~

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
  - imputed New Delhi wind from the required FieldFormer pipeline (city-level v1 or gridded field)
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

The default sanity setup should be fully synthetic and should not require New Delhi data or FieldFormer training:

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

### ~~Task 6: Background basis and projection~~

**Objective**

Implement the paper's low-dimensional background model and projected inverse
problem:

```text
Y = H_lag c + Q beta + E
H_tilde = P_Q_perp H_lag
Y_tilde = P_Q_perp Y
```

**Likely files**

- New `model/iasa/background.py`
- New `model/iasa/projection.py`
- Optional: `scripts/build_pol_response_matrix.py`

**Implementation details**

- Implement low-dimensional background basis constructors for the components
  used in the paper:
  - global constant offset
  - centered global linear trend and optional low-order temporal polynomial
  - smooth time-of-day variation using a small number of sine/cosine harmonics
  - day-level intercepts with one reference level removed or centered
  - regional spatial trends from standardized sensor coordinates
  - optional sensor-specific offsets
  - optional user-provided basis with the same row-alignment contract
- Keep the normal paper-facing basis deliberately low-dimensional. Default
  constructors must use only timestamps, day labels, sensor identifiers, and
  sensor coordinates; they must never derive normal background columns from
  `H_lag`, source inventories, or fitted source signals. Expose a configurable
  rank cap with a paper-facing default of `max_background_rank=8`, and reject a
  normal basis whose effective rank exceeds that cap. Deliberately
  over-flexible or source-like bases are allowed only in labeled stress tests.
- Construct `Q in R^(mT x r)` from the Task 5 time-major `row_index`. Require
  `len(row_index) == H_lag.shape[0] == Y.shape[0] == Q.shape[0]`, preserve the
  exact row metadata, and fail on a sensor/time ordering mismatch rather than
  silently projecting misaligned arrays.
- Implement the Moore-Penrose projection with rank-revealing SVD. Determine the
  effective rank using a recorded numerical tolerance, retain the corresponding
  orthonormal basis `U_r`, and apply:

```text
P_Q_perp X = X - U_r (U_r.T X)
```

  A pivoted rank-revealing QR implementation is acceptable only if it uses the
  same explicit rank tolerance. Ordinary unpivoted QR is not sufficient because
  combinations such as a global constant plus all sensor intercepts are
  linearly dependent.
- Apply the same fitted background subspace to `H_lag` and stacked observations.
- Preserve `Q` or `U_r`, `H_removed = H_lag - H_tilde`, and
  `Y_removed = Y - Y_tilde` so Task 7 can reproduce the paper's background
  absorption statistic. Saving metadata alone is not sufficient.
- Save projection metadata containing basis names and types, requested and
  effective rank, singular values, rank tolerance, rank cap, projection method,
  dependent/discarded columns, input and output shapes, and the exact
  sensor-time row ordering.
- The equivalent joint source/background fit from the paper is deferred to the
  fitting task. Task 6 owns the projected formulation used for all downstream
  identifiability diagnostics.

**Outputs and artifacts**

- `H_tilde`
- `Y_tilde`
- `Q` and/or the effective orthonormal basis `U_r`
- `H_removed` and `Y_removed`
- complete background-basis and projection metadata

**Acceptance checks**

- Projected columns are numerically orthogonal to `Q`.
- With empty `Q`, projected outputs equal inputs.
- With constant `Q`, a constant observation vector projects close to zero.
- Duplicate or linearly dependent background columns produce the same projector
  as their independent span and do not change the effective rank.
- Projection is idempotent within tolerance.
- Normal background constructors respect the rank cap, contain no source-derived
  columns, and retain the required visibility of every designed source column.
- `Q`, `H_lag`, `Y`, and all projected outputs carry identical row ordering.

**Sanity experiment requirements**

Gate S2, completed as part of Task 6, extends the Task 5 sanity runner with `--gate projection`. Use the Task 5 toy `H_lag` and create:

```text
Y = H_lag c_true + Q beta
c_true = [1.0, 0.5, 0.0, 0.25, 0.75, 0.0, 0.4, 0.2]
Q_normal = [global constant, centered linear trend,
            first daily sine harmonic, first daily cosine harmonic]
beta = [0.3, -0.1, 0.2, 0.15]
```

The eight entries of `c_true` follow the exact source-major `(source, basis)`
ordering in the Task 5 `column_index`; Gate S2 must assert this mapping rather
than relying on an undocumented hard-coded order. `Q_normal` has requested and
effective rank four, is built independently of `H_lag`, and is below the normal
rank cap.

Project both `H_lag` and `Y` to produce `H_tilde` and `Y_tilde`. The gate must verify:

- Empty `Q` returns unchanged `H_lag` and `Y`.
- With nonempty `Q`, projected columns and projected observations are numerically orthogonal to `Q`.
- The known background component `Q beta` is removed from `Y_tilde`.
- Source columns are not accidentally erased in the normal low-dimensional
  background case.
- A redundant-basis case containing duplicate or dependent columns yields the
  same projection as its independent span and records the reduced effective
  rank.
- In a deliberately over-flexible `Q` case that adds a normalized source-like
  column from `H_lag`, the affected source visibility drops, background
  absorption increases, and the case is explicitly labeled as a stress test.
- The saved `U_r`, removed components, and metadata reproduce the projection and
  provide Task 7 with everything needed to compute per-source background
  absorption.

Suggested pass/fail tolerances:

```text
||Q.T @ H_tilde||_max <= 1e-6 * max(1, ||H_lag||)
||Q.T @ Y_tilde||_max <= 1e-6 * max(1, ||Y||)
max_abs(P_Q_perp(P_Q_perp(X)) - P_Q_perp(X)) <= 1e-8 * max(1, ||X||)
rank(Q_normal) == 4
normal_projection_visibility_ratio >= 0.8 for every designed visible source
normal_background_absorption_ratio <= 0.6 for every designed visible source
overflexible_projection_visibility_ratio < normal_projection_visibility_ratio
overflexible_background_absorption_ratio > normal_background_absorption_ratio
```

### ~~Task 6A: PyTorch computational backend consolidation~~

**Objective**

Make PyTorch the sole computational backend before implementing diagnostics and
fitting. Pandas and NumPy remain permitted only at CSV/NPZ ingestion and
serialization boundaries.

**Likely files**

- `model/iasa/response.py`
- `model/iasa/background.py`
- `model/iasa/projection.py`
- `model/iasa/activity.py`
- `model/iasa/wind.py`
- Tests and `scripts/run_iasa_sanity.py`

**Implementation details**

- Port puff advection, sensor-kernel evaluation, retained/dropped-mass
  diagnostics, observation masking, and baseline handling to `torch.Tensor`
  operations.
- Port background construction and thin-SVD projection to PyTorch.
- Keep row/column metadata as Python records, but keep all numerical arrays on
  the requested device until artifact serialization.
- Expose and propagate explicit `device` and `dtype` configuration. Use float64
  by default for inverse diagnostics/fitting and configurable float32 or float64
  for response construction.
- Record device, dtype, PyTorch version, and CUDA version in saved artifacts.
- Attach inventory identifier/hash/version, transport configuration, exact wind
  realization, dispersion settings, observation mask, and parent-artifact hashes
  to response and projection artifacts.
- Add `ensemble_kind` provenance with exactly `transport` or `inventory` for
  downstream ensemble products. Untagged ensembles must not enter uncertainty
  aggregation.
- Convert to NumPy only through explicit `tensor.detach().cpu().numpy()` at an
  output boundary.
- Remove SciPy as a runtime or solver requirement. No diagnostic or fit may
  invoke a hidden CPU implementation.
- Preserve the existing public row ordering, source-major/basis-minor column
  ordering, boundary semantics, and provenance metadata.

**Outputs and artifacts**

- PyTorch-native response, background, and projection results.
- CPU/GPU parity summaries for Gates S1 and S2.
- Device/dtype and inventory/transport/mask provenance in every generated
  response and projection artifact.

**Acceptance checks**

- CPU PyTorch outputs match the pre-consolidation implementation within
  recorded dtype-appropriate tolerances.
- CPU and CUDA outputs agree within recorded tolerances.
- Response and projection outputs remain on the requested device until
  serialization.
- Gates S1 and S2 pass without SciPy.
- Task 7 does not begin until this consolidation passes.

### ~~Task 7: Identifiability diagnostics~~

**Objective**

Compute coefficient-level diagnostics for the projected source--basis response
and map those diagnostics to source-level warnings without inventing aggregate
fingerprint columns.

**Likely files**

- New `model/iasa/diagnostics.py`
- Optional: `scripts/diagnose_pol_sources.py`

**Implementation details**

Accept `fixed_zero_indices` as a pre-fit diagnostic configuration. Validate it
against `column_index`, remove those columns before constructing the diagnostic
matrix, and retain `original_to_reduced` and `reduced_to_original` mappings.
The default is an empty mask. A coefficient that is small after fitting must
not modify the mask or any diagnostic.

For `H_tilde in R^(N x J)`, using PyTorch on the input device, compute:

- Exactly `J` singular values through `H_tilde.T @ H_tilde`, padding with zeros
  when `J > N` or the matrix is deficient.
- Numerical tolerance
  `tau_num = max(N,J) * eps(dtype) * sigma_1` and numerical rank
  `r_num = count(sigma_i > tau_num)`.
- Primary identifiability score `sigma_J`, including rank-deficiency zeros.
- Condition number `sigma_1 / sigma_J` only at full numerical rank and infinity
  otherwise.
- Separately configured noise-dependent effective rank threshold `tau_sigma`.
- Coefficient visibility `v_j = ||h_tilde_j||_2`, weak set
  `W = {j: v_j <= tau_v}`, and exact source/basis metadata for every `j`.
- Eligible coefficient-fingerprint coherence:

```text
coherence(i, j) = abs(<h_i, h_j>) / (||h_i|| ||h_j||)
```

- Background absorption using the removed/raw norm ratio:

```text
absorption_j = ||H_removed_j|| / ||H_lag_j||
```

- Signed linear-ray distance `sqrt(max(0, 1 - coherence(i,j)^2))` for eligible
  pairs.
- `null` coherence, ray distance, and absorption entries when their required
  norms are weak/zero; never serialize NaN as a metric.
- Eligible ambiguity set `A = {(i,j): i,j not in W and rho_ij > tau_rho}`.
- Per-source weak-basis flags and cross-source summaries retaining the exact
  coefficient pair that attains maximum eligible coherence and minimum ray
  distance.
- Perturbation sensitivity values `1/sigma_J` and the condition number, with
  infinity serialized through an explicit status field plus JSON-safe value.
- For a predeclared collection of historical or simulated wind windows,
  aggregate per-window diagnostics into 5th, 50th, and 95th percentiles of
  `sigma_J`; probabilities of full numerical/effective rank; per-coefficient
  weak probabilities; per-source-pair ambiguity probabilities; and frequencies
  of conservative report components. This is an empirical wind-distribution
  summary, not a theorem about every possible wind field.
- Keep all rank/coherence guarantees uniform over the reduced admissible
  coefficient set. Do not add a tangent-cone or fitted-active-set
  identifiability path.

**Outputs and artifacts**

- A PyTorch-native diagnostics result retaining device and dtype.
- Human-readable table.
- Machine-readable tensor artifact plus JSON summary using `null` for undefined
  pairwise values.
- Wind-ensemble summary with sampled-window provenance and deterministic
  quantile/probability fields.

**Acceptance checks**

- Duplicate coefficient columns yield high coherence, `sigma_J == 0`, infinite
  condition status, and rank deficiency.
- Orthogonal synthetic columns yield low coherence and stable rank.
- Zero or near-zero coefficient columns are flagged as weakly visible and have
  undefined pairwise metrics.
- A predeclared fixed-zero mask removes the intended columns, preserves the
  original index mapping, and changes diagnostics only through that declared
  reduction; post-fit near-zero values do not.
- Wind-ensemble summaries are exactly repeatable for fixed inputs and seeds.

**Sanity experiment requirements**

Gate S3, completed as part of Task 7, extends the sanity runner with `--gate diagnostics`. Run diagnostics on:

```text
orthogonal_case: nearly orthogonal source columns
duplicate_case: column 2 exactly equals column 1
weak_case: one column has near-zero norm
```

Build a dedicated matched wind comparison containing the same multiple source,
basis, and sensor columns under eastward and two-direction wind. The existing
single-column Gate S1 comparison is only a response-change check and must not be
used to test coherence. The gate must verify:

- `orthogonal_case` has full rank, low maximum coherence, and finite condition number.
- `duplicate_case` has rank deficiency and coherence close to 1 for the duplicated pair.
- `weak_case` flags the near-zero source as weakly visible.
- For the designed matched geometry, adding the second wind direction does not
  reduce `sigma_J` or increase maximum eligible coherence. Redesign the geometry
  if necessary; do not weaken the assertion.
- Diagnostics output includes padded singular values, `sigma_J`, numerical and
  effective rank, condition status, visibility, weak set, eligible coherence,
  ray distance, background absorption, source/basis trigger pairs, and warnings.

Suggested pass/fail tolerances:

```text
duplicate_pair_coherence >= 0.999
weak_column_visibility <= 1e-8 or weak_visibility_flag == true
orthogonal_max_coherence <= 0.1
two_direction_sigma_J >= eastward_sigma_J - 1e-8
two_direction_max_coherence <= eastward_max_coherence + 1e-8
```

If the toy geometry does not satisfy the wind-diversity inequalities, redesign the sanity setup; do not silently weaken this gate.

### ~~Task 8: IASA nonnegative fitting~~

**Objective**

Estimate nonnegative source-basis coefficients:

```text
c_hat = argmin_{c >= 0} ||Y_tilde - H_tilde c||^2
        + lambda ||c - c0||^2
```

**Likely files**

- New `model/iasa/fit.py`
- Optional: `model/iasa/results.py`
- Optional: `scripts/run_pol_iasa.py`

**Implementation details**

- Implement projected FISTA in PyTorch as the sole solver path.
- Compute a valid Lipschitz step from `sigma_1(H_tilde)` or a recorded
  power-iteration estimate.
- Project every update with `torch.clamp(c, min=0)` so inactive coefficients can
  be exactly zero.
- Use monotone restart whenever acceleration raises the objective.
- Stop only when both projected-gradient/KKT residual and relative objective
  change satisfy configured tolerances, or when the recorded iteration cap is
  reached.
- Preserve the input device and use the configured inverse-problem dtype.
- Do not import SciPy or transfer to CPU for fitting.
- Accept the same predeclared `fixed_zero_indices` used by Task 7, fit only the
  reduced columns, restore exact zeros in the full coefficient vector, and
  return both index mappings. Reject a fit/diagnostic mask mismatch.
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
device and dtype
convergence status
iteration count
projected-gradient/KKT residual
objective summary
```

- Aggregate fitted coefficients back into interpretable source activity and contribution summaries:
  - total contribution over the fitted window
  - diurnal or hourly contribution summaries for traffic
  - intermittent or active-period summaries for brick kilns
  - day-to-day summaries for industry
  - merged-source summaries when merge recommendations apply
- Estimate active-set covariance with PyTorch linear algebra on
  `H_active.T @ H_active + lambda I`.
- Batch bootstrap or transport-ensemble refits on GPU when shapes agree; record when
  batching is impossible and why.
- Implement a refitted parametric-bootstrap model-adequacy check. It accepts an
  externally calibrated observation-noise covariance, transforms it to the
  nondegenerate observed-row/background-orthogonal coordinates, computes
  `T_res = r.T @ Sigma_e^{-1} @ r`, simulates from the complete fitted
  source/background model plus that noise, refits each replicate, and compares
  the observed statistic with the empirical null distribution.
- Default to `alpha=0.05`; paper runs use 1,000 refitted bootstrap replicates.
  Return `calibration_status`, `T_res`, bootstrap quantile, add-one Monte Carlo
  `p_value`, `alpha`, replicate count, and noise-model provenance.
- If the noise covariance is absent, invalid, or was estimated from the same
  fitted residual without an independent calibration contract, return
  `calibration_status="uncalibrated"`, null test fields, and no pass/fail
  boolean. Still return raw/projected norms plus sensor-wise, time-wise, and
  autocorrelation summaries.
- Require ensemble provenance. `transport` ensembles may produce uncertainty
  intervals; `inventory` ensembles produce named robustness scenarios. Reject
  any aggregation that pools the two kinds.
- Include warning flags for ill-conditioned matrices or unstable estimates.

**Outputs and artifacts**

- Source-basis coefficient estimates.
- Time-varying source activity estimates.
- Aggregated source apportionment numbers.
- Fitted sensor trajectory.
- Residual metrics.
- Active-set and transport/bootstrap uncertainty intervals.
- Inventory robustness scenarios stored separately from intervals.
- Residual-adequacy result with explicit calibrated/uncalibrated status.

**Acceptance checks**

- Synthetic orthogonal case recovers known source-basis coefficients within tolerance.
- Synthetic temporal case recovers known diurnal traffic and intermittent brick-kiln activity within tolerance.
- Duplicate-column case reports unstable or non-unique activity split.
- Nonnegativity, KKT convergence metadata, and device preservation are enforced
  in the sole solver path.
- Missing noise calibration never yields a model-adequacy pass.
- Correctly specified calibrated simulations are not systematically rejected,
  while a residual-visible omitted signal is detected with increasing strength.
- An omitted signal in `span([H_lag, Q])` demonstrates that non-rejection does
  not establish inventory completeness.
- Transport and inventory ensemble products cannot be pooled.

**Sanity experiment requirements**

Gate S4, completed as part of Task 8, extends the sanity runner with `--gate fit`. Generate synthetic observations from known coefficients:

```text
Y = H_tilde c_true
c_true = [1.5, 0.7, 0.0]
```

Run the nonnegative fitter. The gate must verify:

- In the well-conditioned noiseless case, recovered coefficients match `c_true` within tight tolerance.
- With small Gaussian noise, recovered coefficients remain close and residual norm is lower than the zero-coefficient baseline.
- Nonnegativity is enforced by projection and the final KKT residual satisfies
  its configured tolerance.
- In a duplicate-column case, individual duplicate coefficients may be unstable, but their sum matches the true merged contribution.
- In an ill-conditioned case, fit metadata includes a warning rather than presenting the result as fully stable.
- A mask case confirms the full result restores exact declared zeros and that a
  fitted near-zero unmasked coefficient does not trigger post-hoc column removal.
- An uncalibrated residual case emits summaries but no adequacy decision.
- A synthetic temporal case with a multi-column temporal basis `Phi` recovers a
  known diurnal traffic profile and an intermittent brick-kiln profile within
  tolerance, confirming the source-activity reconstruction
  `theta_k(t) = sum_b c_kb phi_b(t)` and its contribution summaries (diurnal
  mean, active-period fraction, day-to-day totals).
- A calibrated case with a **residual-visible** omitted signal is detected: the
  observed `T_res` exceeds the bootstrap `(1 - alpha)` quantile and inadequacy is
  flagged, while a correctly specified fit is not systematically rejected.
- A calibrated case whose omitted signal lies in `span([H_lag, Q])` is absorbed
  by the fitted source and background terms and is **not** detected: `T_res`
  stays within the bootstrap null and no inadequacy is declared. This
  demonstrates that non-rejection does not establish inventory completeness, in
  contrast with the residual-visible case above.

Suggested pass/fail tolerances:

```text
noiseless_relative_coefficient_error <= 1e-4
noisy_relative_coefficient_error <= 0.1
residual_norm < zero_model_residual_norm
min(c_hat) >= -1e-8
duplicate_pair_sum_error <= 1e-4 in noiseless duplicate case
temporal_relative_activity_error <= 0.1 in noiseless temporal case
residual_visible_omission_inadequate == true
residual_visible_omission_T_res > correctly_specified_T_res
span_absorbed_omission_inadequate == false
span_absorbed_omission_T_res within bootstrap null (<= (1-alpha) quantile)
```

### ~~Task 9: Merge recommendation system~~

**Objective**

Recommend deterministic conservative source reporting groups without silently
replacing the fitted coefficient model or claiming the globally finest
identifiable partition.

**Likely files**

- New `model/iasa/merge.py`
- Update `model/iasa/diagnostics.py`
- Optional: `scripts/run_pol_iasa.py`

**Implementation details**

- Consume the exact `tau_rho`, weak set, eligible ambiguity pairs, and
  source/basis metadata produced by Task 7.
- Add a source-graph edge `(k,k')` when at least one eligible cross-source
  coefficient pair exceeds `tau_rho`.
- Store maximum eligible coherence, minimum ray distance, and the exact
  source--basis trigger pair for each edge.
- Order vertices by the original source index and compute deterministically
  ordered connected components.
- Mark every component result with `is_conservative=true`. Document that an
  `A-B-C` edge chain creates one component even when `A` and `C` are pairwise
  distinguishable; retain both trigger edges so this transitive over-merging is
  inspectable.
- Weak coefficients are flagged independently and never create coherence edges.
- If the matrix is rank deficient but no eligible pair defines an edge, emit a
  global unresolved warning instead of inventing a merge.
- Keep the fine-resolution coefficient fit. Group activity trajectories and
  fitted sensor contributions are sums of component members; do not construct
  an artificial source fingerprint or silently refit a grouped inventory.
- Report:

```text
source-level activity summaries
source-basis coefficient trigger pairs
recommended report components
summed group activity and sensor contributions
weak flags and global unresolved warnings
diagnostic values that triggered each edge
```

- Keep recommendations deterministic for identical tensors and metadata.

**Outputs and artifacts**

- Merge recommendations.
- Summed report-group activities and sensor contributions.
- Per-source flags.

**Acceptance checks**

- Identical source fingerprints are recommended for merge.
- Clearly separated synthetic sources are not merged.
- Weakly visible sources are flagged even if they do not form a high-coherence pair.
- An `A-B-C` chain produces one deterministic conservative component and retains
  both trigger edges and their diagnostic values.

**Sanity experiment requirements**

Gate S5, completed as part of Task 9, extends the sanity runner with `--gate merge`. Use fingerprints and fits from the duplicate and separated cases. The gate must verify:

- Duplicate fingerprints produce one merge recommendation containing exactly the duplicate source pair.
- The edge includes its triggering coefficient pair and diagnostic values.
- Clearly separated synthetic sources are not recommended for merge.
- A weakly visible source is flagged even if it is not connected to a high-coherence pair.
- Grouped activity and sensor-contribution sums match the true total in the
  duplicate synthetic case without a grouped refit.
- Component output sets `is_conservative == true` and makes no `finest`
  guarantee.

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
- The summary includes source/basis names, `c_true`, `c_hat`, coefficient error,
  residual norm, padded singular values, `sigma_J`, numerical/effective rank,
  condition status, weak set, eligible coherence/ray distance, background
  absorption, trigger pairs, report components, solver convergence, device,
  dtype, and response boundary metadata.
- The well-conditioned case recovers source coefficients.
- The duplicate-source case recommends a merge and reports stable merged contribution.
- The fit is exercised on a `H_tilde` produced by the open-boundary response
  builder and background projection (not a hand-constructed matrix), at a
  representative multi-source, multi-basis, multi-sensor scale, confirming that
  projected FISTA converges and recovers coefficients in the response-derived
  regime the experiments use. This closes the gap that Gate S4 only exercises the
  solver on small synthetic matrices.
- Generated artifacts are small and go to `logs/` or `/tmp`, not tracked source paths.

**Gate S7 (adequacy calibration study), completed before Task 10 begins.** Gate
S4 confirms the adequacy check on single instances; Gate S7 confirms it is
statistically calibrated so the p-values the experiments report can be trusted.
Extend the sanity runner with `--gate calibration`. Under a correctly specified,
externally calibrated noise model, simulate a predeclared number of independent
observation sets, refit each, run the refitted parametric-bootstrap adequacy
check, and estimate the empirical rejection rate at `alpha`. The gate must
verify:

- The empirical false-positive (rejection) rate of a correctly specified model is
  close to the nominal `alpha`, within Monte Carlo error for the configured
  number of trials.
- The bootstrap-calibrated `p`-values of correctly specified fits are
  approximately uniform on `[0, 1]` (report a distributional summary, e.g.
  quantiles or a Kolmogorov--Smirnov-style statistic, without tuning to pass).
- A misspecified model with an increasing omitted-signal amplitude yields
  monotonically increasing power (rejection probability rises toward 1).
- The study is deterministic under a fixed seed and records trial count, `alpha`,
  bootstrap replicate count per trial, and noise-model provenance.

Suggested pass/fail tolerances:

```text
correctly_specified_rejection_rate within alpha +/- 2 * sqrt(alpha (1-alpha) / n_trials)
p_value_ks_statistic <= configured_uniformity_threshold
power_increases_monotonically_with_omission_amplitude == true
```

This gate is heavier than S1--S6 (nested bootstrap over many trials); run it on a
GPU allocation with batched refits and a modest per-trial replicate count.

Task 10 should not begin until Gates S1 through S7 pass, except when a gate is
explicitly marked blocked with a documented implementation reason.

### ~~Task 9A: Gridded FieldFormer wind field and transport ensembles~~

**Objective**

Upgrade the New Delhi wind input from the v1 city-level sequence to the
paper-facing gridded FieldFormer field, and add transport ensembles. The paper
(`4.method.tex`, Section "Masked Wind Preparation") specifies a coordinate-query
imputer queried on every response-grid cell and hour.

**Likely files**

- `model/iasa/wind.py`
- `scripts/impute_new_delhi_wind.py`
- Optional: `baselines/fieldformer/` if the coordinate-query imputer is vendored locally
- `model/iasa/response.py` (sampler adapter only)

**Implementation details**

- Convert observed `WD/WS` to transport vectors with the paper convention
  `Ux = -WS*sin(WD*pi/180)`, `Vy = -WS*cos(WD*pi/180)`.
- Supply masked station vectors to a FieldFormer coordinate-query wind imputer
  and query it on every response-grid cell `x_g` and hour `t`:

```text
w_hat_t(x_g) = f_omega(x_g, t; {(z_i, t', u_i, m_i)})   for g = 1..n
```

  producing a gridded field `W_hat in R^(T x n x 2)` rather than a station-only
  or city-averaged sequence.
- Feed the gridded field through the existing `WindSampler`
  `sample(t, position_xy) -> [Vx, Vy]` interface so the Task 5 response builder,
  diagnostics, and fitting APIs are unchanged.
- Convert physical velocity to grid displacement before puff advection using the
  recorded `dt_s`, `dx_m`, `dy_m` scales, and store the scales and convention in
  the response artifact.
- Build wind-field ensembles `{w_hat^(r)}` from held-out-calibrated prediction
  error, station bootstrap resampling, checkpoint ensembles, or
  validation-residual-matched perturbations. Query each member on the same grid,
  convert to displacement, and tag the resulting responses with
  `ensemble_kind="transport"`.
- Keep the v1 city-level provider available for smoke checks under a distinct
  provider label; the gridded field is the default for paper-facing New Delhi
  results.
- Implementation note (2026-07-19): the coordinate-query FieldFormer from the
  sibling repo is vendored inference-only into `baselines/fieldformer/` and
  wrapped by `model/iasa/fieldformer_adapter.py`
  (`FieldFormerCoordinateQueryImputer`) behind the `CoordinateQueryImputer`
  protocol. Because the upstream checkpoints are scalar (`out_dim=1`) and IASA
  wind needs a 2-vector `(Ux, Vy)` (`out_dim=2`), a wind checkpoint must be
  trained to activate it; the default imputer remains the kernel coordinate-query
  interpolator (`KernelCoordinateQueryImputer`) as a fallback. Training that
  checkpoint is Task 9D.
- Validate on real data by masking observed station vectors and measuring
  held-out station-time vector, direction, and speed error; dense city-wide wind
  truth is not assumed.

**Outputs and artifacts**

- Gridded FieldFormer wind product with raw values, masks, station coordinates,
  held-out splits, model/checkpoint metadata, seed, device/dtype, and the
  gridded query field used by the response operator.
- Transport wind-field ensembles tagged `ensemble_kind="transport"`.

**Acceptance checks**

- The gridded field has shape `[T, n, 2]` aligned to the response grid.
- Held-out station-time error is reported over a masked validation split.
- A gridded field drives `build_lagged_response_matrix` through the sampler with
  no change to response/diagnostics/fitting signatures.
- Ensemble members are tagged `transport` and never pooled with inventory
  scenarios.

### Task 9B: Per-sensor footprints and spatial attribution

**Objective**

Implement the per-sensor contribution decomposition and sensor footprints of the
paper (`5.theory.tex`, "Per-Sensor Source Footprints and Spatial Attribution"),
so that each monitor's fitted signal is resolved into contributing source groups
and spatial cells of origin.

**Likely files**

- New `model/iasa/footprints.py`
- Update `model/iasa/reporting.py`
- Update `model/iasa/diagnostics.py` (per-sensor submatrix inheritance note)

**Implementation details**

- Because the response is linear in `c`, the background-corrected fit decomposes
  exactly per observed row `(s,t)`:

```text
y_tilde_{s,t} = sum_{k,b} H_tilde_{Phi,(s,t),(k,b)} * c_hat_{kb}
```

  Report the contribution of source `k` to sensor `s` over the record as
  `Y_hat^{(s)}_k = sum_t sum_b H_lag_{(s,t),(k,b)} * c_hat_{kb}`. Expose both the
  projected (identifiable) and unprojected (raw fitted) forms.
- Construct the sensor footprint as the nonnegative pullback of the sensor's
  response row onto the source grid:

```text
F_{s,t}(i) = sum_{ell in L_t} [ O G^partial_{t,t-ell}(w_hat) ]_{s,i} >= 0,
F_s(i) = sum_t F_{s,t}(i)
```

  weighting by `phi_b(t-ell)`, the inventory map `s_k`, and `c_hat_{kb}` to
  resolve each source group's contribution to each sensor by cell of origin.
  This reuses the Task 5 puff response read backward from the sensor; no new
  transport operator is built.
- Record the inheritance guarantee: for the per-sensor row submatrix
  `H_tilde^{(s)}`, `sigma_J(H_tilde^{(s)}) <= sigma_J(H_tilde)` and
  `rank(H_tilde^{(s)}) <= rank(H_tilde)`, so a single sensor is never more
  identifiable than the pooled network.
- Aggregate per-sensor source shares to the identifiable report groups from
  Task 9 wherever the pooled diagnostics require grouping; never assert a
  per-sensor separation finer than the global resolution.

**Outputs and artifacts**

- Per-sensor source-group contribution tables (projected and raw).
- Nonnegative footprint fields `F_s(i)` over source cells per sensor and source
  group.

**Acceptance checks**

- Per-sensor contributions sum to the fitted sensor-time signal.
- Footprints are nonnegative and localize known upwind source origins on
  controlled trials.
- Per-sensor shares aggregate to report groups and never exceed the pooled
  identifiable resolution.

### Task 9C: Constrained end-to-end refinement (optional)

**Objective**

Implement the optional constrained refinement stage of the paper
(`4.method.tex`, "Constrained End-to-End Refinement"; `6.algorithm.tex`,
"Acceptance Criteria for Refinement") that jointly corrects wind, dispersion,
source, and background coefficients only within physical limits and only when
identifiability is preserved.

**Likely files**

- New `model/iasa/refine.py`
- Update `model/iasa/fit.py`

**Implementation details**

- Initialize at the fixed-response IASA solution (pretrained wind, default
  dispersion). Refine wind-imputer parameters `phi`, dispersion `psi`
  (`sigma_parallel`, `sigma_perp`, minimum dispersion age), source coefficients
  `c >= 0`, and background coefficients `gamma` by minimizing

```text
L_refine = ||Y - H_lag(phi,psi) c - Q gamma||^2
         + lambda_theta R(c)
         + lambda_w ||w_phi - w_phi0||^2
         + lambda_psi ||psi - psi0||^2
         + lambda_sm R_sm(w_phi)
```

  subject to `c >= 0`, `psi in Psi_phys`, and
  `||w_phi - w_phi0||_inf <= eps_w`.
- Accept the refined response only if it does not degrade separability:

```text
sigma_J(H_tilde_ref) >= (1 - eta_id) * sigma_J(H_tilde_0)
max_{i!=j, i,j not in W} rho_ij^ref <= tau_rho^ref   (default tau_rho^ref = tau_rho)
```

  If either response has fewer than `J` numerically nonzero singular values, its
  `sigma_J` is taken as zero, so refinement cannot be accepted by exploiting a
  rank-deficient response.
- Report the refined solution with its own response diagnostics, and keep the
  fixed-response solution as the default reported estimate. Refinement is an
  optional stage, not the primary estimator.

**Outputs and artifacts**

- Refined wind/dispersion/source/background estimates with the refinement
  acceptance decision and pre/post identifiability diagnostics.

**Acceptance checks**

- Refinement stays within `eps_w` and `Psi_phys`.
- A refinement that lowers `sigma_J` below the acceptance threshold or raises
  eligible coherence above `tau_rho^ref` is rejected.
- The fixed-response estimate remains available and is the default report.

### Task 9D: Train the FieldFormer 2-vector wind checkpoint

**Objective**

Train the coordinate-query FieldFormer to impute the 2-vector transport wind
field `(Ux, Vy)` for New Delhi, producing the checkpoint that activates the
`FieldFormerCoordinateQueryImputer` vendored in Task 9A. Task 9A wired the
adapter behind the `CoordinateQueryImputer` protocol but shipped only the kernel
fallback, because the upstream FieldFormer checkpoints are scalar (`out_dim=1`)
pollution/heat/SWE fields and no 2-vector wind checkpoint exists. This task
removes that gap so the paper-facing New Delhi runs (and Task 10's observed-mode
wind) can use the learned coordinate-query field rather than the kernel
interpolator.

**Likely files**

- New `scripts/train_fieldformer_wind.py` (or `experiments/fieldformer_wind/`)
- `baselines/fieldformer/model.py` (vendored model; training-only helpers may be
  added alongside, kept separate from the inference module)
- `model/iasa/fieldformer_adapter.py` (no API change; default-imputer switch only
  after validation)
- New checkpoint artifact, e.g. `data/fieldformer_wind_new_delhi.pt`

**Implementation details**

- Build supervision from the government `WD/WS` record: convert to transport
  vectors with `transport_vectors_from_wd_ws` (paper eq. wind_direction_conversion),
  preserve the observation mask, and map station lon/lat to response-grid
  coordinates exactly as `gridded_new_delhi_wind_field`. Supervision is the sparse
  observed station-time tuples `(x, y, t) -> (Ux, Vy)`; there is no dense
  city-wide wind truth.
- Train `FieldFormerCoordinateQuery` with `out_dim=2` under the **same
  conventions the adapter queries with**: min-max `xy` normalization to `[0, 1]`,
  time `t/(T-1)`, and identical `k_neighbors`/`time_radius`. The checkpoint and
  the adapter must agree on these or the learned attention is queried
  off-distribution (see `baselines/fieldformer/README.md`).
- Loss is supervised regression (Huber or MSE) on held-in observed tuples; the
  upstream scalar-field physics regularizers (sponge/radiation) are not assumed
  to transfer to wind and are opt-in only if justified. Hold out a station-time
  split for validation; never read held-out tuples as neighbors
  (`allowed_indices`).
- Record full provenance: seed, config, data hashes, normalization, `out_dim=2`,
  `k_neighbors`/`time_radius`, EMA setting, device/dtype. Save a checkpoint
  loadable by `load_fieldformer_checkpoint(out_dim=2)` and
  `build_fieldformer_wind_imputer(checkpoint_path=...)`.
- Validate on the held-out split with `evaluate_gridded_wind_heldout` (vector
  RMSE, circular direction error, speed error) and compare against the kernel
  interpolator and a city-mean baseline on the identical split. The learned
  imputer must add value on held-out error to justify replacing the kernel
  default; if it does not, keep the kernel default and record why.
- Only after the checkpoint passes held-out validation, switch the paper-facing
  `gridded_new_delhi_wind_field` default imputer to FieldFormer; the kernel
  imputer remains available as a labeled fallback and for smoke checks.
- Run training on a SLURM GPU node (container + overlay); training is the heavy
  step, so record wall-clock, checkpoint size, and reproducibility command.

**Outputs and artifacts**

- Trained 2-vector `(Ux, Vy)` FieldFormer wind checkpoint with recorded config,
  seed, data hashes, and normalization/neighbor settings.
- Held-out station-time validation report (vector/direction/speed error) with the
  kernel and city-mean baselines on the same split.
- Transport wind-field ensembles from the trained model (checkpoint ensembles,
  station bootstrap, or validation-residual-matched perturbations), tagged
  `ensemble_kind="transport"`.

**Acceptance checks**

- A `out_dim=2` checkpoint trains, saves, and loads via
  `load_fieldformer_checkpoint`/`build_fieldformer_wind_imputer` and drives
  `build_gridded_wind_field` end to end.
- Training and query use identical normalization and `k_neighbors`/`time_radius`,
  recorded in provenance so the checkpoint and adapter are consistent.
- Held-out station-time error is reported and compared to the kernel and
  city-mean baselines on the same masked split; the FieldFormer default is
  adopted only if it improves held-out error, otherwise the kernel default and
  the reason are recorded.
- The kernel imputer remains available as a labeled fallback; no untrained or
  unvalidated model is presented as a scientific wind product.
- Depends on Task 9A (vendored adapter). Should complete before Task 10's
  paper-facing observed-New-Delhi wind, though Task 10 may run on the kernel
  fallback and be upgraded once the checkpoint validates.

### Task 10: Controlled experiment suite

**Objective**

Implement the paper's Experiment 1--10 controlled matrix on the single New Delhi
platform after Gates S1--S6 pass. The paper (`7.evaluation.tex`) fixes one
platform -- one regulatory sensor geometry, one PM2.5/wind record, one set of
proxy inventories -- and `8.evaluation.tex` varies exactly one controlled axis
per experiment. This task implements that platform and the ten experiments;
Task 11 produces the matching result tables.

**Likely files**

- New `experiments/iasa_pol/` with `configs/`, `run_experiment.py`, `summarize_results.py`
- New scripts under `scripts/` or `experiments/iasa_pol/`
- `sim/polsim.py` retained and labeled as the Experiment 5 structural
  edge-hold PDE generator
- New IASA data loading and evaluation utilities if experiment-specific loaders/metrics are needed

**Implementation details**

Define the single shared New Delhi base platform, then implement one-factor
experiments. Every wind, source, background, and geometry comparison must use
identical source, temporal-basis, sensor, observation-mask, and background
columns on both sides. Real imputed New Delhi wind is used as a controlled
transport input with synthetic coefficients; the observed-PM2.5 mode is a
separate evaluation and is never assigned synthetic recovery metrics.

Implement the paper's ten controlled experiments:

- **Experiment 1 -- conditioning predicts recovery.** Vary source geometry and
  Gaussian observation noise at 0%, 1%, 5%, 10%, 20% of the maximum clean sensor
  signal. Compare coefficient and reconstructed-activity error against
  `sigma_J`, numerical/effective rank, condition number, and visibility.
- **Experiment 2 -- coherent sources require grouped reporting.** Form
  increasingly coherent source pairs by shifting a copy of one inventory map by a
  controlled spatial offset (smaller shift drives coherence toward one). Compare
  individual-source error with summed activity/sensor-contribution error of each
  recommended connected component; retain the triggering source--basis pair.
- **Experiment 3 -- background correction can help or hurt.** Compare no
  background, the primary rank-four basis (constant, centered linear trend, first
  daily sine, first daily cosine), a redundant-column basis with the same span,
  and a labeled source-like stress basis. Audit through visibility and the
  removed/raw absorption ratio. Every basis is declared before fitting; none is
  selected from `Y` or its recovery score.
- **Experiment 4 -- wind diversity and sensor geometry change resolution.**
  Evaluate matched columns under constant, single-direction, diurnal, AR(1),
  multi-directional, and real imputed New Delhi wind, crossed with sensor
  layouts: regulatory, seeded random, downwind-focused, and layouts selected to
  increase `sigma_J` or reduce maximum eligible coherence. Add separate
  historical and simulated wind-window ensembles for the 5th/50th/95th
  percentiles of `sigma_J`, rank probabilities, coefficient-weakness
  probabilities, pairwise-ambiguity probabilities, and conservative-component
  frequencies.
- **Experiment 5 -- transport error is amplified by ill-conditioning.** Use two
  kinds of transport error: (a) *parametric* perturbations of wind speed,
  direction, and dispersion within the puff family, and (b) a *structural*
  mismatch that generates observations with the auxiliary edge-hold
  advection--diffusion PDE (`sim/polsim.py`: first-order upwind advection,
  five-point Laplacian, two-stage Heun, diffusivity `3e-4`, edge-hold
  boundaries) while fitting with the open-boundary puff response. Report operator
  error norm, coefficient/activity error, residual, and singular spectrum
  together. Parametric transport ensembles may support transport-uncertainty
  intervals; the structural case additionally drives the residual-adequacy test
  and reports its rejection rate.
- **Experiment 6 -- inventory error changes the attribution target.** Perturb
  inventory locations, spatial scales, category assignments, and map versions
  with transport held fixed. Each alternative inventory is a separate robustness
  scenario, never pooled with Experiment 5 transport error and never reported as
  a confidence interval.
- **Experiment 7 -- lag-window sensitivity.** Derive the candidate grid from
  physical travel/residence times. For adjacent candidates compute
  `||H(L+delta)-H(L)||_F / max(||H(L+delta)||_F, eps)`, select the smallest `L`
  at or below `tau_L=1e-3`, and report rank, conditioning, and report-component
  stability across the grid. The observation row count stays fixed and fitted
  coefficients are never used for selection.
- **Experiment 8 -- missing-source model adequacy.** Omit either a
  residual-visible source outside `span([H_lag, Q])` or an aligned source whose
  signature lies inside that span. Evaluate the refitted parametric-bootstrap
  test (independently declared Gaussian noise covariance, 1,000 replicates,
  `alpha=0.05`) for calibration and power. The aligned case is a required
  negative control demonstrating that non-rejection cannot certify inventory
  completeness.
- **Experiment 9 -- temporal-basis recovery.** Reconstruct known traffic
  diurnal, brick-kiln intermittent, industry day/night, and mixed source--basis
  coefficients at increasing noise, separating coefficient error from
  reconstructed activity-trajectory error.
- **Experiment 10 -- per-sensor footprints and spatial attribution.** On
  controlled trials with known source origins, verify the Task 9B footprints
  localize the responsible upwind cells and that per-sensor contributions sum to
  the fitted sensor signal. On observed New Delhi data, report per-monitor fitted
  source-group contributions and their spatial cells of origin, aggregated to the
  identifiable report groups.

The experiments map to the paper's hypotheses as follows: Experiment 1 -> H1
(conditioning predicts recovery), Experiment 2 -> H2 (coherent sources merged),
Experiment 3 -> H3 (background helps or hurts), Experiment 4 -> H4 (wind
diversity and geometry change resolution), Experiment 5 -> H5a (transport error),
Experiment 6 -> H5b (inventory robustness). Do not encode the expected direction
of any hypothesis as an empirical conclusion before results exist.

Also implement the observed-New-Delhi study mode (real PM2.5, its observation
mask, gridded imputed wind, named normalized inventories, declared proxy temporal
bases) with no ground-truth source activities; it reports residuals, geometry,
uncertainty, weak/ambiguous coefficients, normalized proxy contributions, and
recommended report groups rather than recovery error.

**Outputs and artifacts**

- Configured Experiment 1--10 runs plus the observed New Delhi run.
- Saved `H_lag`, `H_tilde`, masks, diagnostics, fits, uncertainty, report
  components, per-sensor footprints, lag sweeps, adequacy results, inventory
  scenarios, wind-distribution summaries, device/dtype, and evaluation outputs.
- Summary CSV or JSON tables for analysis.

**Acceptance checks**

- Each of Experiments 1--10 has at least one runnable one-factor configuration.
- Results include both attribution accuracy and identifiability diagnostics.
- Runs are reproducible from saved config and seed.
- The primary `Q`, lag rule, fixed-zero mask, and inventory version are
  recoverable from provenance and are not selected from final fit quality.
- Transport and inventory ensembles remain type-separated through aggregation.
- The Experiment 5 structural generator is labeled `edge_hold_pde` and is never
  silently substituted for the open-boundary puff response.

### Task 11: Evaluation and reporting

**Objective**

Produce separate controlled-ground-truth and observed-New-Delhi reports matching
the paper's result placeholders.

**Likely files**

- New `evaluation/eval_pol_iasa.py`
- New `model/iasa/reporting.py`
- `model/iasa/footprints.py` (from Task 9B) for per-sensor spatial attribution tables
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

- Padded singular values, `sigma_J`, numerical/effective rank, and condition
  status.
- Visibility, weak set, eligible coherence/ray distance, absorption, trigger
  pairs, global warnings, deterministic conservative report components, and
  `is_conservative=true`.
- Every lag candidate, convergence ratio, selected lag, `tau_L`, physical-grid
  rationale, and diagnostic/group stability across the lag sweep.
- Primary/sensitivity background-basis identifiers and confirmation that they
  were fixed independently of `Y` and recovery results.
- Fixed-zero coefficient mask plus original/reduced index mappings.
- Uncertainty interval coverage when synthetic ground truth is known.
- Residual `calibration_status`, `T_res`, bootstrap quantile, `p_value`,
  `alpha`, replicate count, noise-model provenance, and sensor/time/ACF
  summaries. Never serialize an adequacy pass for an uncalibrated run.
- Correlation between diagnostics and attribution error across experiment sweeps.
- Per-sensor footprint localization error against known source origins
  (controlled trials) and the per-sensor contribution reconstruction, verifying
  contributions sum to the fitted sensor signal.
- Held-out New Delhi wind-imputation validation: station-time vector error,
  direction/speed error, mask coverage, query-grid resolution, checkpoint, seed,
  and device/dtype metadata.
- Real New Delhi summaries from observed-only PM2.5 rows, imputed `WD/WS`, named
  inventories, and temporal activity bases, without source-ground-truth error.
- Normalized proxy coefficient/activity tables and sensor-space source/group
  contribution tables with active-set and transport/bootstrap intervals.
- Per-monitor fitted source-group contributions and their spatial cells of
  origin, aggregated to the identifiable report groups.
- Inventory-version scenario tables kept separate and labeled as robustness,
  not confidence intervals.
- Wind-distribution tables containing 5th/50th/95th percentiles of `sigma_J`,
  full numerical/effective-rank probabilities, coefficient weakness
  probabilities, source-pair ambiguity probabilities, and component frequencies.
- Raw and projected residuals, fitted trajectories, observation count, response
  provenance, solver convergence, device, and dtype.
- If percentages are shown, label them as fractions of the fitted
  inventory-attributed sensor signal, never physical-emission shares.

Produce tables matching the paper's evaluation result subsections
(`8.evaluation.tex`):

- Controlled result tables, one per Experiment 1--10 (conditioning/recovery;
  coherence/grouped reporting; background stress; wind diversity and sensor
  geometry; transport error; inventory robustness; lag-window selection;
  missing-source adequacy; temporal-basis recovery; per-sensor footprints).
- Observed New Delhi result tables: wind-imputation validation; identifiability
  and report groups; proxy apportionment and uncertainty; sensor fit and residual
  diagnostics.
- Human-readable run summaries and machine-readable result files.
- New Delhi tables that clearly separate identifiable source-level contributions from merged or ambiguous groups.
- Experiment 5 transport-uncertainty and Experiment 6 inventory-robustness tables
  that cannot be mistaken for one combined interval.

**Outputs and artifacts**

- IASA evaluation script.
- Result tables for each experiment family.

**Acceptance checks**

- Evaluation works on a single saved IASA run.
- Evaluation aggregates multiple runs into one summary table.
- Grouped metrics are reported whenever a non-singleton report component exists.
- New Delhi smoke run emits an apportionment report without requiring ground-truth source activities.
- Per-sensor contribution and footprint tables are produced, with contributions
  summing to the fitted sensor signal and shares aggregated to report groups.
- Undefined weak-pair metrics remain `null` through single-run and aggregate
  reporting.
- An `A-B-C` chain retains both trigger edges in the aggregate report.
- Omitted-source results state that rejection diagnoses model inadequacy without
  identifying its cause and that non-rejection cannot certify completeness.

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
2. Load and impute New Delhi `WD/WS` into a gridded FieldFormer wind field
   queried per response-grid cell, or choose a synthetic wind provider; build
   transport wind-field ensembles as needed.
3. Build temporal activity bases.
4. Declare the primary Q, lag-candidate grid, inventory version, and any
   fixed-zero coefficient mask.
5. Generate or load observations.
6. Build H_lag, select lag by response convergence, and preserve the sweep.
7. Build Q and project to H_tilde.
8. Run realized-wind and, when requested, wind-distribution diagnostics.
9. Fit nonnegative source-basis coefficients and check residual adequacy when a
   calibrated noise model is available.
10. Optionally run constrained end-to-end refinement, accepting it only if it
    does not degrade identifiability.
11. Report source contributions, per-sensor contributions and footprints,
    observation/transport uncertainty, inventory robustness, and conservative
    merge recommendations.
12. Evaluate controlled Experiments 1--10 and real New Delhi apportionment.
```

- State that pandas/NumPy are ingestion and serialization tools while response,
  projection, diagnostics, fitting, covariance, and ensemble computation use
  PyTorch on an explicit device/dtype.
- Add commands for wind-product generation, Gates S1--S6, controlled sweeps,
  observed-New-Delhi evaluation, aggregation, and paper compilation.
- Document container image, PyTorch/CUDA versions, device model, deterministic
  settings, seeds, generated-product policy, checkpoint provenance, response
  provenance, inventory hashes/versions, lag protocol, coefficient masks,
  background preregistration, ensemble-kind semantics, noise-model provenance,
  and diagnostic/adequacy/grouping result schemas.
- Document that identifiability certificates are conditional on the declared
  inventory, transport, temporal basis, lag, background basis, observation
  mask, and noise assumptions. State that a calibrated residual rejection shows
  model inadequacy but not its cause, while non-rejection does not establish
  inventory completeness.
- Document that connected components are conservative deterministic report
  groups rather than a guaranteed finest partition, and that inventory
  scenarios are never confidence-interval draws.
- Do not list SciPy as a required solver dependency.

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
- Documentation names supported CPU/CUDA devices and dtype defaults.
- Documentation does not present archived legacy code as maintained, required,
  or available in clean checkouts.
- Documentation exposes all new artifact fields and reproduces the paper
  defaults `tau_L=1e-3`, `alpha=0.05`, 1,000 adequacy bootstrap refits, and
  wind quantiles 0.05/0.50/0.95.

## 4. Test Plan

The implementation should add tests or smoke checks as the IASA modules are introduced.

In addition to code-level tests, Tasks 5 through 9 must run the tiny sanity experiment gates defined above. These gates are deliberately smaller than paper experiments: they should use synthetic sources, synthetic wind, short horizons, and known coefficients so failures identify conceptual or algorithmic mistakes early.

Required sanity-gate ordering:

```text
Task 5 complete -> Gate S1 response sanity passes
Task 6 complete -> Gate S2 projection sanity passes
Task 6A complete -> PyTorch CPU/CUDA parity and device-preservation checks pass
Task 7 complete -> Gate S3 diagnostic sanity passes
Task 8 complete -> Gate S4 fitting sanity passes (incl. temporal recovery and
                   residual-visible / span-absorbed adequacy cases)
Task 9 complete -> Gate S5 merge sanity passes
Before Task 10 -> Gate S6 minimal end-to-end IASA sanity passes
Before Task 10 -> Gate S7 adequacy calibration study passes
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
H_selected.shape == (num_observed_pm25_rows, num_source_basis_columns)
H_tilde.shape == H_selected.shape
c_hat.shape == (num_source_basis_columns,)
theta_hat_t.shape == (num_sources, num_times)
Y_tilde.shape == (num_observed_pm25_rows,)
```

- One-hot source-basis activity should match the corresponding open-boundary response column after projection and baseline subtraction.
- Empty background basis should be a no-op.
- PM2.5 masking should remove identical ordered rows from observations,
  responses, backgrounds, and metadata; a mismatch must fail.
- `WD/WS -> Vx/Vy` conversion should pass cardinal-direction cases with the documented meteorological convention.
- CPU and CUDA paths should preserve requested device/dtype and agree within
  recorded tolerances.

### Numerical diagnostics tests

- Duplicate columns should produce rank deficiency, `sigma_J == 0`, and high
  coherence.
- Orthogonal synthetic columns should produce low coherence.
- Near-zero columns should produce weak flags and `null` pairwise metrics.
- Increasing background basis flexibility should not increase projected source visibility.
- Synthetic one-direction wind should produce lower or equal identifiability scores than a matched multi-direction wind case in a controlled layout where wind diversity is designed to reveal source separation.
- A predeclared fixed-zero mask removes exactly its columns, preserves both
  index mappings, and recomputes diagnostics on the reduced matrix. A fitted
  near-zero coefficient does not alter diagnostics.
- Fixed historical/simulated wind inputs and seeds produce deterministic
  quantiles, rank/weakness/ambiguity probabilities, and component frequencies.

### Fitting tests

- Recover known nonnegative activities in a well-conditioned synthetic case.
- Recover known diurnal traffic activity and intermittent brick-kiln activity in a synthetic temporal-basis case.
- Preserve nonnegative estimates, KKT convergence metadata, and device placement
  in the PyTorch projected-FISTA solver.
- Flag instability in a high-coherence or rank-deficient case.
- Reject any attempt to pool `transport` and `inventory` ensemble members.

### Lag and model-adequacy tests

- Lag selection returns the smallest candidate satisfying `tau_L=1e-3`, keeps
  the observation row count fixed, and returns every convergence and stability
  diagnostic.
- A calibrated correctly specified simulation is not systematically rejected
  at `alpha=0.05` over repeated trials.
- Detection probability increases with the strength of an omitted signal
  outside `span([H_lag, Q])`.
- An omitted signal in `span([H_lag, Q])` exercises the documented
  residual-invisibility limitation.
- A missing or invalid external noise covariance produces `uncalibrated`, null
  test fields, residual summaries, and no pass/fail decision.

### Reporting-group tests

- An `A-B-C` ambiguity chain yields one deterministic component with
  `is_conservative=true` and both trigger edges retained, even when `A-C` is not
  an edge.
- Clearly separated sources remain separate and weak sources remain independent
  flags rather than invented edges.

### Gridded wind tests

- The FieldFormer imputer queried on the response grid returns a field of shape
  `[T, n, 2]` aligned to the grid cells.
- Held-out station-time vector, direction, and speed error are reported over a
  masked validation split.
- A gridded field drives `build_lagged_response_matrix` through the `WindSampler`
  interface with no change to response/diagnostics/fitting signatures.
- Wind-field ensemble members are tagged `ensemble_kind="transport"` and are
  rejected from any pooling with inventory scenarios.

### Per-sensor footprint tests

- Per-sensor contributions sum to the fitted sensor-time signal.
- Footprint fields are nonnegative and localize a known upwind source origin on a
  controlled trial.
- Deleting rows to a single sensor never increases `sigma_J` or rank
  (`sigma_J(H_tilde^(s)) <= sigma_J(H_tilde)`), and per-sensor shares aggregate to
  report groups.

### Constrained-refinement tests

- Refinement stays within the `eps_w` wind-drift cap and the physical dispersion
  set `Psi_phys`.
- A refinement that lowers `sigma_J` below `(1 - eta_id) * sigma_J(H_tilde_0)` or
  raises eligible coherence above `tau_rho^ref` is rejected.
- The fixed-response estimate remains available and is the default reported
  result.

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
- Verify the summary contains source activities, residuals, padded singular
  values, `sigma_J`, visibility, eligible coherence/ray distance, trigger pairs,
  conservative report components, solver convergence, lag selection,
  coefficient mask/index mapping, primary Q, residual calibration status,
  separated ensemble outputs, wind-distribution summary, device/dtype, and
  response/inventory/noise provenance.
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
  footprints.py     # per-sensor contribution decomposition and spatial footprints
  refine.py         # optional constrained end-to-end refinement
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
baselines/fieldformer/
```

The paper's wind imputer is FieldFormer, a coordinate-query model queried on the
response grid (`4.method.tex`). Use this path only if that implementation is not
already available elsewhere in the project. The roadmap should document whether
the code is vendored, referenced as an external dependency, or reimplemented
locally.

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
- Do not impute `pm25`; apply its observed-row selection identically to `Y`,
  `H_lag`, `Q`, and row metadata.
- Use imputed real `WD/WS` as the default wind input for the New Delhi apportionment workflow.
- Use synthetic wind providers for controlled identifiability experiments.
- The first IASA version should use temporal-basis coefficients, not unconstrained `theta_k(t)` at every timestamp.
- Traffic should have a diurnal activity basis, brick kilns should support seasonal or intermittent activity, and industry should support day-to-day or slowly varying activity.
- The response-matrix builder should support fixed supplied `Vx/Vy` sequences, imputed New Delhi wind, and synthetic wind providers.
- Paper-facing `H_lag` construction should prefer the dedicated open-boundary puff/plume response implementation over minimal reuse of the current edge-hold PDE simulator.
- A city-level imputed wind sequence is acceptable for the v1 open-boundary
  response path if the metadata documents that choice, but the paper-facing New
  Delhi response uses the gridded FieldFormer wind field queried per response-grid
  cell (Task 9A). The city-level sequence and the gridded field share the same
  `WindSampler` interface, so upgrading does not change response, diagnostics, or
  fitting APIs.
- Wind-imputation uncertainty is propagated through transport wind-field
  ensembles tagged `ensemble_kind="transport"`; these may support uncertainty
  intervals and are never pooled with inventory scenarios.
- Per-sensor footprints and per-sensor contributions are spatial overlays of the
  same global fit; per-sensor source shares are asserted only at the globally
  identifiable resolution and aggregate to report groups where the pooled
  diagnostics require it.
- Constrained end-to-end refinement is optional; the fixed-response estimate is
  the default report, and refinement is accepted only if it does not degrade
  `sigma_J` or raise eligible coherence above `tau_rho^ref`.
- Background projection should be implemented before diagnostics and fitting, because the new paper treats `H_tilde` as the central object.
- After ingestion, use PyTorch for response, projection, diagnostics, fitting,
  covariance, and ensemble computation. SciPy is not a required dependency.
- Preserve explicit device and dtype metadata; avoid implicit CPU transfers.
- Merge recommendations should be reported as recommendations, not silently applied to source-level estimates.
- Connected components are deterministic conservative report groups and are not
  claimed to be the globally finest identifiable partition.
- Identifiability diagnostics are uniform over the full predeclared admissible
  coefficient set. Do not infer tangent-cone identifiability from fitted zeros;
  only a scientifically declared pre-fit fixed-zero mask may reduce columns.
- Select lag before coefficient fitting with the physical candidate grid and
  default response-convergence threshold `tau_L=1e-3`.
- The paper-facing residual adequacy test uses `alpha=0.05` and 1,000 refitted
  parametric-bootstrap replicates. Without an externally calibrated noise model,
  status is `uncalibrated` and no adequacy decision is allowed.
- Wind-distribution summaries use quantiles 0.05, 0.50, and 0.95.
- Transport ensembles may support uncertainty intervals; inventory alternatives
  are named robustness scenarios and must never be pooled into those intervals.
- Independently normalized inventory coefficients are normalized proxy units,
  not physical emission totals or physical source shares.
- Any generated large matrices or experiment outputs should go under `logs/` or a clearly named generated-data path, not be committed by default.
- If the FieldFormer code is not present in this repo, adding or adapting it is required before the real New Delhi apportionment experiment can be considered complete.

## 7. Implementation Milestones

### Milestone A: Minimal IASA core

Complete Tasks 1 through 8, including Tasks 3A and 6A, and sanity Gates S1
through S4.

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

- Fit output includes diagnostics, separated uncertainty/robustness where
  available, residual calibration status, source flags, and conservative merge
  recommendations.
- One command can run a minimal source-apportionment report.
- New Delhi smoke run produces an apportionment report using imputed wind.
- Toy merge and end-to-end IASA sanity gates pass before broad controlled experiments begin.

### Milestone C: Paper-style experiments

Complete Tasks 9A, 9B, 9C, 10, and 11.

Deliverable:

- Gridded FieldFormer wind field and transport ensembles drive the paper-facing
  New Delhi response.
- Per-sensor footprints and spatial attribution are available for controlled and
  observed runs.
- Optional constrained refinement is available and gated on identifiability.
- Controlled Experiment 1--10 runs cover conditioning/recovery, coherence
  grouping, background stress, wind diversity and geometry, transport error,
  inventory robustness, lag selection, missing-source adequacy, temporal-basis
  recovery, and per-sensor footprints, plus realized/distributional wind
  diagnostics.
- Results tables connect attribution error to response-matrix geometry.
- New Delhi tables report source-level and merged-source apportionment numbers.

### Milestone D: Documentation and cleanup

Complete Task 12.

Deliverable:

- The README and docs identify the new IASA workflow as the main implementation of `STAMP_new.pdf`.
- Archive policy is documented as non-contract local reference storage only.
- No README or roadmap command depends on archived legacy files.
