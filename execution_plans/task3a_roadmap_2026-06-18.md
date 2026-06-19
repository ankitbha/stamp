# Roadmap Update Plan: IASA-Only Clean Slate With Non-Contract Archive

## Summary

Update `ROADMAP_STAMP_NEW.md` so the repository target is explicitly IASA-only. Legacy files should be removed from active tracked paths by moving them into `archive/`, with `archive/` treated as a local, ignored, non-contract holding area that code, tests, and documentation must not depend on.

## Key Roadmap Changes

- Revise the opening and Current Codebase Map to say active support is limited to IASA pollution and source-apportionment work.
- Add an Archive Policy:
  - legacy files removed from active paths move under `archive/`;
  - `archive/` is ignored, untracked, and outside the repository contract;
  - no active code, tests, roadmap tasks, or README commands may import from or rely on it.
- Add a Clean Slate Baseline task after Task 3:
  - move old heat/SWE modules, pollution free-field generators, evaluators, tuners, notebooks, prior/calibrator code, and ablation simulators into `archive/`;
  - remove stale imports and references from tracked active files;
  - verify the tracked repository has no broken dependencies on archived files.
- Update Task 4 to require new IASA primitives instead of extending legacy simulators:
  - inventory activity construction;
  - wind provider interface;
  - source combination from named inventories;
  - a minimal IASA rollout or response-input path.
- Make `model/iasa/` the core implementation area, covering wind, activity, response, background, projection, diagnostics, fitting, merge logic, and reporting.
- Expand the test plan with clean-slate checks:
  - every tracked Python module imports;
  - no active references remain to `S_unknown`, `load_known_sources_40x40`, `rollout_pollution`, old heat/SWE APIs, old SimGrad entrypoints, or `archive/`;
  - source/weather tests and a minimal IASA sanity runner pass.
- Update Milestone A to require a coherent IASA skeleton and one tiny end-to-end sanity run before broader experiments.

## Acceptance Criteria

- The roadmap clearly says legacy files move to `archive/` rather than remaining maintained in active paths.
- The roadmap says `archive/` is non-contract and may be absent from clean checkouts.
- No roadmap task instructs future code to import from, test against, or document commands using `archive/`.
- Active repository success means tracked imports pass, legacy references are gone, and the minimal IASA sanity path runs.

## Assumptions

- Removed means removed from tracked active paths, with optional local copies under ignored `archive/`.
- This roadmap update is documentation-only.
- The destination is an IASA-only repository without legacy compatibility requirements.
