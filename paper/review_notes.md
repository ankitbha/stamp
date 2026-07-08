# Review Notes — *Identifiability-Aware Source Apportionment in City-Scale Advection-Diffusion Systems*

Bhardwaj & Subramanian (Preprint, June 20, 2026)

Each note below records **(a)** the location and a short quote of the paragraph/equation the handwritten annotation points to, and **(b)** the transcribed annotation. Notes are numbered sequentially (N1, N2, …) and grouped by paper section. Reviewer shorthand has been expanded for readability (e.g. "b/w" → "between", "eqn" → "equation").

---

## Abstract & Section 1 (Introduction)

**N1 — Abstract, closing sentence**
Context: "…reports uncertainty and conservative deterministic **grouping recommendations for indistinguishable sources**."
Annotation: (underlined) Flagged phrase — grouping recommendations for indistinguishable sources.
Status: addressed.

**N2 — Abstract / framing**
Context: general framing of the abstract.
Annotation: "This part also feels a bit weird ⇒ Have to say that heterogeneous studies have distinct findings."
Status: addressed.

---

## Section 2 (Related Work)

**N3 — Receptor vs. source-oriented methods, India ambiguity**
Context: "This ambiguity is especially visible in settings such as India, where studies can produce divergent source-contribution estimates for the same city…"
Annotation: This is the paragraph the "heterogeneous studies have distinct findings" comment (N2) is really about — make the point explicit here.
Status: addressed.

**N4 — Constant-activity model sentence**
Context: "The familiar constant-activity model is recovered by using one constant basis function per source."
Annotation: (underlined) Note that this reduces to recovering the average / constant source activity — tie back to the control/activity model.
Status: addressed.

**N5 — "Our work is complementary to this literature…"**
Context: end of the air-pollution apportionment paragraph.
Annotation: "Break in prose. Have to talk about the different [reasons] the approaches [differ] first."
Status: addressed.

**N6 — Data Assimilation & Inverse Problems / Physics-Informed ML**
Context: "…these approaches typically assume that the underlying system is sufficiently observable…" and the PINN paragraph.
Annotation: "Repetitive prose from previous section. Both methods are talking about the identifiability assumption. Also: this part can be used to talk about the PyTorch backend — and to justify why this and the earlier paragraph even belong in the paper."
Status: addressed.

**N7 — Spatio-Temporal Models paragraph**
Context: "Graph neural networks (GNNs), transformers, and sequence models have been widely used…"
Annotation: "Need to justify if this part even should be here, and whether we should be talking about the inverse-problem solvers and their limitations. A 'just-random' model doesn't make much sense."
Status: addressed.

**N8 — Observability and System Identification paragraph**
Context: "…we focus on the regime of sparse spatial sensing, where the observation operator induces a low-dimensional projection…"
Annotation: "We are bringing this perspective to the source-apportionment problem. Also add measurements / diagnostics that make sense, and a corresponding instantiation for a real-world system to inform the source-apportionment approach." Also: "Doesn't seem relevant here — talk about it in the background."
Status: addressed.

---

## Section 3 (Problem Setup)

**N9 — Shared temporal basis, Eqs (5)–(6)**
Context: "We represent time-varying source activity with a shared low-dimensional temporal basis … θ_k(t) = Σ_{b=1}^{B} c_{kb} φ_b(t)."
Annotation: "The temporal basis is **shared** between sources ⇒ the basis function is the same and shared; only the coefficients change. This is equivalent to something like a day–night shift, or a weekday–weekend shift, etc. → Requires better explanation here on this specific choice."
Status: addressed.

**N10 — Sensor response, Eq (9)**
Context: "h^lag_{t,kb} = Σ_{ℓ∈L_t} φ_b(t−ℓ) O G^∂_{t,t−ℓ}(w_{t−ℓ:t}) s_k."
Annotation: "The equations should come right after s_k, with φ(t−ℓ) shown right there."
Status: addressed.

**N11 — Lag-selection ratio, Eq (11)**
Context: "η_L = ‖H^lag_Φ(L+Δ) − H^lag_Φ(L)‖_F / max{‖H^lag_Φ(L+Δ)‖_F, ε}."
Annotation: "Seems to be a measure of the increase in relative fingerprint → explain what ‖·‖_F (the F-norm) is. Also: if we already have the method of picking L (max lag), then why should rank, conditioning, and ambiguity components be reported across the whole candidate grid?"
Status: addressed.

**N12 — Background/temporal basis Q, Eq (12)**
Context: "We represent these components with a low-dimensional basis Q ∈ ℝ^{mT×r}."
Annotation: "The low-dimensional Q should explain that it prevents source fingerprints from being absorbed into it."
Status: addressed.

**N13 — Vectorization / source activity, Eq before (13)**
Context: "We vectorize C in source-major, basis-minor order as c = vec_src(C)…" and "Y = H^lag_Φ c + Qγ + E."
Annotation: (label) "source activity"; check "ref" the equation.
Status: addressed.

---

## Section 4 (Wind-Conditioned Response Construction and Estimation)

**N14 — WD/WS → transport vectors, Eq (18)**
Context: "U_x = −WS sin(WD π/180), V_y = −WS cos(WD π/180)."
Annotation: "This seems to be incorrect." (verify the direction/sign convention)
Status: addressed.

**N15 — §4.1 Masked wind preparation, ImputeFormer, Eq (19)**
Context: "The observed vectors and their masks are supplied to a fixed-node masked ImputeFormer model … the completed station vectors are averaged at each hour."
Annotation: "Understand whether the wind-imputation model is run for **all** grid cells in the city or only for station grid cells. If it's only for station grid cells, update to the correct version. → Also required for the (paper-facing) results. (Needs to be clarified.) Also, if a city-scale imputer is used, explain it better here in the paper." Add: "add data".
Status: addressed.

**N16 — Covariance model, Eq (25)**
Context: "Σ_i(t,τ) = R_i(t,τ) diag(σ_∥², σ_⊥²) R_i(t,τ)^T … The age in this covariance is lower-bounded by a positive minimum dispersion time, so same-time releases remain finite."
Annotation: "What does the 'same-time' covariance mean? What is 'age' in this age factor?" Also: "(might not be needed based on the data)".
Status: addressed.

**N17 — Response-matrix entries, Eq (26)**
Context: "[O G^∂_{t,τ}(ŵ_{τ:t}; ψ) e_i]_s = χ_i(t,τ) K_ψ(x_s; z_i(t), Σ_i(t,τ))."
Annotation: "What is χ_i(t,τ)? Explain after the equation."
Status: addressed.

**N18 — Kernel boundary / truncation loss**
Context: "Kernel boundary or truncation loss and whole-release exit loss are recorded separately and are never renormalized. Summing repeated sensor observations is not a mass estimate."
Annotation: "What is kernel boundary / truncation loss, and how are we calculating it? What does the last line mean?"
Status: addressed.

**N19 — §4.2 Trajectory discretization, Eq (23)**
Context: "z^{r+1}_i = z^r_i + δa ŵ_{a_r}(z^r_i)."
Annotation: (marked) IMPORTANT — verify.
Status: addressed.

**N20 — Paper-facing response matrices paragraph**
Context: "Paper-facing response matrices use a zero-source, zero-initial-state baseline. Kriged initial conditions belong only to auxiliary PDE mismatch experiments…"
Annotation: "A bit weird to phrase." / "Has to be written better." / **IMPORTANT: this replacement paragraph has to be moved to Section 7.**
Status: addressed.

**N21 — Computational roadmap sentence**
Context: "The computational roadmap implements this response in PyTorch. Tensor operations retained in the graph can be differentiated, but discrete exit events and reporting logic are not assigned gradients."
Annotation: "A bit weird to phrase."
Status: addressed.

**N22 — §4.3 Background construction, effective rank**
Context: "…Its paper-facing effective rank is capped at eight. Source-derived columns are allowed only in labeled stress tests."
Annotation: "What does this sentence mean?" / "for paper-facing".
Status: addressed.

**N23 — §4.3/4.4 Background declared before fitting**
Context: "The primary background specification is declared before source fitting and is not selected by inspecting Y or by optimizing source recovery."
Annotation: "Is the background not fitted using the data at all? If so, how do we decide? If not, why are we saying we're not inspecting Y?"
Status: addressed.

**N24 — Alternative source-independent bases**
Context: "Alternative source-independent bases are predeclared sensitivity analyses; a source-like basis is only a labeled stress test. None is data-selected as a replacement authoritative model."
Annotation: "(repetitive) — what does that mean?"
Status: addressed.

**N25 — Projection, Eq (27)**
Context: "P⊥_Q X = X − U_r(U_r^T X)."
Annotation: (marked for clarification)
Status: addressed.

**N26 — Response matrix and source-activity fit, Eq (28)**
Context: "h^lag_{t,kb} = Σ_{ℓ∈L_t} φ_b(t−ℓ) O G^∂_{t,t−ℓ}(ŵ_{t−ℓ:t}; ψ) s_k."
Annotation: "Same problem here — where is c_{kb}? Saying φ, but where does it begin and end?"
Status: addressed.

**N27 — Coefficient fit, Eq (30)**
Context: "ĉ = arg min_{c≥0} ‖Ỹ − H̃_Φ c‖²₂ + λR(c)."
Annotation: "Okay, so c is treated as a parameter to fit."
Status: addressed.

**N28 — §4.5 Constrained end-to-end refinement, Eqs (34)–(35)**
Context: "A future PyTorch implementation may refine wind and dispersion under explicit constraints … L_refine = …"
Annotation: "We are doing it now, not in the future." / "(explain the terms)". Also (Eq 34): "(see diagram/derivation)".
Status: addressed.

**N29 — Regularizer, Eqs (32)–(33)**
Context: "R(c) = ‖c‖²₂" and "R(c) = ‖c − c₀‖²₂."
Annotation: (checked) verify.
Status: addressed.

---

## Section 5 (Identifiability Theory)

**N30 — Prop 5.2, nonnegative orthant**
Context: "…c is identifiable uniformly for all c ∈ ℝ^J_+ if and only if rank(H̃_Φ) = J."
Annotation: "What is the non-negative orthant?"
Planned fix: Define \(\mathbb R_+^J\) in the theorem prose or immediately before it as the feasible set of vectors with all entries nonnegative; use "nonnegative coefficient vectors" in prose to avoid relying on the jargon.

**N31 — Boundary-cone paragraph after Prop 5.2**
Context: "Nonnegativity can remove some null directions at the boundary of the feasible cone, so a particular boundary point may be locally more identifiable than this uniform condition indicates…"
Annotation: "How can non-negativity remove some null directions at the boundary of the feasible cone? → Explain the paragraph."
Status: addressed by removing the boundary-cone discussion and replacing it with the full-column-rank/predeclared-mask statement.

**N32 — Prop 5.2 / 5.3 proofs**
Context: proof of the identifiability propositions.
Annotation: "Include a detailed proof in the appendix."
Status: addressed by adding detailed appendix proofs for Propositions~\ref{prop:rank_identifiability} and~\ref{prop:noise_robust}.

**N33 — Rank/kernel argument in proof**
Context: "If rank(H̃_Φ) = J, then ker(H̃_Φ) = {0}…"
Annotation: "The rank & kernel thing is fairly simple, but it would be great to add a citation."
Planned fix: Use explicit "rank-nullity theorem" wording in the main proof and appendix. Question: do you want a formal linear-algebra citation in the bibliography, or is naming rank-nullity enough?
No formal citation required.

**N34 — Numerical rank tolerance, Eq (42)**
Context: "τ_num = max(N, J) ε_mach σ_1, r_num = #{i : σ_i > τ_num}."
Annotation: "What is ε_mach? The τ_num definition is not clear."
Planned fix: Define \(\epsilon_{\mathrm{mach}}\) as the machine precision of the numerical dtype used for the SVD, and explain that \(\tau_{\mathrm{num}}\) is a floating-point numerical-rank tolerance, distinct from the scientific noise threshold \(\tau_\sigma\).

**N35 — Primary identifiability score, Eq (43)**
Context: "I = σ_J(H̃_Φ)."
Annotation: "Should say the primary identifiability score is σ̃_J, including zero values if the matrix is rank-deficient." Also: "Call this something else — give it a name."
Planned fix: Rename \(\mathcal I\) to a descriptive metric name, e.g. "padded smallest singular value" or "minimum coefficient singular value", and state explicitly that rank-deficient cases have score zero because \(\sigma_J=0\).

**N36 — Smallest positive singular value σ_min,+ / effective rank, Eq (45)**
Context: "r_eff(τ_σ) = #{i : σ_i(H̃_Φ) > τ_σ}."
Annotation: "Give a strategy to obtain τ_σ from the noise (even as an example)."
Planned fix: Add a noise-calibration strategy: choose \(\tau_\sigma\) from a predeclared observation-noise scale or bootstrap residual scale, and record it separately from \(\tau_{\mathrm{num}}\). Include an example rule rather than claiming a single universal threshold.

**N37 — Coefficient visibility / weak set, Eq (46)**
Context: "v_j = ‖h̃_j‖₂, W = {j : v_j ≤ τ_v}."
Annotation: "Explain τ_v and what the weak set signifies. Give a strategy to determine a suitable τ_v. [Should be interpretable.]"
Planned fix: Explain that \(v_j\) is the projected sensor-time magnitude of a unit coefficient fingerprint, and \(\tau_v\) marks fingerprints too small to detect reliably. Give a threshold strategy based on minimum detectable signal or signal-to-noise ratio.

**N38 — Ambiguous-pair set, Eq (48)**
Context: "A = {(i,j) : i,j ∉ W, ρ_ij > τ_ρ}, τ_ρ ∈ [0,1]."
Annotation: "How to fit τ_ρ?"
Planned fix: Rephrase "fit" as "predeclare/calibrate." Give a strategy such as choosing \(\tau_\rho\) from controlled calibration experiments or using a conservative default like high coherence near one. State it is not optimized after seeing source-recovery results.

**N39 — Coherence 'N/A' convention**
Context: "Coherence is undefined when either fingerprint is weak. Such entries are reported as 'N/A'… and never allowed to hide the corresponding weak-visibility flags."
Annotation: "Unnecessary to add this part."
Planned fix: Shorten the weak-pair convention. Keep only that coherence/ray distance are reported for nonweak pairs; remove the extra defensive prose if it interrupts the flow.

**N40 — §5.4 Ray distance, Eq (49)**
Context: "…we reactivate the ray distance for eligible, generally signed projected fingerprints: d^ray_ij = min_α ‖h̃_i − α h̃_j‖₂ / ‖h̃_i‖₂."
Annotation: "Don't use 'reactivate'."
Planned fix: Replace "reactivate" with neutral wording such as "We also report ray distance..." or "As a distance counterpart to coherence..."

**N41 — Ray distance simplification, (49) → (50)**
Context: "For nonweak fingerprints, d^ray_ij = √(1 − ρ²_ij)."
Annotation: "How does (49) become (50)? What value to put for α?" / "Requires a short proof."
Planned fix: Add a short appendix proof. Show the minimizer \(\alpha^\star=(\widetilde{\mathbf h}_i^\top\widetilde{\mathbf h}_j)/\|\widetilde{\mathbf h}_j\|_2^2\) and substitute it to obtain \(\sqrt{1-\rho_{ij}^2}\) for nonweak fingerprints.

**N42 — Ray distance as complementary report**
Context: "…ray distance is a complementary report rather than an independent identifiability theorem."
Annotation: (label) "diagnostic".
Planned fix: Rephrase the sentence to call ray distance a diagnostic, not a "complementary report"; make clear it is geometrically equivalent to coherence under the stated normalization.

**N43 — §5.4 Background absorption fraction, Eq (51)**
Context: "Machine-readable reports ... after projection."
Annotation: "Not required."
Planned fix: Compress or remove the machine-readable-report details. Keep the background absorption definition only if it remains in the diagnostic table/reporting protocol.

**N44 — §5.5 Response-matrix perturbations, Eq (52)**
Context: "Write the projected response discrepancy as Δ_H = Δ_tr + Δ_inv + Δ_int."
Annotation: (label) "use".
Planned fix: State explicitly how the decomposition is used: \(\Delta_{\mathrm{tr}}\) motivates transport ensembles/uncertainty, \(\Delta_{\mathrm{inv}}\) motivates inventory robustness scenarios, and \(\Delta_{\mathrm{int}}\) records interaction rather than a separate pooled uncertainty.

**N45 — Prop 5.4 perturbation bound, Eq (55)**
Context: "‖ĉ − c‖₂ ≤ (‖Ẽ‖₂ + δ_H‖c‖₂) / σ_J(H̃_Φ)."
Annotation: (checked) verify.
Planned fix: Verify the perturbation-bound algebra and either add an appendix proof or cite the same least-squares singular-value argument used in the noise-robustness proof.

---

## Section 6 (Identifiability-Aware Apportionment Method) & Table 1

**N46 — Table 1 (diagnostics reported)**
Context: "Table 1. Identifiability diagnostics reported with source activity estimates."
Annotation: "Should include further things like ambiguity graphs, merge recommendations, and so on. Also include the **name** of each metric alongside its interpretation."
Planned fix: Expand Table 1 to include metric names, symbols, interpretation, and reporting action. Add graph/reporting outputs such as weak set, ambiguous-pair set, source ambiguity graph, connected report groups, and global unresolved warning.

**N47 — Table 1 / thresholds**
Context: the diagnostic/action columns of Table 1.
Annotation: "Should add a table of thresholds and determination strategies."
Planned fix: Add a separate threshold table for \(\tau_{\mathrm{num}}\), \(\tau_\sigma\), \(\tau_v\), \(\tau_\rho\), \(\tau_\rho^{\mathrm{ref}}\), and \(\epsilon_w\), with meaning, default/source, and calibration strategy.

**N48 — §6.2 Procedure (10 steps)**
Context: "The procedure is: 1. Predeclare the model… 10. Report."
Annotation: "Give an explanation of the procedure. Also add a justification of why this particular procedure has been selected."
Planned fix: Add a short paragraph before the enumerated algorithm explaining the order: predeclare model choices, build response, project background, fit coefficients, diagnose identifiability, then report conservative groups. Emphasize that this ordering prevents post-fit diagnostic tuning.

**N49 — Projected FISTA solver note**
Context: "The nonnegative quadratic problem is solved with projected FISTA in PyTorch…"
Annotation: Get an explanation of the procedure and justify why this procedure.
Planned fix: Explain projected FISTA as a solver for the nonnegative quadratic objective: gradient step for the smooth least-squares/ridge loss, projection by clamping to \(\mathbf c\ge0\), and convergence by KKT/projected-gradient residual. Add a citation or move solver details to appendix if the main text gets too crowded.

**N50 — §6.3 Uncertainty reporting, Eqs (59)–(60)**
Context: "σ̂² = ‖r̃‖²₂ / max(N − r_eff(τ_σ), 1)" and "Cov(ĉ_{A_c}) = σ̂²(H̃^T_{Φ,A_c} H̃_{Φ,A_c} + λI)^†."
Annotation: "Understand why we calculate the uncertainty this way. Add citations if these are standard results. What does the covariance tell us about uncertainty?"
Planned fix: Explain \(\widehat\sigma^2\) as a residual variance estimate after effective degrees of freedom, and the covariance formula as the active-set/ridge least-squares covariance approximation. State that diagonal terms give coefficient uncertainty and off-diagonal terms show coupled estimates; add standard regression/least-squares citations if desired.

**N51 — §6.3 Wind ensembles paragraph**
Context: "When wind ensembles are available, IASA also constructs response matrices H̃^(b)_Φ, refits ĉ^(b), and reports empirical quantiles across ensemble members…"
Annotation: "This is a bit weirdly written. Paraphrase." Also: "short?"
Planned fix: Shorten to: for each wind ensemble member, rebuild the response, refit coefficients, and summarize empirical quantiles as transport uncertainty. Keep this separate from inventory scenarios.

**N52 — §6.4 Residual model-adequacy check, Eqs (61)–(62)**
Context: "r̃ = Z^T(Y − H^lag_Φ ĉ), Σ̄_e = Z^T Σ_e Z … T_res = r̃^T Σ̄_e^{−1} r̃."
Annotation: "What exactly is Z? Why are we doing this exactly? What does σ̄ mean? What does Σ_e mean? In our model, do we even have a noise covariance? Isn't the noise independent?"
Planned fix: Define \(Z\) as an orthonormal basis for the background-orthogonal observed-row space, \(\Sigma_e\) as an externally declared observation-noise covariance, and \(\bar\Sigma_e=Z^\top\Sigma_e Z\) as its projected covariance. Add the independent-noise special case \(\Sigma_e=\sigma^2 I\). State that without a calibrated \(\Sigma_e\), only uncalibrated residual summaries are reported.

**N53 — §6.4 Parametric bootstrap**
Context: "alpha=0.05, it flags model ... add-one correction."
Annotation: "What is happening here? Understand + justification."
Planned fix: Explain the parametric bootstrap in plain language: simulate observations from the fitted model plus declared noise, refit each synthetic dataset, recompute the residual statistic, and compare the observed statistic to this null distribution. Justification: it accounts for the fitting step.

**N54 — §6.6 Acceptance criteria for refinement, Eq (64)**
Context: "max_{i≠j, i,j∉W} ρ^ref_ij ≤ τ^ref_ρ."
Annotation: "How to set [τ^ref_ρ]?"
Planned fix: Define \(\tau_\rho^{\mathrm{ref}}\) in the threshold table. Question: should the default be \(\tau_\rho^{\mathrm{ref}}=\tau_\rho\), or should refinement use a stricter threshold to prevent separability degradation?
Keeping it same is ok.

**N55 — §6.6 Padded-spectrum convention**
Context: "If either response is numerically rank-deficient, its σ_J is zero under the padded-spectrum convention of Section 5.3."
Annotation: "What?"
Planned fix: Replace "padded-spectrum convention" with explicit wording: if \(\widetilde H_\Phi\) has fewer than \(J\) numerically nonzero singular values, define \(\sigma_J=0\) for identifiability scoring.

**N56 — §6.7 Reporting protocol**
Context: "For sources k ≠ k′, an edge is added when at least one eligible coefficient pair … belongs to A."
Annotation: "Repeating in the last section."
Planned fix: De-duplicate Section 6.7. Keep the mathematical source-graph definition in Section 5 and make Section 6.7 a concise operational reporting summary, or vice versa.

---

## Section 7 (New Delhi Case Study and Experimental Platform)

**N57 — §7 intro**
Context: "The empirical study is anchored in New Delhi rather than in an abstract city. Its regulatory sensor geometry, pollution observations…"
Annotation: "Paraphrase / rewrite."
Planned fix: Rewrite the Section 7 intro to directly state why New Delhi is the shared empirical platform: same regulatory sensors, PM\(_{2.5}\), wind observations, and proxy inventories support both controlled and observed-data studies.

**N58 — §7.1 Government observations table**
Context: "The authoritative local table contains hourly government observations with columns monitor_id, timestamp_round, AT, RH, WD, WS, pm10, and pm25…"
Annotation: "Rebase / rephrase."
Planned fix: Rephrase the government-observation paragraph as a data description, keeping columns, date range, station count, and missingness, but avoiding overly legalistic "authoritative local table" phrasing.

**N59 — §7.1 Final 32-sensor layout**
Context: "…This operation, rather than grid-cell deduplication, produces the final 32-sensor layout."
Annotation: "(remove)".
Planned fix: Remove the "rather than grid-cell deduplication" clause. Keep only that the two Pusa monitors are averaged into one output sensor, producing the 32-sensor layout.

**N60 — §7.2 Imputed product paragraph**
Context: "The imputed product is required for paper-facing New Delhi runs. An explicitly labeled observed-only station-mean fallback exists solely for smoke tests…"
Annotation: "(remove)".
Planned fix: Remove any smoke-test/fallback wording from the manuscript. Keep fallback details in code/README only.

**N61 — §7.3 Crop rows/columns**
Context: "Every 80×80 map is cropped by rows 21:61 and columns 16:56, giving a 40×40 study map, then divided by its own cropped 99th percentile."
Annotation: "This is the crop corresponding to the 32 stations (but this can be changed in the code)."
Planned fix: Rephrase the crop as the chosen New Delhi study window covering the regulatory network, not as an immutable model assumption. Mention that the crop is recorded as experiment metadata.

**N62 — §7.3 Named proxy inventories, Table 2 (traffic 00/06/12/18)**
Context: "Table 2. Named New Delhi proxy inventories… Traffic 00 / 06 / 12 / 18 → nearest traffic time slot."
Annotation: "The traffic_00 to traffic_18 inventories should be used to put the time-varying diurnal pattern in order (→ other patterns; the roads should be visible). **Do not use them as individual inventories.**"
Planned fix: Treat traffic as one source category with time-varying/diurnal structure derived from the 00/06/12/18 traffic maps, rather than four independent source inventories. This requires updating the inventory table, source count \(K\), and any claims about "seven" named inventories. Question: should the source list become brick kilns, industries, population density, and traffic (four groups), or should traffic have multiple temporal basis components but one source group?
All source groups have multiple temporal basis components. Create an additional temporal basis component for traffic, which can be set to 0 scientifically for the other source groups.

**N63 — §7.3 Brick-kiln deterministic seed**
Context: "Brick-kiln blocks use a recorded deterministic seed."
Annotation: "(Justification) What seed? Understand the explanation. (Also the alignment part.) Actually, should this be changed?"
Planned fix: Either remove seed-level detail from the main paper or justify it as a deterministic controlled-experiment proxy for intermittent brick-kiln activity. Question: do we still want the brick-kiln 12-hour block model in the paper-facing New Delhi setup? Yes.

**N64 — §7.4 Forward models baseline**
Context: "…the New Delhi-derived 40×40 domain and regulatory sensor response matrix uses the open-boundary Gaussian puff operator from Section 4.2, evaluates kernels directly at sensors, and starts from a zero-source, zero-initial-state baseline."
Annotation: (marked) verify.
Planned fix: Reconcile the baseline text with Section 4: controlled operator-matched runs may use zero-initial-state responses, while observed New Delhi runs subtract a declared zero-source transported initial-condition baseline.

**N65 — §7.4 Kriged / initial conditions**
Context: "…kriged government initial conditions are confined to auxiliary PDE trials; their matched zero-source trajectory is subtracted before source fitting."
Annotation: "Maybe kriged or spatially imputed initial conditions is the right approach? → or how are we going to fit it? Can we do some kind of warm-up so that it fits with the observed data on the first day?" Also: "why?"
Planned fix: Clarify the observed-run initialization policy and justify it. Question: should the paper use kriging from first observations, a learned/spatial imputation, or a warm-up/spin-up period before fitting source activity? It should use kriging from first observations for real-data results.

**N66 — §7.4 Auxiliary advection-diffusion simulator (H5)**
Context: "An auxiliary advection-diffusion simulator uses first-order upwind advection, a five-point Laplacian, a two-stage Heun update…"
Annotation: "Why are we using the old simulator for H5? Can we not use the puff transport simulator?"
Planned fix: Decide whether H5 is an operator-mismatch test using the auxiliary PDE simulator, or a perturbation test within the puff family. Question: should the old advection-diffusion simulator remain as the mismatch generator, or should H5 use only puff-response perturbations? Use only puff-response perturbations.

**N67 — §7.4 Normal background effective rank**
Context: "Normal backgrounds are built from the source-independent components in Section 4.3 with effective rank at most eight … uses a redundant-column version and a labeled source-like stress basis."
Annotation: "Not useful — [is this] pairwise required? What [is this]?"
Planned fix: Shorten the Section 7 background paragraph and leave detailed background-rank/stress-basis explanation in Sections 4/8. Question: is the concern that the source-like stress background is unnecessary, or only that it is overexplained here? This has been explained before, so shorten it.

**N68 — §7.4 traffic_06 zero response column**
Context: "The traffic_06 inventory is identically zero in the selected crop and therefore has a zero response column for every basis component…"
Annotation: (marked) verify handling of the all-zero inventory.
Planned fix: Revisit after the traffic-source decision in N62. If traffic is one source, remove the "traffic_06 source" unsupported-source language and instead mention that one traffic time slice has no support in the selected crop if still relevant.

**N69 — Overall: spatial structure of apportionment**
Context: §7.3 / whole-field apportionment framing.
Annotation: "What is really going to happen if we have two source classes that are far apart? → Barycenter? Is this framework applicable to each sensor location separately, or is it going to be one apportionment output for the whole field? Also, where are we even talking about the **spatial structure** of the apportionment?  I want to extend to this angle."
Planned fix: Add a spatial-structure paragraph: IASA reports source-group activities over the declared domain, not independent per-sensor apportionments; spatial structure enters through inventory maps, wind-conditioned response fingerprints, and sensor geometry. Question: do you want to add a new spatial diagnostic such as source centroid/barycenter, or only clarify the current whole-domain interpretation? I want to extend in the direction of tracing the puffs back at each sensor level to explain the spatial distribution of sources that are responsible for polution for each sensor location.

---

## Section 8 (Evaluation)

**N70 — §8.1 H1 (conditioning predicts recovery)**
Context: "H1: conditioning predicts recovery. We vary source geometry and Gaussian observation noise at 0%, 1%, 5%, 10%, and 20%…"
Annotation: "(only H?) Use Exp 1?"
Planned fix: Rename hypothesis labels to experiment labels, e.g. "Experiment 1: Conditioning and Recovery", unless the final paper should keep the H1--H5 hypothesis style.

**N71 — §8.1 H2 (coherent sources require grouped reporting)**
Context: "H2: coherent sources require grouped reporting. We form increasingly coherent source pairs through nearby spatial splits and matched transport paths…"
Annotation: "How are we going to do nearby spatial splits?"
Planned fix: Define the split construction explicitly: partition one inventory into nearby spatial components, shifted copies, or upstream/downstream regions, then vary separation to control coherence. Question: which split construction should be used in the experiments? Shifted copies should make the most sense here.

**N72 — §8.1 H3**
Context: Full paragraph
Annotation: "Why? Requires a better explanation."
Planned fix: Add rationale for H3: background correction is tested because too little background leaves broad confounding in residuals, while too flexible/source-like backgrounds can absorb source signal and reduce identifiability.

**N73 — §8.1 H5a / H5b / H6–H9 labels**
Context: "H5a: transport error is amplified by ill-conditioning… H5b: inventory error changes the attribution target… Lag-window sensitivity… Missing-source model adequacy… Temporal-basis recovery."
Annotation: (margin labels grouping the hypotheses) — "H5, H6, H7 (can be done if H8 trip), H9".
Planned fix: Reorganize the experiment matrix into sequential experiments rather than H5a/H5b plus unlabeled studies: Exp 5 transport error, Exp 6 inventory robustness, Exp 7 lag-window sensitivity, Exp 8 missing-source adequacy, Exp 9 temporal-basis recovery.

**N74 — §8.8–8.15 Results-pending sections**
Context: "Results pending. The table will identify the primary and alternative inventory versions…" (§8.14 New Delhi Proxy Apportionment).
Annotation: "Real-data results." (mark the sections tied to the real New Delhi run)
Planned fix: Separate controlled-result placeholders from observed New Delhi result placeholders. Group wind-imputation validation, New Delhi identifiability/report groups, proxy apportionment, and residual diagnostics as real-data results.

---

## Section 9 (Discussion and Conclusion)

**N75 — Limitations paragraph**
Context: "The framework has several limitations. The response operator is an approximation to atmospheric transport and can be extended… The theory is linear in source–basis coefficients conditional on fixed wind… The selected temporal basis also limits the activity patterns that can be recovered…"
Annotation: "→ 'Can be extended' but this is not a limitation. → This is not a limitation. → Again, not a limitation. Should point to a future-work paragraph."
Planned fix: Split true limitations from future work. Keep limitations that affect current claims (declared inventories, sparse wind validation, temporal-basis restriction, missing-source risk) and move richer chemistry/deposition/vertical mixing/boundary inflow to a future-work paragraph.

**N76 — Spatial wind ignorance**
Context: "…current city-level wind sequence ... meteorological variation ..."
Annotation: "Should not be the case."
Planned fix: Current context is unclear after the note update. Question: does this mean the paper should not use a city-level wind sequence and must claim/implement spatially varying gridded wind, or does it mean the limitation should be removed because spatial wind is already handled? It should be removed because it is already handled.

---

### Summary of cross-cutting requests

- **Restructure / de-duplicate prose** in Related Work and the DA/PINN/spatio-temporal paragraphs; several read as repetitive and their relevance to the paper must be justified (N5–N8, N56).
- **Move the paper-facing "replacement" paragraph to Section 7** and rewrite the low-level implementation directives out of the theory sections (N20, N29-adjacent, N60–N61).
- **Explain and give selection strategies for every threshold**: τ_num, τ_σ, τ_v, τ_ρ, τ^ref_ρ, and ε_mach (N34, N36–N38, N54); add a dedicated thresholds table (N47).
- **Add proofs / citations**: appendix proof for the identifiability propositions, the (49)→(50) ray-distance step, and the rank/kernel argument (N32–N33, N41).
- **Clarify the wind-imputation scope** (all grid cells vs. station cells) since paper-facing results depend on it (N15).
- **Address the spatial structure of the apportionment** — whole-field vs. per-sensor output, and behavior for widely separated source classes (N69).
- **Reframe "limitations" as future work** where they are design choices rather than true limitations (N75).
