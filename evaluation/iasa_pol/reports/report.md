# IASA Evaluation Report

## Experiment 1: Conditioning and Recovery

| noise_frac | geometry | sigma_J | numerical_rank | effective_rank | condition_number | coefficient_relative_error | residual_norm | min_visibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | separated | 3.273974 | 2 | 2 | 8.894849 | 0.0 | 0.0 | 3.274887 |
| 0.01 | separated | 3.273974 | 2 | 2 | 8.894849 | 0.013187 | 1.27302 | 3.274887 |
| 0.05 | separated | 3.273974 | 2 | 2 | 8.894849 | 0.065937 | 6.365102 | 3.274887 |
| 0.1 | separated | 3.273974 | 2 | 2 | 8.894849 | 0.131873 | 12.730204 | 3.274887 |
| 0.2 | separated | 3.273974 | 2 | 2 | 8.894849 | 0.263747 | 25.460408 | 3.274887 |
| 0.0 | close | 9.598349 | 2 | 2 | 2.263032 | 0.0 | 0.0 | 12.097923 |
| 0.01 | close | 9.598349 | 2 | 2 | 2.263032 | 0.005665 | 1.684167 | 12.097923 |
| 0.05 | close | 9.598349 | 2 | 2 | 2.263032 | 0.028325 | 8.420834 | 12.097923 |
| 0.1 | close | 9.598349 | 2 | 2 | 2.263032 | 0.056651 | 16.841667 | 12.097923 |
| 0.2 | close | 9.598349 | 2 | 2 | 2.263032 | 0.113302 | 33.683334 | 12.097923 |

- _Recovery error vs sigma_J and condition number._

## Experiment 2: Coherence and Grouped Reporting

| offset | triggering_pair | max_eligible_coherence | ray_distance | individual_relative_error | grouped_relative_error | merged | numerical_rank | sigma_J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.0 |  | 0.047697 |  | 0.0 | 0.0 | False | 2 | 21.767119 |
| 6.0 |  | 0.861804 |  | 0.0 | 0.0 | False | 2 | 10.476535 |
| 4.0 |  | 0.905274 |  | 0.0 | 0.0 | False | 2 | 8.396899 |
| 2.0 |  | 0.945356 |  | 0.0 | 0.0 | False | 2 | 5.425126 |
| 1.0 |  | 0.958354 |  | 0.0 | 0.0 | False | 2 | 4.6416 |

- _Undefined coherence/ray-distance for weak or ineligible pairs are null._
- _Grouped error reported wherever a non-singleton component merges sources._

## Experiment 3: Background Stress

| background_mode | declared_before_fit | min_visibility | max_absorption | sigma_J | coefficient_relative_error | residual_norm |
| --- | --- | --- | --- | --- | --- | --- |
| none | True | 1.0 | 0.0 | 0.08039 | 2.362513 | 6.859066 |
| primary | True | 0.972067 | 0.234704 | 0.079969 | 2.606339 | 6.851815 |
| redundant | True | 0.972067 | 0.234704 | 0.079969 | 2.606339 | 6.851815 |
| stress | True | 0.0 | 1.0 | 0.0 | 0.775381 | 6.851815 |

- _Predeclared background basis; declared_before_fit records independence from Y and recovery results._

## Experiment 4: Wind Diversity and Sensor Geometry

| wind_provider | wind | layout | sigma_J | numerical_rank | max_eligible_coherence | coefficient_relative_error |
| --- | --- | --- | --- | --- | --- | --- |
| constant_direction | constant | regulatory | 1.526926 | 2 | 0.975115 | 0.0 |
| constant_direction | constant | random | 0.0 | 2 | 0.028406 | 4e-06 |
| constant_direction | constant | downwind | 0.125677 | 2 | 0.989677 | 0.0 |
| single_direction_synthetic | single | regulatory | 6.44243 | 2 | 0.741096 | 0.0 |
| single_direction_synthetic | single | random | 1e-06 | 2 | 0.999983 | 0.573115 |
| single_direction_synthetic | single | downwind | 1.362939 | 2 | 0.99011 | 0.0 |
| diurnal_synthetic | diurnal | regulatory | 6.111524 | 2 | 0.721872 | 0.0 |
| diurnal_synthetic | diurnal | random | 2.4e-05 | 2 | 0.880117 | 0.000896 |
| diurnal_synthetic | diurnal | downwind | 2.604398 | 2 | 0.897464 | 0.0 |
| ar1_synthetic | ar1 | regulatory | 7.35123 | 2 | 0.640798 | 0.0 |
| ar1_synthetic | ar1 | random | 6.6e-05 | 2 | 0.773931 | 0.115098 |
| ar1_synthetic | ar1 | downwind | 2.14415 | 2 | 0.915978 | 0.0 |
| multi_direction_synthetic | multi | regulatory | 5.698914 | 2 | 0.1288 | 0.0 |
| multi_direction_synthetic | multi | random | 2.794802 | 2 | 0.321684 | 0.0 |
| multi_direction_synthetic | multi | downwind | 0.410317 | 2 | 0.109755 | 0.0 |
| gridded_kernel_new_delhi | real | regulatory | 2.845609 | 2 | 0.155812 | 0.0 |
| gridded_kernel_new_delhi | real | random | 2.102207 | 2 | 0.287236 | 0.0 |
| gridded_kernel_new_delhi | real | downwind | 0.379748 | 2 | 0.220531 | 0.0 |

- _Wind comparisons use identical source-basis columns._

## Experiment 4: Wind-Window Ensemble Distribution

| ensemble | provider | n_members | n_windows | sigma_J_quantiles | prob_full_numerical_rank | prob_full_effective_rank | coefficient_weak_probabilities | source_pair_ambiguity_probabilities | report_component_frequencies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical | historical_real_new_delhi_window | 20 | 20 | {"q05":1.449809,"q50":4.510983,"q95":9.959122} | 1.0 | 1.0 | {"0":0.0,"1":0.0} | {} | {"src_a":1.0,"src_b":1.0} |
| simulated | ar1_synthetic | 20 | 20 | {"q05":3.499547,"q50":5.576352,"q95":6.801457} | 1.0 | 1.0 | {"0":0.0,"1":0.0} | {} | {"src_a":1.0,"src_b":1.0} |

- _Historical windows are contiguous real-record slices; simulated windows are AR(1). Quantiles are [0.05, 0.5, 0.95]._

## Experiment 5: Transport Error (Parametric)

| perturbation_kind | perturbation_value | operator_error_norm | sigma_J | singular_values | coefficient_relative_error | residual_norm |
| --- | --- | --- | --- | --- | --- | --- |
| wind_direction_deg | 0.0 | 0.0 | 0.865305 | [21.138254,0.865305] | 0.0 | 0.0 |
| wind_direction_deg | 5.0 | 0.28799 | 3.369168 | [21.569815,3.369168] | 0.622829 | 3.627949 |
| wind_direction_deg | 10.0 | 0.59723 | 7.068079 | [21.414989,7.068079] | 0.79715 | 6.951087 |
| wind_direction_deg | 20.0 | 0.86758 | 7.97813 | [15.362633,7.97813] | 0.861271 | 10.750982 |
| wind_speed_factor | 1.0 | 0.0 | 0.865305 | [21.138254,0.865305] | 0.0 | 0.0 |
| wind_speed_factor | 1.25 | 0.288187 | 2.088865 | [19.601894,2.088865] | 0.450884 | 1.782354 |
| wind_speed_factor | 1.5 | 0.52893 | 4.566836 | [19.235006,4.566836] | 0.273777 | 1.55143 |
| dispersion_factor | 1.0 | 0.0 | 0.865305 | [21.138254,0.865305] | 0.0 | 0.0 |
| dispersion_factor | 1.5 | 0.189049 | 1.417145 | [17.677014,1.417145] | 0.496889 | 1.258896 |
| dispersion_factor | 2.0 | 0.351806 | 1.759434 | [14.722803,1.759434] | 0.89331 | 2.516423 |

- _Transport ensemble kind: transport._

## Experiment 5: Structural (Edge-Hold PDE) Mismatch

| generator | operator_mismatch_norm | coefficient_relative_error_mean | adequacy_rejection_rate | n_trials |
| --- | --- | --- | --- | --- |
| edge_hold_pde | 0.444317 | 0.861678 | 1.0 | 10 |

- _Structural generator differs in shape from the puff family; a global scale cannot hide the mismatch._

## Experiment 6: Inventory Robustness

| scenario | sigma_J | coefficient_relative_error | c_hat |
| --- | --- | --- | --- |
| baseline | 21.563225 | 0.0 | [1.0,0.6] |
| location_shift | 28.846704 | 0.0 | [1.0,0.6] |
| spatial_scale | 47.001535 | 0.0 | [1.0,0.6] |
| alt_map_version | 46.007366 | 0.0 | [1.0,0.6] |
| category_swap | 21.563225 | 0.0 | [1.0,0.6] |

- _Rows are robustness scenarios, NOT confidence-interval draws._
- _Transport/inventory pooling rejected: True._

## Experiment 7: Lag-Window Selection

| lag_window_steps | sigma_J | numerical_rank | condition_number | n_report_components | coefficient_relative_error |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.078112 | 2 | 569.16498 | 2 | 0.0 |
| 6 | 0.70979 | 2 | 63.788337 | 2 | 0.0 |
| 8 | 2.45787 | 2 | 18.605219 | 2 | 0.0 |
| 10 | 4.572274 | 2 | 10.101366 | 2 | 0.0 |
| 12 | 6.15736 | 2 | 7.539568 | 2 | 0.0 |
| 16 | 9.074497 | 2 | 5.140908 | 2 | 0.0 |

- _Selected lag: 16 (tau_L=0.001, criterion=smallest_L_with_relative_frobenius_delta_le_tau_L)._
- _Row count fixed across lag: True; coefficients used for selection: False._

## Experiment 8: Missing-Source Adequacy

| case | rejection_rate |
| --- | --- |
| null | 0.05 |
| residual_visible | 1.0 |
| aligned (in-span) | 0.05 |

- _alpha=0.05, n_replicates=1000, n_trials=100, omission_amplitude=1.2._
- _Omitted-source out-of-span fraction=0.913622, background rank=4 (non-empty Q on the platform)._
- _Fitted-design spectrum: sigma_J=45.845516, numerical_rank=3, singular_values=[120.555606, 69.205218, 45.845516]._
- _Rejection diagnoses model inadequacy without identifying its cause; non-rejection cannot certify inventory completeness._

## Experiment 9: Temporal-Basis Recovery

| noise_frac | coefficient_relative_error | activity_relative_error | sigma_J |
| --- | --- | --- | --- |
| 0.0 | 0.0 | 0.0 | 0.141245 |
| 0.02 | 0.131714 | 0.075004 | 0.141245 |
| 0.05 | 0.329296 | 0.187515 | 0.141245 |
| 0.1 | 0.658624 | 0.37504 | 0.141245 |
| 0.2 | 0.900481 | 0.582125 | 0.141245 |

- _Basis names: ['diurnal', 'block', 'day_night']._

## Experiment 10: Per-Sensor Footprints

| footprint_localization_error_cells | footprint_mass_fraction_within_radius | localization_radius_cells | n_active_cells | contribution_sum_error | footprints_nonnegative | coefficient_relative_error |
| --- | --- | --- | --- | --- | --- | --- |
| 0.742408 | 1.0 | 4.0 | 50 | 0.0 | True | 0.787509 |

- _Footprint localization error vs known source origins; contributions sum to the fitted sensor signal (contribution_sum_error ~ 0)._
- _Non-singleton report component present; footprints reported per group._

## New Delhi Identifiability and Report Groups (weeks 1-4)

| week | window_start | window_end | sigma_1 | sigma_J | numerical_rank | effective_rank | condition_status | max_eligible_coherence | visibility | weak_set | ambiguous_pairs | report_components |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2018-05-01 00:00:00+05:30 | 2018-05-07 23:00:00+05:30 | 50.066158 | 3.714655 | 7 | 7 | finite | 0.31647 | [4.211047,33.624407,47.148512,25.778945,23.044029,22.271834,20.982911] | [] | [] | [[0],[1],[2],[3]] |
| 2 | 2018-05-08 00:00:00+05:30 | 2018-05-14 23:00:00+05:30 | 60.253055 | 5.670984 | 7 | 7 | finite | 0.355326 | [6.031379,33.711076,57.217343,25.844248,24.159022,22.973226,21.761241] | [] | [] | [[0],[1],[2],[3]] |
| 3 | 2018-05-15 00:00:00+05:30 | 2018-05-21 23:00:00+05:30 | 77.004576 | 10.154152 | 7 | 7 | finite | 0.495568 | [10.617332,40.186352,72.603524,25.986977,24.809319,23.361772,21.428431] | [] | [] | [[0],[1],[2],[3]] |
| 4 | 2018-05-22 00:00:00+05:30 | 2018-05-28 23:00:00+05:30 | 81.200623 | 7.646498 | 7 | 7 | finite | 0.475097 | [7.954414,36.187254,76.721497,27.85015,23.31544,26.530985,24.76379] | [] | [] | [[0],[1],[2],[3]] |

- _No source-activity ground truth: geometry/residuals/groups only._

## New Delhi Proxy Apportionment (fraction of fitted sensor signal)

| week | brick_kilns | industries | population_density | traffic |
| --- | --- | --- | --- | --- |
| 1 | 0.0 | 0.0 | 1.0 | 0.0 |
| 2 | 0.0 | 0.157134 | 0.805365 | 0.0375 |
| 3 | 0.05587 | 0.0 | 0.94413 | 0.0 |
| 4 | 0.934903 | 0.065097 | 0.0 | 0.0 |

- _Shares are fractions of fitted inventory-attributed sensor signal, NOT physical-emission shares._
- _Denominator: sum over groups of L1 fitted per-group sensor-signal magnitude._
- _Groups with no admissible temporal component are marked unsupported._

## New Delhi Sensor Fit and Residual Diagnostics (weeks 1-4)

| week | n_observed_rows | n_total_rows | observed_mask_fraction | residual_norm | projected_residual_norm | kriged_baseline_subtracted | pm25_imputed | wind_provider | calibration_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4561 | 5376 | 0.8484 | 3768.200619 |  | True | False | gridded_kernel_new_delhi | uncalibrated: no external calibrated noise model exists for observed PM2.5, so per the paper's adequacy contract the residual adequacy test emits no pass/fail verdict |
| 2 | 4278 | 5376 | 0.795759 | 5955.26548 |  | True | False | gridded_kernel_new_delhi | uncalibrated: no external calibrated noise model exists for observed PM2.5, so per the paper's adequacy contract the residual adequacy test emits no pass/fail verdict |
| 3 | 4315 | 5376 | 0.802641 | 3049.873933 |  | True | False | gridded_kernel_new_delhi | uncalibrated: no external calibrated noise model exists for observed PM2.5, so per the paper's adequacy contract the residual adequacy test emits no pass/fail verdict |
| 4 | 4236 | 5376 | 0.787946 | 3760.168265 |  | True | False | gridded_kernel_new_delhi | uncalibrated: no external calibrated noise model exists for observed PM2.5, so per the paper's adequacy contract the residual adequacy test emits no pass/fail verdict |

- _PM2.5 is never imputed. Uncalibrated: no adequacy pass is presented._

## New Delhi Per-Monitor Fitted Group Contributions (projected sensor signal)

| week | monitor | brick_kilns | industries | population_density | traffic |
| --- | --- | --- | --- | --- | --- |
| 1 | reg_0_19 | 0.0 | 0.0 | -206.551562 | 0.0 |
| 1 | reg_11_0 | 0.0 | 0.0 | -136.641945 | 0.0 |
| 1 | reg_11_39 | 0.0 | 0.0 | -282.825462 | 0.0 |
| 1 | reg_12_13 | 0.0 | 0.0 | 1262.347513 | 0.0 |
| 1 | reg_12_37 | 0.0 | 0.0 | 3.702619 | 0.0 |
| 1 | reg_14_28 | 0.0 | 0.0 | 1846.046252 | 0.0 |
| 1 | reg_16_24 | 0.0 | 0.0 | 258.239437 | 0.0 |
| 1 | reg_16_25 | 0.0 | 0.0 | 938.447786 | 0.0 |
| 1 | reg_18_32 | 0.0 | 0.0 | 1387.08437 | 0.0 |
| 1 | reg_19_37 | 0.0 | 0.0 | -419.282677 | 0.0 |
| 1 | reg_21_13 | 0.0 | 0.0 | -79.369691 | 0.0 |
| 1 | reg_21_31 | 0.0 | 0.0 | -262.372199 | 0.0 |
| 1 | reg_23_23 | 0.0 | 0.0 | 256.566114 | 0.0 |
| 1 | reg_23_36 | 0.0 | 0.0 | -475.624566 | 0.0 |
| 1 | reg_25_11 | 0.0 | 0.0 | -280.016916 | 0.0 |
| 1 | reg_27_17 | 0.0 | 0.0 | -184.093795 | 0.0 |
| 1 | reg_28_15 | 0.0 | 0.0 | -323.595703 | 0.0 |
| 1 | reg_28_20 | 0.0 | 0.0 | -101.327903 | 0.0 |
| 1 | reg_30_14 | 0.0 | 0.0 | -237.05531 | 0.0 |
| 1 | reg_30_22 | 0.0 | 0.0 | -52.8208 | 0.0 |
| 1 | reg_30_33 | 0.0 | 0.0 | -470.829662 | 0.0 |
| 1 | reg_32_4 | 0.0 | 0.0 | -494.33947 | 0.0 |
| 1 | reg_33_11 | 0.0 | 0.0 | -414.775174 | 0.0 |
| 1 | reg_33_8 | 0.0 | 0.0 | -645.181465 | 0.0 |
| 1 | reg_35_21 | 0.0 | 0.0 | 118.037815 | 0.0 |
| 1 | reg_37_29 | 0.0 | 0.0 | -457.439651 | 0.0 |
| 1 | reg_39_25 | 0.0 | 0.0 | -577.072582 | 0.0 |
| 1 | reg_39_28 | 0.0 | 0.0 | -608.363188 | 0.0 |
| 1 | reg_5_14 | 0.0 | 0.0 | 639.107814 | 0.0 |
| 2 | reg_0_19 | 0.0 | -232.589993 | -884.243476 | -47.210043 |
| 2 | reg_11_0 | 0.0 | -197.43419 | -675.138199 | -36.40506 |
| 2 | reg_11_39 | 0.0 | -104.313736 | -534.209607 | -26.561746 |
| 2 | reg_12_13 | 0.0 | 92.716338 | 2488.695005 | 224.777314 |
| 2 | reg_12_37 | 0.0 | 10.33084 | -175.226479 | -34.339611 |
| 2 | reg_14_28 | 0.0 | 30.245638 | 3202.217656 | 5.92758 |
| 2 | reg_16_24 | 0.0 | -63.88316 | 756.135348 | 42.663491 |
| 2 | reg_16_25 | 0.0 | -68.410661 | 2006.471547 | 20.941693 |
| 2 | reg_18_32 | 0.0 | 812.507541 | 2197.072013 | -3.140909 |
| 2 | reg_19_37 | 0.0 | -44.433733 | -830.82802 | -24.201712 |
| 2 | reg_21_13 | 0.0 | -85.990274 | -89.885327 | -2.234793 |
| 2 | reg_21_31 | 0.0 | 217.911791 | -472.640998 | -6.544542 |
| 2 | reg_23_23 | 0.0 | -163.168164 | 574.370882 | -0.425855 |
| 2 | reg_23_36 | 0.0 | -64.511338 | -631.78919 | -2.979409 |
| 2 | reg_25_11 | 0.0 | -15.186144 | -117.281875 | -4.458508 |
| 2 | reg_27_17 | 0.0 | -83.562116 | 202.100233 | -4.029745 |
| 2 | reg_28_15 | 0.0 | -45.94317 | -96.929252 | -12.932426 |
| 2 | reg_28_20 | 0.0 | -147.83032 | 210.85701 | 247.919935 |
| 2 | reg_30_14 | 0.0 | -40.20719 | -246.586927 | -7.911509 |
| 2 | reg_30_22 | 0.0 | -122.182251 | 25.326531 | 2.579618 |
| 2 | reg_30_33 | 0.0 | -112.533602 | -869.112161 | -25.821512 |
| 2 | reg_32_4 | 0.0 | 478.149175 | -767.186213 | -21.097505 |
| 2 | reg_33_11 | 0.0 | -132.66371 | -905.876003 | -38.389568 |
| 2 | reg_33_8 | 0.0 | 327.80601 | -1481.180819 | -36.412845 |
| 2 | reg_35_21 | 0.0 | -211.757969 | 12.963241 | -32.929358 |
| 2 | reg_37_29 | 0.0 | 106.105194 | -1103.915362 | -50.353418 |
| 2 | reg_39_25 | 0.0 | -188.318127 | -1122.970742 | -47.877157 |
| 2 | reg_39_28 | 0.0 | -217.729085 | -1405.716778 | -51.907969 |
| 2 | reg_5_14 | 0.0 | 266.876406 | 734.507961 | -26.644429 |
| 3 | reg_0_19 | -1.091652 | 0.0 | -31.055388 | 0.0 |
| 3 | reg_11_0 | -3.165036 | 0.0 | -47.628163 | 0.0 |
| 3 | reg_11_39 | 4.301618 | 0.0 | -87.055093 | 0.0 |
| 3 | reg_12_13 | 1.61901 | 0.0 | 242.759114 | 0.0 |
| 3 | reg_12_37 | 11.522576 | 0.0 | -66.914896 | 0.0 |
| 3 | reg_14_28 | -2.996507 | 0.0 | 199.607581 | 0.0 |
| 3 | reg_16_24 | -2.011873 | 0.0 | 16.200407 | 0.0 |
| 3 | reg_16_25 | -2.660712 | 0.0 | 123.814688 | 0.0 |
| 3 | reg_18_32 | 3.076781 | 0.0 | 135.991482 | 0.0 |
| 3 | reg_19_37 | 0.350937 | 0.0 | -76.793915 | 0.0 |
| 3 | reg_21_13 | -3.503572 | 0.0 | 68.726808 | 0.0 |
| 3 | reg_21_31 | -0.79688 | 0.0 | -51.244695 | 0.0 |
| 3 | reg_23_23 | -3.737797 | 0.0 | 16.629017 | 0.0 |
| 3 | reg_25_11 | -0.465803 | 0.0 | 6.556428 | 0.0 |
| 3 | reg_27_17 | -2.234071 | 0.0 | -6.988466 | 0.0 |
| 3 | reg_28_15 | -0.575448 | 0.0 | -2.138678 | 0.0 |
| 3 | reg_28_20 | 19.351197 | 0.0 | -23.053673 | 0.0 |
| 3 | reg_30_14 | 7.555307 | 0.0 | 1.117239 | 0.0 |
| 3 | reg_30_22 | -2.396385 | 0.0 | -9.786582 | 0.0 |
| 3 | reg_30_33 | -3.416176 | 0.0 | -87.321279 | 0.0 |
| 3 | reg_32_4 | -1.659003 | 0.0 | -59.890543 | 0.0 |
| 3 | reg_33_11 | -2.713029 | 0.0 | -48.638777 | 0.0 |
| 3 | reg_33_8 | -3.165263 | 0.0 | -89.364529 | 0.0 |
| 3 | reg_35_21 | -2.453144 | 0.0 | -10.710844 | 0.0 |
| 3 | reg_37_29 | -1.087857 | 0.0 | -62.457747 | 0.0 |
| 3 | reg_39_25 | -3.460208 | 0.0 | -82.517655 | 0.0 |
| 3 | reg_39_28 | -2.28683 | 0.0 | -76.89388 | 0.0 |
| 3 | reg_5_14 | -1.90018 | 0.0 | 109.05204 | 0.0 |
| 4 | reg_0_19 | -27.421476 | -7.065071 | 0.0 | 0.0 |
| 4 | reg_11_0 | -80.081031 | -7.919522 | 0.0 | 0.0 |
| 4 | reg_11_39 | 130.073721 | -0.988813 | 0.0 | 0.0 |
| 4 | reg_12_13 | -28.045941 | 4.467489 | 0.0 | 0.0 |
| 4 | reg_12_37 | 210.543025 | 11.566189 | 0.0 | 0.0 |
| 4 | reg_14_28 | -58.680922 | -4.329771 | 0.0 | 0.0 |
| 4 | reg_16_24 | -29.39852 | -2.480201 | 0.0 | 0.0 |
| 4 | reg_16_25 | -65.948471 | -5.784498 | 0.0 | 0.0 |
| 4 | reg_18_32 | -79.145391 | 19.27505 | 0.0 | 0.0 |
| 4 | reg_19_37 | -51.102963 | -3.193174 | 0.0 | 0.0 |
| 4 | reg_21_13 | -10.229209 | 0.971971 | 0.0 | 0.0 |
| 4 | reg_21_31 | -73.05319 | -2.262313 | 0.0 | 0.0 |
| 4 | reg_23_23 | -67.895559 | -5.67431 | 0.0 | 0.0 |
| 4 | reg_25_11 | -42.791633 | -0.777219 | 0.0 | 0.0 |
| 4 | reg_27_17 | -1.440142 | -0.463551 | 0.0 | 0.0 |
| 4 | reg_28_15 | 25.879114 | -1.864181 | 0.0 | 0.0 |
| 4 | reg_28_20 | 552.629726 | -1.504439 | 0.0 | 0.0 |
| 4 | reg_30_14 | 214.023375 | -2.503206 | 0.0 | 0.0 |
| 4 | reg_30_22 | -37.161195 | -3.298845 | 0.0 | 0.0 |
| 4 | reg_30_33 | -59.644711 | -7.135732 | 0.0 | 0.0 |
| 4 | reg_32_4 | -14.256627 | 13.679406 | 0.0 | 0.0 |
| 4 | reg_33_11 | -51.413475 | -3.229462 | 0.0 | 0.0 |
| 4 | reg_33_8 | -57.179217 | 10.804208 | 0.0 | 0.0 |
| 4 | reg_35_21 | -40.449109 | -5.221705 | 0.0 | 0.0 |
| 4 | reg_37_29 | -73.820408 | 5.143129 | 0.0 | 0.0 |
| 4 | reg_39_25 | -73.871751 | -5.79355 | 0.0 | 0.0 |
| 4 | reg_39_28 | -52.733862 | -3.311361 | 0.0 | 0.0 |
| 4 | reg_5_14 | -57.384158 | 8.893483 | 0.0 | 0.0 |

- _Projected (P_Q^perp) per-monitor contributions can be signed; the reported apportionment shares use L1 magnitudes._

## New Delhi Wind-Imputation Validation

| metric | value |
| --- | --- |
| status | unavailable -- paper-facing wind is the kernel coordinate-query imputer (FieldFormer evaluated but not adopted; see reconciliation) |

- _Dense real wind truth is unavailable; gridded-field accuracy is assessed in controlled synthetic-wind experiments (Experiment 4)._
