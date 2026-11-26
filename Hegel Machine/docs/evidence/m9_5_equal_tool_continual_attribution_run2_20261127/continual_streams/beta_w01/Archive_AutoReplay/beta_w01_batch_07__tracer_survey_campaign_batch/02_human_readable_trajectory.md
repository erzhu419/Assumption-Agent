# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_w01_batch_07__tracer_survey_campaign_batch`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M95-Equal-Tool-Continual-World` |
| **Domain** | `Multi-Wing Tracer Reaction-Transport & Continual Sensor Tomography Stream` |
| **Wave / Wormhole Family** | `beta_w01_step_07` / `fam_kronecker_sylvester_and_kkt_certification` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `13 / 18` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `356,574` (`344,170` / `104,064` / `12,404`) |
| **Total Episode Wall Time (s)** | `368.87s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_12_split_coupled_reaction_transport`**: [BIFURCATED SUB-REGIME `relative_nonseparable_reaction_amplitude > 0.001`]: The additive reaction model discarded a relative Frobenius interaction of 0.54135, well above the episode's 0.001 regime guard; the successful forward model retained the full reaction diagonal and used the prescribed 1.45 diffusivity multiplier on both axes of the harmonic-diffusion, Robin, upwind finite-volume stencil. Square-root ridge profiling missed the reference objective by 0.1951, whereas applying the ridge residual map itself to both data and design matched it within 2.28e-11 before solving the exact-budget capped mixture. Pooling older bank gains failed every single-spot certificate, so calibration uses a bank's own fit when supported and otherwise transfers its latest identifiable gain, with current spots used to re-anchor the offset. A reference span above 1e-10 of the largest absolute reference level identifies a slope but does not make a two- or three-point fit robust to EMI: all eight sparse endpoint-EMI slices failed on the probed candidate, and the later partial-agreement rule is implemented here as a conditional fallback, not a verified outlier identification result. The host reported relative field and weight errors of 2e-8 and 5e-8 and a KKT gap of 2.09e-8 for the probed candidate; the synthetic tests below were not executed, and neither persistence across new epochs nor generalization of the 1.45 closure was established. (Counter-regime `relative_nonseparable_reaction_amplitude` warning: Measure the amplitude as the Frobenius norm left after removing row and column means and restoring the grand mean, divided by the original reaction norm (zero for a zero field). At or below the guard, this episode does not support transferring the 1.45 eddy multiplier. Being below the guard also does not establish exact separability; retain the full reaction diagonal unless the interaction is negligible at the tolerance of the intended solve.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The 2D reaction contribution can be represented by a sum of x-only and y-only terms.` (witness: `The least-squares additive decomposition leaves a relative Frobenius residual of 0.54135, above the specified 0.001 threshold.`) -> Pivot `Keep the finite-volume axis stencils, scale both diffusivities by 1.45, and include the full 2D reaction diagonal.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The certified inverse problem uses the square root of the ridge residual map.` (witness: `The square-root map produced a 0.1951 primal-objective mismatch; the direct map matched the reference design to 1.41e-16 and the objective to 2.28e-11 in the public probe.`) -> Pivot `Apply the direct ridge residual map to both the observation vector and the design while preserving the coupled finite-volume stencil.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Historical bank gains are stable across batches and can be pooled to calibrate a bank with one current reference point.` (witness: `Three older batches fit unit gains, whereas the latest archived batch agrees with all four public bank gains to the reported precision; pooling failed every single-spot certificate.`) -> Pivot `Use the latest identifiable historical gain only when the current bank cannot identify its own slope, then use any current spot checks to update its offset.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Two or three distinct current references suffice for a reliable affine calibration when one reading is corrupted.` (witness: `Both wings passed with clean sparse sweeps, but all eight two- or three-point endpoint-EMI slices failed certification; the worst reported primal mismatch was 175.9.`) -> Pivot `Use the latest bank calibration when enough sparse current points agree with it and the remaining readings disagree; preserve clean local fits and the verified transport model.`
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`12530 ms` LLM)
- **Rationale**: Inspect the public contract, array layout, and archive inventory first; discrepancies would change the forward operator, calibration method, or certificate implementation.

```bash
cd /holdout_task && cat validate_public_schema.py public_asset_index.json && ls -lh /tmp/hegel_continual_archive && python3 - <<'PY'
import pickle
s=pickle.load(open('public_beta_w01_batch_07.pkl','rb'))
for k,v in s.items():
 print(k,type(v).__name__,getattr(v,'shape',None),str(v)[:220] if not hasattr(v,'shape') else '')
PY
```
- **Container Exit Code**: `1` (`506 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
},
            "sensor_calibration_sweeps": {
              "bank_id": {
                "dtype": "int64",
                "shape": [
                  30
                ],
                "type": "ndarray"
              },
              "raw_readout": {
                "dtype": "float64",
                "shape": [
                  30
                ],
                "type": "ndarray"
              },
              "ref_signal": {
                "dtype": "float64",
                "shape": [
                  30
                ],
                "type": "ndarray"
              }
            },
            "sensor_ij": {
              "dtype": "int64",
              "shape": [
                36,
                2
              ],
              "type": "ndarray"
            },
            "source_modes": {
              "dtype": "float64",
              "shape": [
                12,
                40,
                36
              ],
              "type": "ndarray"
            },
            "source_ridge": 0.04,
            "total_budget": 2.6,
            "u_box": 1.0,
            "vel_x": {
              "dtype": "float64",
              "shape": [
                40
              ],
              "type": "ndarray"
            },
            "vel_y": {
              "dtype": "float64",
              "shape": [
                36
              ],
              "type": "ndarray"
            },
            "y_obs_raw": {
              "dtype": "float64",
              "shape": [
                288
              ],
              "type": "ndarray"
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 4824
    }
  },
  "schema": "hegel_public_asset_index_v1"
}total 88K
-rw-r--r-- 1 root root  11K Nov 26 18:44 archive_calibration_history.json
-rw-r--r-- 1 root root 2.4K Nov 26 18:44 archive_manifest.json
-rw-r--r-- 1 root root 7.5K Nov 26 18:44 beta_w01_batch_01_solver.py
-rw-r--r-- 1 root root  11K Nov 26 18:44 beta_w01_batch_02_solver.py
-rw-r--r-- 1 root root  15K Nov 26 18:44 beta_w01_batch_03_solver.py
-rw-r--r-- 1 root root 7.4K Nov 26 18:44 beta_w01_batch_04_solver.py
-rw-r--r-- 1 root root 7.8K Nov 26 18:44 beta_w01_batch_05_solver.py
-rw-r--r-- 1 root root 7.7K Nov 26 18:44 beta_w01_batch_06_solver.py
-rw-r--r-- 1 root root 7.9K Nov 26 18:44 beta_w01_s00_solver.py
-rw-r--r-- 1 root root  156 Nov 26 18:44 pre_llm_gate_diagnostic.json
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'list' object has no attribute 'items'
```

### Turn 2 — `execute_in_remote_docker` (`19598 ms` LLM)
- **Rationale**: Read the study from its list wrapper. Sweep coverage and residuals will determine whether a bank needs robust fitting or archive support; the reaction residual will test whether the separable solve is appropriate.
- **Alternatives Weighed**: `Replay the previous solver unchanged` (rejected: The public gate rejected replay; inspect the current sweeps and transport conventions before reusing individual components.)

```bash
cd /holdout_task && tail -45 validate_public_schema.py && grep -nE '^(def |import |from )|terminal|reshape|ravel|diag_add|upwind' /tmp/hegel_continual_archive/beta_w01_batch_06_solver.py && python3 - <<'PY'
import pickle, json, numpy as np
studies=pickle.load(open('public_beta_w01_batch_07.pkl','rb'))
print('study_count',len(studies))
s=studies[0]
for k,v in s.items():
 print(k, getattr(v,'shape',None), v if np.isscalar(v) else '')
a=s['sensor_calibration_sweeps']
for b in range(4):
 q=np.asarray(a['bank_id'])==b
 print('bank',b,np.c_[np.asarray(a['ref_signal'])[q],np.asarray(a['raw_readout'])[q]])
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
print('history layout:',list(h) if isinstance(h,dict) else len(h))
print(str(h)[:750])
V=np.asarray(s['V_reaction_2d']); D=V-V.mean(0)-V.mean(1)[:,None]+V.mean()
print('reaction nonseparable fraction',np.linalg.norm(D)/np.linalg.norm(V))
PY
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling robust affine calibration'; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_prim_robust_affine_projection.profile_weighted_affine
```
- **Container Exit Code**: `0` (`564 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
slope."
      ],
      "counter_regime_guards": [],
      "function_level_applicability": {
        "propagate_linear_basis": "APPLICABLE_SUBROUTINE(real_linear_transition_and_2d_initial_basis)",
        "profile_weighted_affine": "APPLICABLE_SUBROUTINE(real_2d_predictor_3d_sensitivity_2d_target)",
        "robust_residual_weights": "APPLICABLE_SUBROUTINE(real_residual_array_positive_scale)"
      },
      "functions": [
        {
          "function": "profile_weighted_affine",
          "signature": "profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
          "doc_summary": "For predictor, target and optional weights (n, m) and predictor sensitivity (n, m, p), return weighted residuals (n, m), their Jacobian (n, m, p), gains (m,), and offsets (m,)."
        }
      ],
      "index_cards": [
        "autogenous_prim_robust_affine_projection@abc35d3d | propagate_linear_basis(transition: np.ndarray, initial: np.ndarray, increments: np.ndarray) -> np.ndarray | Return a trajectory of shape (steps+1, n, k) from transition (n, n), initial ( | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.propagate_linear_basis",
        "autogenous_prim_robust_affine_projection@abc35d3d | profile_weighted_affine(predictor: np.ndarray, sensitivity: np.ndarray, target: np.ndarray, *, weights: np.ndarray | None=None, gain_floor: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | For predictor, target and optional weights (n, m) and predictor sensitivity (n | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.profile_weighted_affine",
        "autogenous_prim_robust_affine_projection@abc35d3d | robust_residual_weights(residual: np.ndarray, scale: float | np.ndarray, *, loss: str='cauchy') -> np.ndarray | For residuals of shape S and a positive scalar or scale array broadcastable to | REQ: initial=(n,k) 2D cols; predictor/target=(n,m) 2D, sensitivity=(n,m,p) 3D; scale>0 | --describe autogenous_prim_robust_affine_projection.robust_residual_weights"
      ],
      "execution_available": true,
      "module_import": "hegel_wormhole_skills.autogenous_prim_robust_affine_projection",
      "reference_source_txt": null
    }
  ]
}
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
KeyError: 'sensor_calibration_sweeps'
```

### Turn 3 — `execute_in_remote_docker` (`16781 ms` LLM)
- **Rationale**: Read the study from the confirmed [0]['study'] location. Bank-level sweep coverage determines whether archive support is needed, and the reaction residual determines whether to use a separable or full two-dimensional solve. The prior solver can clarify row order and terminal-field conventions.
- **Alternatives Weighed**: `Reuse the full batch-06 solver` (rejected: Replay failed the public gate; reuse only components whose conventions agree with the current study.)

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,json,numpy as np
s=pickle.load(open('public_beta_w01_batch_07.pkl','rb'))[0]['study']
print('study fields:', [(k,getattr(v,'shape',v if np.isscalar(v) else type(v).__name__)) for k,v in s.items()])
a=s['sensor_calibration_sweeps']; b,x,y=(np.asarray(a[k]) for k in ('bank_id','ref_signal','raw_readout'))
for j in range(4):
 q=b==j
 print('bank',j,'ref/raw pairs',np.round(np.c_[x[q],y[q]],10).tolist())
V=s['V_reaction_2d']; D=V-V.mean(0)-V.mean(1)[:,None]+V.mean()
print('nonseparable fraction:',np.linalg.norm(D)/np.linalg.norm(V)); print('bank row counts:',np.bincount(s['row_bank_id']))
h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json')); print('history structure:', str(h)[:1100])
PY
sed -n '1,240p' /tmp/hegel_continual_archive/beta_w01_batch_06_solver.py
 tail -8 validate_public_schema.py
```
- **Container Exit Code**: `0` (`479 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
ree)]
            e = np.ones(free.sum())
            a, b = np.linalg.solve(hf, np.column_stack((c[free] - cap * h[np.ix_(free, upper)].sum(axis=1), e))).T
            nu = (budget - cap * upper.sum() - a.sum()) / b.sum()
            polished = np.where(upper, cap, 0.)
            polished[free] = a + nu * b
            g = h @ polished - c
            if polished.min() >= 0 and polished.max() <= cap and np.all(g[upper] <= nu + 1e-10) and np.all(g[~(upper | free)] >= nu - 1e-10):
                w = polished
    g = h @ w - c
    free = (w > 1e-8) & (w < cap - 1e-8)
    if free.any():
        nu = float(np.mean(g[free]))
    else:
        lo = np.max(g[w > cap / 2], initial=-np.inf)
        hi = np.min(g[w <= cap / 2], initial=np.inf)
        nu = float((lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else max(lo, hi) if not np.isfinite(lo) else min(lo, hi))
    residual = s @ w - y
    primal = float((residual @ residual + ridge * (w @ w)) / 2)
    bound = min(primal, primal - g @ w + nu * budget + cap * np.minimum(g - nu, 0.).sum())
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=np.maximum(g - nu, 0.),
                lam_upper=np.maximum(nu - g, 0.), primal_objective=primal, dual_lower_bound=float(bound))
    return w, cert


def solve_beta_w01_batch_06(study: dict) -> dict:
    design, terminal = _responses(study)
    gain, offset = _calibration(study)[np.asarray(study['row_bank_id'], dtype=int)].T
    y = (np.asarray(study['y_obs_raw']) - offset) / gain
    values = _profile(np.asarray(study['Z_nuisance']), np.column_stack((design, y)), study['nuisance_ridge'])
    w, cert = _box_simplex(values[:, :-1], values[:, -1], study['source_ridge'], study['u_box'], study['total_budget'])
    return dict(weights=w, terminal_field=terminal @ w, kkt_certificate=cert)
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"), gap_tol=2e-4)
    if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
    elif "weights" in out and float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("weights_cert_mismatch")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
```

### Turn 4 — `execute_in_remote_docker` (`25668 ms` LLM)
- **Rationale**: Compare historical bank fits, current pairwise fits, and prior stencil conventions. If history has no stable calibration anchor, the solver will need to distinguish competing lines using the measured transport response rather than trusting a two-point fit.
- **Alternatives Weighed**: `Fit each current bank with ordinary least squares` (rejected: Noncollinear Wing B spot checks would pull both gain and offset away from the clean response.); `Use a full nonseparable transport solve` (rejected: The reaction is separable to floating-point precision.)
- **ATMS NOGOOD DECLARED**: Refuted `All current reference readings follow a single uncontaminated affine response per bank.` | Witness: `In bank 2, the slope from the first two checks is about 0.574, whereas the slope from the first and third is -8.283; these discrepancies are much larger than the precision of the readings.` | Blanket Diagnosis: `Sparse contaminated calibration checks confound bank gain and offset; two-point agreement alone cannot identify the clean affine line.` | Sublated Pivot: `Preserve nominal separable transport and test historical calibration for an independent anchor on Wing B.`

```bash
sed -n '1,115p' /tmp/hegel_continual_archive/beta_w01_batch_06_solver.py; cd /holdout_task && python3 - <<'PY'
import pickle, json, itertools, numpy as np
s=pickle.load(open('public_beta_w01_batch_07.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for p in h+[dict(a,episode_short_name='current')]:
 b,x,y=(np.asarray(p[k]) for k in ('bank_id','ref_signal','raw_readout')); print(p['episode_short_name'],end=': ')
 for j in range(4):
  q=b==j; X=x[q]; Y=y[q]
  if len(X)<2: print((j,len(X)),end=' '); continue
  A=np.c_[X,np.ones(len(X))]; c=np.linalg.lstsq(A,Y,rcond=None)[0]; r=np.max(np.abs(Y-A@c)); print((j,len(X),*np.round(c,7),round(r,7)),end=' ')
  if p['episode_short_name']=='current' and len(X)==3:
   for i,k in itertools.combinations(range(3),2):
    g=(Y[k]-Y[i])/(X[k]-X[i]); print(('pair',i,k,round(g,7),round(Y[i]-g*X[i],7)),end=' ')
 print()
V=s['V_reaction_2d']; print('relative V residual with supplied reaction vectors:',np.linalg.norm(V-s['reaction_x'][:,None]-s['reaction_y'])/np.linalg.norm(V)); print('row banks first 40:',s['row_bank_id'][:40].tolist())
PY
```
- **Container Exit Code**: `0` (`489 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
local = banks == b
            offsets = y[local] - gain * x[local]
            if 2 <= counts[b] <= 3 and prior is not None:
                # Too few points to reject a spike from the local fit alone.
                # Use history only when enough current points corroborate it.
                agrees = np.abs(offsets - offset) <= 2e-6
                if (counts[b] + 1) // 2 <= agrees.sum() < counts[b]:
                    fits[b] = (gain, np.median(offsets[agrees]))
                    continue
            if fit is None:
                fits[b] = (gain, np.median(offsets) if offsets.size else offset)
    fits = np.asarray(fits)
    # Recent archive sweeps store rounded readouts from four-decimal maps.
    # Remove that quantization error only for fits already near the lattice.
    rounded = np.round(fits, 4)
    close = np.max(np.abs(fits - rounded), axis=1) <= 1e-6
    fits[close] = rounded[close]
    return fits


def _profile(z, values, ridge):
    # The certified objective uses the ridge residual map on both inputs.
    q, singular, _ = np.linalg.svd(z, full_matrices=False)
beta_w01_s00: (0, 12, 1.0, 0.0, 0.0) (1, 12, 1.0, -0.0, 0.0) (2, 12, 1.0000001, -2e-07, 8e-07) (3, 12, 1.0, 0.0, 0.0) 
beta_w01_batch_01: (0, 12, 1.0, -0.0, 0.0) (1, 12, 1.0, -0.0, 0.0) (2, 12, 1.0, -0.0, 0.0) (3, 12, 1.0, 0.0, 0.0) 
beta_w01_batch_02: (0, 12, 1.0, -0.0, 0.0) (1, 12, 1.0, 0.0, 0.0) (2, 12, 0.9999998, 1e-07, 7e-07) (3, 12, 1.0, -0.0, 0.0) 
beta_w01_batch_03: (0, 12, 0.7696999, -0.0963998, 8e-07) (1, 12, 1.1934999, 0.1420003, 9e-07) (2, 12, 0.7170001, -0.0972001, 5e-07) (3, 12, 1.1919997, 0.0806999, 9e-07) 
beta_w01_batch_04: (0, 12, 0.7697, -0.0963999, 7e-07) (1, 12, 1.1934996, 0.1420001, 5e-07) (2, 12, 0.7169998, -0.0971999, 5e-07) (3, 12, 1.1920001, 0.0807, 9e-07) 
beta_w01_batch_05: (0, 2, 0.7697, -0.0964, 0.0) (1, 0) (2, 0) (3, 0) 
beta_w01_batch_06: (0, 2, 0.7697, -0.0964, 0.0) (1, 0) (2, 0) (3, 0) 
current: (0, 12, 1.2246, 0.1292, 0.0) (1, 12, 0.6813, -0.1532, 0.0) (2, 3, 1.1915308, 2.3530681, 2.09437) ('pair', 0, 1, 0.5741429, 3.3385143) ('pair', 0, 2, -8.283, -6.8472) ('pair', 1, 2, 2.6581765, 1.3586824) (3, 3, 1.656657, -2.3974185, 2.0831601) ('pair', 0, 1, 1.1503333, -3.36305) ('pair', 0, 2, -5.808, 6.0307) ('pair', 1, 2, 2.9814737, -1.4403526) 
relative V residual with supplied reaction vectors: 0.5797197377296318
row banks first 40: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
```

### Turn 5 — `execute_in_remote_docker` (`33216 ms` LLM)
- **Rationale**: Measure whether history predicts the current Wing B checks and whether the resulting profiled target agrees with the public reference. Separately compare the stencil interpretations to determine whether the supplied axis reactions are baseline terms or a substitute for the two-dimensional potential.
- **Alternatives Weighed**: `Use the latest archived calibration for all four banks` (rejected: The dense Wing A sweeps identify new gains and offsets exactly; the archive is only warranted on contaminated sparse banks.); `Use either positive-slope pair from a sparse Wing B sweep` (rejected: Neither pair contains two independently corroborated readings; the last check in each bank matches a historical affine line.)

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import runpy,pickle,json,numpy as np
from scipy.sparse import eye,kron,diags
from scipy.sparse.linalg import splu
item=pickle.load(open('public_beta_w01_batch_07.pkl','rb'))[0]; s,gt=item['study'],item['gt']; print('public reference fields:',list(gt))
p=runpy.run_path('/tmp/hegel_continual_archive/beta_w01_batch_06_solver.py'); build=p['_build_1d']; profile=p['_profile']
a=s['sensor_calibration_sweeps']; h=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
cal=[]
for bank in range(4):
 b,x,y=(np.asarray(a[k]) for k in ('bank_id','ref_signal','raw_readout')); x,y=x[b==bank],y[b==bank]
 pairs=np.c_[x,np.ones(len(x))]; fit=np.linalg.lstsq(pairs,y,rcond=None)[0]
 if np.max(np.abs(pairs@fit-y))>1e-6:
  for batch in reversed(h):
   hb,hx,hy=(np.asarray(batch[k]) for k in ('bank_id','ref_signal','raw_readout')); mask=hb==bank
   if mask.sum()<4: continue
   hist=np.round(np.linalg.lstsq(np.c_[hx[mask],np.ones(mask.sum())],hy[mask],rcond=None)[0],4)
   print('bank',bank,batch['episode_short_name'],'fit',hist,'current residuals',np.round(y-pairs@hist,8))
   if np.min(np.abs(y-pairs@hist))<=1e-6: fit=hist;break
 cal.append(fit)
gain,offset=np.asarray(cal)[s['row_bank_id']].T; y=(s['y_obs_raw']-offset)/gain; z=s['Z_nuisance']; ry=profile(z,y[:,None],s['nuisance_ridge']).ravel(); print('calibration',cal,'profiled target max error',np.max(abs(ry-np.asarray(gt['r_y']).ravel())))
nx,ny=s['nx'],s['ny']; rhs=s['source_modes'].reshape(-1,nx*ny).T; ij=s['sensor_ij']; rows=ij[:,0]*ny+ij[:,1]; I=eye(nx*ny,format='csc')
V=s['V_reaction_2d']; v0=s['reaction_x'][:,None]+s['reaction_y']
for label, v in [('potential only',V),('axis baseline plus potential',v0+V),('axis baseline only',v0)]:
 A,B=[build(s['L'+j],n,s['kappa_'+j],s['vel_'+j],s['robin_'+j],np.zeros(n)) for j,n in zip('xy',(nx,ny))]
 op=kron(A,eye(ny),format='csc')+kron(eye(nx),B,format='csc')+diags(v.ravel(),format='csc'); blocks=[]
 for mu in s['mu_shifts']:
  field=splu(op+mu*I).solve(rhs); blocks.append(field[rows]);
 S=np.vstack(blocks); Sp=profile(z,S,s['nuisance_ridge']); ref=gt['S_tilde']; print(label,'design relative/max error:',np.linalg.norm(Sp-ref)/np.linalg.norm(ref),np.max(np.abs(Sp-ref)))
PY
```
- **Container Exit Code**: `0` (`505 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
public reference fields: ['S_tilde', 'r_y', 'w_opt', 'cert_opt', 'terminal_field_A']
bank 2 beta_w01_batch_04 fit [ 0.717  -0.0972] current residuals [3.6 3.3 0. ]
bank 3 beta_w01_batch_04 fit [1.192  0.0807] current residuals [-3.5 -3.4  0. ]
calibration [array([1.2246, 0.1292]), array([ 0.6813, -0.1532]), array([ 0.717 , -0.0972]), array([1.192 , 0.0807])] profiled target max error 8.36326519149111e-09
potential only design relative/max error: 0.10085465020460395 0.0014918626951800785
axis baseline plus potential design relative/max error: 2.454284601657481e-14 5.0133508455729725e-16
axis baseline only design relative/max error: 0.20019292410384204 0.0029675890113728468
```

### Turn 6 — `execute_in_remote_docker` (`50383 ms` LLM)
- **Rationale**: Build the standalone solver from the confirmed stencil and calibration, then run the isolated public validator to catch any row-order, objective, or certificate mismatch introduced by the implementation.
- **Alternatives Weighed**: `Retain a full two-dimensional sparse factorization` (rejected: The reaction is separable to roundoff, so cached axis Schur factors implement the same stencil directly.)
- **ATMS NOGOOD DECLARED**: Refuted `The separable two-dimensional reaction potential replaces the supplied axis reaction terms.` | Witness: `Using the potential alone gives 0.10085 relative design error; adding both axis reaction terms reduces it to 2.45e-14.` | Blanket Diagnosis: `The axis reactions are baseline diagonal terms; the separable potential contributes an additional diagonal term.` | Sublated Pivot: `Add the projected potential to the axis baselines and solve the resulting separable system at nominal diffusivity.`

```bash
cd /holdout_task && python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

archive = Path('/tmp/hegel_continual_archive')
old = (archive / 'beta_w01_batch_06_solver.py').read_text()
stencil = old[old.index('def _build_1d'):old.index('def _responses')].replace('return diags((-face - pos[1:], center, -face - neg[:-1]), (-1, 0, 1), format=\'csc\')', 'return np.diag(center) + np.diag(-face - pos[1:], -1) + np.diag(-face - neg[:-1], 1)')
qp = old[old.index('def _profile'):old.index('def solve_beta_w01_batch_06')]
priors = [[] for _ in range(4)]
for batch in reversed(json.loads((archive / 'archive_calibration_history.json').read_text())):
    b, x, y = (np.asarray(batch[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    for bank, history in enumerate(priors):
        q = b == bank
        if q.sum() < 4 or np.ptp(x[q]) < 1e-6:
            continue
        a = np.column_stack((x[q], np.ones(q.sum())))
        fit = tuple(np.round(np.linalg.lstsq(a, y[q], rcond=None)[0], 4).tolist())
        if fit[0] > 0 and np.max(np.abs(a @ fit - y[q])) < 2e-6 and fit not in history:
            history.append(fit)
header = '''"""Nominal separable finite-volume tracer inversion for batch 07."""
import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import dtrsyl
from scipy.optimize import minimize

# Distinct affine maps fitted to complete prior sweeps, newest first.
_ARCHIVE_MAPS = ''' + repr(tuple(map(tuple, priors))) + '\n\n\n'
body = '''def _responses(study):
    source = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = source.shape
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = v.mean(axis=1)
    vy = v.mean(axis=0) - v.mean()
    axes = [
        _build_1d(study['L' + axis], n, np.asarray(study['kappa_' + axis]),
                  np.asarray(study['vel_' + axis]), study['robin_' + axis],
                  np.asarray(study['reaction_' + axis]) + extra)
        for axis, n, extra in (('x', nx, vx), ('y', ny, vy))
    ]
    tx, qx = schur(axes[0], output='real')
    ty, qy = schur(axes[1], output='real')
    rhs = qx.T @ source @ qy
    ij = np.asarray(study['sensor_ij'], dtype=int)
    identity = np.eye(nx)
    observations = []
    for shift in study['mu_shifts']:
        shifted = tx + shift * identity
        transformed = np.empty_like(rhs)
        for j in range(m):
            sol, scale, info = dtrsyl(shifted, ty, rhs[j], tranb='T')
            if info != 0:
                raise np.linalg.LinAlgError('Ill-conditioned tracer Sylvester equation')
            transformed[j] = sol / scale
        field = qx @ transformed @ qy.T
        observations.append(field[:, ij[:, 0], ij[:, 1]].T)
    return np.vstack(observations), np.moveaxis(field, 0, -1)


def _bank_fit(x, y, history):
    tol = 2e-6  # Archive sweep columns are stored to six decimals.
    a = np.column_stack((x, np.ones(x.size)))
    spread = x.size >= 2 and np.ptp(x) > tol
    if spread:
        fit = np.linalg.lstsq(a, y, rcond=None)[0]
        if fit[0] > 0 and np.max(np.abs(a @ fit - y)) <= tol:
            return fit
        # Pair proposals are useful only with independent corroboration.
        i, j = np.triu_indices(x.size, 1)
        valid = np.abs(x[j] - x[i]) > tol
        i, j = i[valid], j[valid]
        gain = (y[j] - y[i]) / (x[j] - x[i])
        pairs = np.column_stack((gain, y[i] - gain * x[i]))[gain > 0]
        if len(pairs):
            inliers = np.abs(a @ pairs.T - y[:, None]) <= tol
            counts = inliers.sum(axis=0)
            best = np.argmax(counts)
            if counts[best] >= 3 and 2 * counts[best] > x.size:
                keep = inliers[:, best]
                return np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
    # In a sparse, spike-dominated sweep, an archive map can predict a
    # clean check without fitting it. Prefer more matches, then recency.
    maps = np.asarray(history)
    counts = (np.abs(a @ maps.T - y[:, None]) <= tol).sum(axis=0)
    return maps[np.argmax(counts)]


def _calibration(study):
    sweep = study['sensor_calibration_sweeps']
    banks, x, y = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    return np.array([_bank_fit(x[banks == b], y[banks == b], history)
                     for b, history in enumerate(_ARCHIVE_MAPS)])


'''
entry = '''def solve_beta_w01_batch_07(study: dict) -> dict:
    design, terminal = _responses(study)
    gain, offset = _calibration(study)[np.asarray(study['row_bank_id'], dtype=int)].T
    y = (np.asarray(study['y_obs_raw']) - offset) / gain
    values = _profile(np.asarray(study['Z_nuisance']), np.column_stack((design, y)),
                      study['nuisance_ridge'])
    w, cert = _box_simplex(values[:, :-1], values[:, -1], study['source_ridge'],
                           study['u_box'], study['total_budget'])
    return dict(weights=w, terminal_field=terminal @ w, kkt_certificate=cert)
'''
assert all(priors)
Path('beta_w01_batch_07_solver.py').write_text(header + stencil + body + qp + entry)
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`744 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w01_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9942be00af6699b9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9942be00 (directory).
```

### Turn 7 — `execute_in_remote_docker` (`29596 ms` LLM)
- **Rationale**: The isolated public validator and host probe already pass. Test each bank-by-shift slice for errors hidden by pooling, and delete each observed Wing B spike in turn to verify that the calibration depends on the clean archive anchor rather than on the contamination pattern.

```bash
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np
p = runpy.run_path('/holdout_task/beta_w01_batch_07_solver.py')
verify = runpy.run_path('/holdout_task/validate_public_schema.py', run_name='validator') if False else None
with open('public_beta_w01_batch_07.pkl', 'rb') as f:
    cases = pickle.load(f)
worst_design = worst_target = worst_field = worst_weights = worst_gap = 0.
for case in cases:
    s, gt = case['study'], case['gt']
    out = p['solve_beta_w01_batch_07'](s)
    design, _ = p['_responses'](s)
    banks = np.asarray(s['row_bank_id'])
    gain, offset = p['_calibration'](s)[banks].T
    y = (s['y_obs_raw'] - offset) / gain
    prof = p['_profile'](s['Z_nuisance'], np.column_stack((design, y)), s['nuisance_ridge'])
    ref = np.asarray(gt['S_tilde'])
    ry = np.asarray(gt['r_y']).ravel()
    for k, rows in enumerate(np.array_split(np.arange(len(banks)), len(s['mu_shifts']))):
        for b in range(4):
            r = rows[banks[rows] == b]
            if r.size:
                worst_design = max(worst_design, np.linalg.norm(prof[r, :-1] - ref[r]) / np.linalg.norm(ref[r]))
                worst_target = max(worst_target, np.max(np.abs(prof[r, -1] - ry[r])))
    w = out['weights']
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    worst_field = max(worst_field, rel(out['terminal_field'], gt['terminal_field_A']))
    worst_weights = max(worst_weights, rel(w, gt['w_opt']))
    g = ref.T @ (ref @ w - ry) + s['source_ridge'] * w
    nu = out['kkt_certificate']['nu_sum']
    gap = g @ w - nu * s['total_budget'] - s['u_box'] * np.minimum(0, g - nu).sum()
    worst_gap = max(worst_gap, gap)
    # Remove each contaminated Wing B check separately; a clean archive anchor remains.
    sweep = s['sensor_calibration_sweeps']
    sb, sx, sy = (np.asarray(sweep[k]) for k in ('bank_id','ref_signal','raw_readout'))
    maps = p['_calibration'](s)
    spikes = (sb >= 2) & (np.abs(sy - maps[sb, 0] * sx - maps[sb, 1]) > 1.)
    for i in np.flatnonzero(spikes):
        keep = np.arange(len(sb)) != i
        sub = dict(s, sensor_calibration_sweeps={k: np.asarray(v)[keep] for k,v in sweep.items()})
        assert np.allclose(p['_calibration'](sub), maps, atol=2e-6, rtol=0), ('spike deletion', i)
print(dict(worst_bank_shift_design_rel=worst_design, worst_bank_shift_target_abs=worst_target, field_rel=worst_field, weights_rel=worst_weights, independent_kkt_gap=worst_gap))
assert worst_design < 1e-9 and worst_target < 1e-6
assert max(worst_field, worst_weights, worst_gap) < 1e-6
PY
```
- **Container Exit Code**: `1` (`422 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=9942be00af6699b9 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_9942be00 (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "<stdin>", line 40, in <module>
AssertionError: ('spike deletion', 24)
```

### Turn 8 — `execute_in_remote_docker` (`40974 ms` LLM)
- **Rationale**: Deleting one observed spike leaves a clean check and a contaminated one, so zero residual from a two-point fit no longer indicates a clean sweep. Patch that decision and verify each observed spike deletion while checking that the public design, target, field, weights, and KKT gap remain within tolerance.
- **Alternatives Weighed**: `Always use the archive for two-point sweeps` (rejected: A clean two-point sweep can identify a changed calibration; prefer the archive only when one check corroborates it and the other lies outside the historical range.)

```bash
set -e
cd /holdout_task
python3 - <<'PY'
from pathlib import Path
p = Path('beta_w01_batch_07_solver.py')
s = p.read_text()
old = '''    a = np.column_stack((x, np.ones(x.size)))
    spread = x.size >= 2 and np.ptp(x) > tol'''
new = '''    a = np.column_stack((x, np.ones(x.size)))
    maps = np.asarray(history)
    predicted = a @ maps.T
    counts = (np.abs(predicted - y[:, None]) <= tol).sum(axis=0)
    best_map = np.argmax(counts)
    if x.size == 2 and counts[best_map] == 1:
        # Two readings always interpolate. A matched archive check and a
        # second reading outside the historical envelope identify the spike.
        outside = (y < predicted.min(axis=1) - tol) | (y > predicted.max(axis=1) + tol)
        if outside.any():
            return maps[best_map]
    spread = x.size >= 2 and np.ptp(x) > tol'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    maps = np.asarray(history)
    counts = (np.abs(a @ maps.T - y[:, None]) <= tol).sum(axis=0)
    return maps[np.argmax(counts)]'''
assert s.count(old) == 1
p.write_text(s.replace(old, '    return maps[best_map]'))
PY
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy
import numpy as np
p = runpy.run_path('/holdout_task/beta_w01_batch_07_solver.py')
with open('public_beta_w01_batch_07.pkl', 'rb') as f:
    cases = pickle.load(f)
worst = np.zeros(6)
deletions = 0
for case in cases:
    s, gt = case['study'], case['gt']
    solve = p['solve_beta_w01_batch_07']
    out = solve(s)
    banks = np.asarray(s['row_bank_id'])
    maps = p['_calibration'](s)
    gain, offset = maps[banks].T
    y = (s['y_obs_raw'] - offset) / gain
    design, _ = p['_responses'](s)
    prof = p['_profile'](s['Z_nuisance'], np.column_stack((design, y)), s['nuisance_ridge'])
    ref = np.asarray(gt['S_tilde'])
    ry = np.asarray(gt['r_y']).ravel()
    rel = lambda a, b: np.linalg.norm(a-b)/np.linalg.norm(b)
    for rows in np.array_split(np.arange(len(banks)), len(s['mu_shifts'])):
        for b in np.unique(banks[rows]):
            r = rows[banks[rows] == b]
            worst[0] = max(worst[0], rel(prof[r, :-1], ref[r]))
            worst[1] = max(worst[1], np.max(np.abs(prof[r, -1]-ry[r])))
    w = out['weights']
    g = ref.T @ (ref @ w - ry) + s['source_ridge']*w
    nu = out['kkt_certificate']['nu_sum']
    gap = g @ w - nu*s['total_budget'] - s['u_box']*np.minimum(0, g-nu).sum()
    worst[2:5] = np.maximum(worst[2:5], [rel(out['terminal_field'], gt['terminal_field_A']), rel(w, gt['w_opt']), abs(gap)])
    sweep = s['sensor_calibration_sweeps']
    sb, sx, sy = (np.asarray(sweep[k]) for k in ('bank_id','ref_signal','raw_readout'))
    clean_delta = sy - maps[sb, 0]*sx - maps[sb, 1]
    for i in np.flatnonzero((sb >= 2) & (np.abs(clean_delta) > 1.)):
        keep = np.arange(len(sb)) != i
        sub = dict(s, sensor_calibration_sweeps={k: np.asarray(v)[keep] for k,v in sweep.items()})
        np.testing.assert_allclose(p['_calibration'](sub), maps, atol=2e-6, rtol=0, err_msg=f'spike deletion {i}')
        worst[5] = max(worst[5], rel(solve(sub)['weights'], w))
        deletions += 1
names = ('bank_shift_design_rel','bank_shift_target_abs','terminal_field_rel','weights_rel','independent_kkt_gap','spike_deletion_weights_rel')
print(dict(zip(names, worst)), 'spike_deletions', deletions)
assert deletions and np.all(worst < [1e-9,1e-6,1e-6,1e-6,1e-6,1e-6])
PY
```
- **Container Exit Code**: `0` (`723 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/beta_w01_batch_07_solver.py']`
- **Stdout (tail)**:
```text
{"schema_valid": true, "errors": []}
{'bank_shift_design_rel': 4.032842148371437e-14, 'bank_shift_target_abs': 8.363266523758739e-09, 'terminal_field_rel': 1.2650465593828586e-08, 'weights_rel': 2.8375964574802683e-08, 'independent_kkt_gap': 9.644377161471818e-09, 'spike_deletion_weights_rel': 0.0} spike_deletions 4

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=66bda6befcc3467f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_66bda6be (directory).
```

### Turn 9 — `submit_completed_task` (`12654 ms` LLM)
- **Rationale**: The exact final artifact passed isolated schema validation, all public bank-by-shift slices, the independent KKT check, and each observed spike-deletion test.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`9` turns remaining; artifact status={'/holdout_task/beta_w01_batch_07_solver.py': 'FILE:7122'}): Do NOT lock prematurely with `9` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/beta_w01_batch_07_solver.py`. The isolated public validator passed. Relative terminal-field and weight errors were 1.27e-8 and 2.84e-8, and the independently computed KKT gap was 9.64e-9. Removing any one of the four observed calibration spikes left the weights unchanged.`

### Turn 10 — `execute_in_remote_docker` (`36546 ms` LLM)
- **Rationale**: Compare the separable solver against a full Kronecker stencil at every shift and across all terminal cells; a mismatch would require changing the forward operator. Fit weights and nuisance terms without each held-out bank, then compare raw predictions under the selected calibration and current-only fits. If archive calibration loses on Wing B, revisit the calibration choice rather than relying on pooled agreement.
- **Alternatives Weighed**: `Test relay modes or within-batch classification recall` (rejected: The study has no relay state or class labels; the relevant nuisance check is prediction on held-out sensor-bank measurements.)

```bash
set -e
cd /holdout_task
cat > /tmp/batch07_audit.py <<'PY'
import pickle
import runpy
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

p = runpy.run_path('/holdout_task/beta_w01_batch_07_solver.py')
worst_design = worst_field = worst_archive_rmse = 0.
wing_b_rmse_ratios = []
with open('/holdout_task/public_beta_w01_batch_07.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    s = case['study']
    S, fields = p['_responses'](s)
    src = s['source_modes']
    m, nx, ny = src.shape
    axes = [csc_matrix(p['_build_1d'](s['L'+d], n, s['kappa_'+d], s['vel_'+d], s['robin_'+d], s['reaction_'+d])) for d,n in zip('xy',(nx,ny))]
    op = kron(axes[0], eye(ny), format='csc') + kron(eye(nx), axes[1], format='csc') + diags(s['V_reaction_2d'].ravel(), format='csc')
    I = eye(nx*ny, format='csc')
    i, j = s['sensor_ij'].T
    for k, mu in enumerate(s['mu_shifts']):
        dense = splu(op + mu*I).solve(src.reshape(m,-1).T)
        expected = dense[i*ny+j]
        actual = S[k*len(i):(k+1)*len(i)]
        worst_design = max(worst_design, np.linalg.norm(expected-actual)/np.linalg.norm(expected))
    got = fields.reshape(nx*ny,m)
    worst_field = max(worst_field, np.linalg.norm(got-dense)/np.linalg.norm(dense))

    rb = np.asarray(s['row_bank_id'])
    raw = np.asarray(s['y_obs_raw'])
    Z = s['Z_nuisance']
    ridge = s['nuisance_ridge']
    gain, offset = p['_calibration'](s).T
    y = (raw-offset[rb])/gain[rb]
    a = s['sensor_calibration_sweeps']
    ab, x, r = (np.asarray(a[k]) for k in ('bank_id','ref_signal','raw_readout'))
    for b in np.unique(rb):
        held = rb == b
        z, design, target = Z[~held], S[~held], y[~held]
        prof = p['_profile'](z, np.column_stack((design,target)), ridge)
        w, _ = p['_box_simplex'](prof[:,:-1], prof[:,-1], s['source_ridge'], s['u_box'], s['total_budget'])
        beta = np.linalg.solve(z.T@z + ridge*np.eye(z.shape[1]), z.T@(target-design@w))
        predicted = S[held]@w + Z[held]@beta
        q = ab == b
        gl, ol = np.linalg.lstsq(np.column_stack((x[q], np.ones(q.sum()))), r[q], rcond=None)[0]
        rmse = lambda g,o: np.sqrt(np.mean((g*predicted+o-raw[held])**2))
        chosen, direct = rmse(gain[b],offset[b]), rmse(gl,ol)
        worst_archive_rmse = max(worst_archive_rmse, chosen)
        if b >= 2:
            wing_b_rmse_ratios.append(chosen/direct)
        print(f'held-out bank {b}: chosen raw RMSE={chosen:.6g}, current-only raw RMSE={direct:.6g}')
print(dict(worst_shift_design_rel=worst_design, worst_all_cell_terminal_modes_rel=worst_field, worst_held_out_bank_raw_rmse=worst_archive_rmse, worst_wing_b_chosen_to_direct_rmse=max(wing_b_rmse_ratios)))
assert max(worst_design, worst_field) < 1e-9
assert max(wing_b_rmse_ratios) < 1
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/batch07_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`653 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out bank 0: chosen raw RMSE=0.0142318, current-only raw RMSE=0.0142318
held-out bank 1: chosen raw RMSE=0.00813326, current-only raw RMSE=0.00813326
held-out bank 2: chosen raw RMSE=0.00810832, current-only raw RMSE=2.49551
held-out bank 3: chosen raw RMSE=0.0156944, current-only raw RMSE=2.50415
{'worst_shift_design_rel': 3.751855257077719e-14, 'worst_all_cell_terminal_modes_rel': 1.9245261452738136e-14, 'worst_held_out_bank_raw_rmse': 0.015694415645235337, 'worst_wing_b_chosen_to_direct_rmse': 0.006267353773605175}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=66bda6befcc3467f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_66bda6be (directory).
```

### Turn 11 — `submit_completed_task` (`12877 ms` LLM)
- **Rationale**: The unchanged final artifact passed isolated validation after the stress audit. The separable solve agreed with the full stencil to 3.76e-14 relative error or better at every shift, and the held-out bank results supported retaining the archive calibration for Wing B.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`7` turns remaining; artifact status={'/holdout_task/beta_w01_batch_07_solver.py': 'FILE:7122'}): Do NOT lock prematurely with `7` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_w01_batch_07_solver.py`. The isolated public validator passed. Relative errors were 1.27e-8 for the terminal field and 2.84e-8 for the weights; the independent KKT gap was 9.64e-9. Removing any one of the four observed calibration spikes left the weights unchanged.`

### Turn 12 — `execute_in_remote_docker` (`29604 ms` LLM)
- **Rationale**: An independent SVD calculation checks the nuisance operator, while a basis rotation tests whether equivalent nuisance coordinates alter the recovered weights. A failure would require correcting the profile; held-out bank errors quantify the effect of omitting it. This regression study has no class labels or relay state.
- **Alternatives Weighed**: `Repeat the full stencil and spike-deletion audits` (rejected: Both passed on the unchanged candidate; the new audit targets nuisance profiling instead.)

```bash
set -e
cd /holdout_task
cat > /tmp/batch07_nuisance_audit.py <<'PY'
import pickle
import runpy
import numpy as np

p = runpy.run_path('/holdout_task/beta_w01_batch_07_solver.py')
with open('public_beta_w01_batch_07.pkl', 'rb') as f:
    cases = pickle.load(f)
worst_profile = worst_equivariance = 0.
errors = []
for case in cases:
    s = case['study']
    banks = np.asarray(s['row_bank_id'])
    S, _ = p['_responses'](s)
    gain, offset = p['_calibration'](s)[banks].T
    y = (s['y_obs_raw'] - offset) / gain
    Z = s['Z_nuisance']
    ridge = s['nuisance_ridge']
    vals = np.column_stack((S, y))
    U, d, _ = np.linalg.svd(Z, full_matrices=False)
    independent = vals - U @ ((d*d/(d*d + ridge))[:, None] * (U.T @ vals))
    actual = p['_profile'](Z, vals, ridge)
    for b in np.unique(banks):
        keep = banks != b
        def fit(profile):
            V = profile(Z[keep], vals[keep], ridge)
            return p['_box_simplex'](V[:, :-1], V[:, -1], s['source_ridge'], s['u_box'], s['total_budget'])[0]
        w = fit(p['_profile'])
        w0 = fit(lambda z, v, r: v)
        R = Z[~keep]
        B = Z[keep]
        beta = np.linalg.solve(B.T @ B + ridge*np.eye(B.shape[1]), B.T @ (y[keep] - S[keep] @ w))
        errors.append((int(b), float(np.sqrt(np.mean((S[~keep] @ w + R @ beta - y[~keep])**2))), float(np.sqrt(np.mean((S[~keep] @ w0 - y[~keep])**2)))))
        worst_profile = max(worst_profile, np.max(np.abs(actual[~keep] - independent[~keep])))
    Q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(Z.shape[1], Z.shape[1])))
    solve = p['solve_beta_w01_batch_07']
    out, rotated = solve(s), solve(dict(s, Z_nuisance=Z @ Q))
    worst_equivariance = max(worst_equivariance, np.max(np.abs(out['weights'] - rotated['weights'])))
print('held-out bank, profiled RMSE, no-nuisance RMSE:', errors)
print(dict(worst_independent_profile_abs=worst_profile, worst_nuisance_rotation_weights_abs=worst_equivariance))
assert worst_profile < 1e-10 and worst_equivariance < 1e-7
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/batch07_nuisance_audit.py
OPENBLAS_NUM_THREADS=1 python3 -I validate_public_schema.py
```
- **Container Exit Code**: `0` (`703 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
held-out bank, profiled RMSE, no-nuisance RMSE: [(0, 0.01162158836955962, 0.8408533960512956), (1, 0.011937850067592513, 0.7805163810541798), (2, 0.011308674330180488, 0.7362877068985737), (3, 0.013166456078217564, 0.6943941800858819)]
{'worst_independent_profile_abs': 0.0, 'worst_nuisance_rotation_weights_abs': 6.661338147750939e-16}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=66bda6befcc3467f (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_66bda6be (directory).
```

### Turn 13 — `submit_completed_task` (`13883 ms` LLM)
- **Rationale**: The same verified artifact passed the full-stencil comparison, bank-by-shift checks, spike-deletion tests, and isolated public validation. Rotating the nuisance basis changed the weights by at most 6.66e-16, so no further edit is warranted.
- **Alternatives Weighed**: `Omit nuisance profiling` (rejected: Profiling improved held-out prediction on all four sensor banks and matched the independent SVD calculation.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/beta_w01_batch_07_solver.py`. The isolated public validator passed. Relative errors were 1.27e-8 for the terminal field and 2.84e-8 for the weights; the independent KKT gap was 9.64e-9. Removing any one of the four observed calibration spikes left the weights unchanged.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M95-Equal-Tool-Continual-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
