# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `beta_m99_c03_batch_05__blind_continual_campaign__downstream_module`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World` |
| **Domain** | `coupled_2d_transport_reaction_box_simplex_kkt` |
| **Wave / Wormhole Family** | `m9_9_autonomous_decomposition_and_interaction_withdrawal` / `m99_beta` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `12 / 18` |
| **ATMS Nogoods Recorded** | `1` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `337,785` (`327,816` / `150,656` / `9,969`) |
| **Total Episode Wall Time (s)** | `303.72s` |

## 2. Recalled PhilosophicalForest Wormhole Notes & Wave-1 Sublations

- **`node_auto_08_split_sparse_ridge_certification`**: [BIFURCATED SUB-REGIME `row_incidence_component_count > 1`]: Sibling compaction could not repair the residual-only perspective gap: the seven-variable orthogonal tie still required 128 leaves after 448 sibling tests, because the relaxation discarded separable loss curvature. A Gershgorin-certified diagonal part of the Gram matrix restores that curvature without assuming an estimated eigenvalue is a valid bound, but coordinate-wise bounds still produced 545 leaves for eight independent duplicate pairs. For exactly row-disjoint blocks with at most eight columns each, minimize the support-wise box-ridge dual bounds inside each block and add the block minima, preserving the joint indicator choices. A coupled 16-column instance still had 101 leaves despite a 1.78e-15 gap, so when compaction exceeds the 90-leaf budget, certify the root using the minimum of separate residual-dual bounds over all supports; return a certificate only when every reported cube has a lower bound at least the incumbent upper bound minus eps. (Counter-regime `row_incidence_component_count` warning: Additive block certification requires exactly disjoint row sets, and the shortcut enumerates blocks only when the largest has at most eight columns. A connected design needs joint certification; a small off-diagonal Gram entry is not evidence of row independence.)
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Sibling compaction alone can enforce the 90-leaf cap on the orthogonal tie.` (witness: `The 7-variable orthogonal tie still used 128 leaves after testing 448 sibling pairs.`) -> Pivot `Retain a certified diagonal part of the Gram matrix in the box-ridge conjugate, while keeping the residual-only bound as a fallback.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `Diagonal loss curvature and sibling compaction suffice to meet the cap for independent blocks of interchangeable columns.` (witness: `Eight independent duplicate pairs required 545 leaves although the certified objective matched the analytic optimum.`) -> Pivot `Certify each small independent block with the minimum of its support-wise box-ridge duals and sum the independent block bounds.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The local perspective tree with independent-block bounds meets the 90-leaf limit on coupled full-dimension designs.` (witness: `Normalized dense seed 15 required 101 leaves after compaction, with a gap of 1.78e-15.`) -> Pivot `When the compacted tree exceeds 90 leaves, evaluate the joint disjunctive box-ridge dual at the root using separate residual duals for every support.`
- **`node_auto_05_conservative_elliptic_multig`**: No empirical nogoods were recorded, so the episode does not establish that a particular baseline failed or that a permeability-contrast threshold separates physical regimes. Harmonic conductivity assigns a single flux to each interior face, giving equal and opposite contributions to its neighboring cells; zero-valued ghosts with face conductivity equal to the adjacent cell permeability give a symmetric positive-definite pressure operator. The solver preconditions matrix-free CG with a symmetric Galerkin V-cycle: it aggregates normalized couplings of at least 0.08, retries aggregation on all nonzero couplings if the coarse level retains more than 76% of the unknowns, and uses dense Cholesky only at 64 unknowns or fewer. A directly recomputed residual, not the CG recurrence, gates acceptance: exceeding 0.85 times the requested tolerance triggers residual correction, and the solver raises rather than returning a pressure whose recomputed residual exceeds the requested tolerance.
- **`node_auto_07_control_panel_deconfounding`**: Raw pooled fitting learned a scanner–prevalence shortcut: in center 3, balanced accuracy rose from 0.650 to 0.845 after within-center alignment and fitting separate training-center intercepts. Directly centering each signal feature was also unsafe because its marginal location depends on class composition; replacing that step with control-panel alignment raised worst-center leave-one-center-out balanced accuracy from 0.861 to 0.930 with the same five-term basis. Estimate each center’s shared affine scanner transform from the median of control-feature means and the median of control-feature population standard deviations, requiring at least two observations per contributing control and a positive estimated gain; retain the interaction and both constituent main effects after alignment. Give each training center–class cell equal total weight for imputation and L2-regularized logistic fitting (C=3), interpret 0.5 under equal class priors rather than as a clinical prevalence estimate, and assess transfer against the 0.84 balanced-accuracy target separately in every held-out center.
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw pooled feature coordinates support a transferable decision boundary.` (witness: `In center 3 the raw pooled model reached 0.650 balanced accuracy, versus 0.845 after within-center alignment and fitting separate training-center intercepts.`) -> Pivot `Retain the specified biological terms and isolate scanner alignment, biological ranking, and neutral threshold calibration.`
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Per-feature test-center centering removes only scanner variation.` (witness: `With the same five-term basis, control-panel alignment raised worst-center LOCO balanced accuracy from 0.861 to 0.930.`) -> Pivot `Estimate scanner offset and gain from the 43 non-signal features, then fit the contrast, interaction with both main effects, and remaining signal using equal class weights within training centers.`
- **`node_auto_04_creep_corrected_retraction_c`**: On the recorded 8 N trace, apparent compliance rose from 0.01597 to 0.05741 µm/N across unloading windows, so a single steady hold-rate correction could not explain both slopes. Remove frame compliance, exclude the 15 ms transition window, and estimate the terminal hold rate robustly so displacement spikes do not set the creep correction. On a linear force ramp, constant-rate creep and elastic compliance are collinear: finite-saturation-only fits gave 404–442% stiffness errors on the recorded constant-rate stresses, so fix the independently measured hold rate when the corrected compliance is positive and binned residual curvature is at most 0.8 times the hold-noise MAD scale. Otherwise fit saturating creep with its onset derivative fixed to the hold rate; for loading slips, require a robust local level change greater than 0.35 µm rather than accepting a raw first-difference spike. [REGIME SPLIT GUARD `rms_binned_median_corrected_unloading_residual_over_hold_noise <= 0.8` — Counter-Regime Warning: The fixed-rate branch also requires positive corrected compliance. Otherwise the fit uses an onset-constrained saturation model; resolved curvature alone does not distinguish creep saturation from elastic nonlinearity.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `The steady hold creep rate can be added unchanged to a broad-window unloading slope.` (witness: `For spec_A at 8 N, apparent compliance is 0.01597 near onset and 0.05741 later, while steady hold creep is about 0.420 µm/s and the unloading rate is 4.509 N/s. The same additive correction cannot explain both windows.`) -> Pivot `Keep the measured hold rate as an onset constraint, exclude the transition samples, and compare robust unloading fits with and without force curvature.`
  - *Wave-1 Consolidated Lesson*: [RUNG_2_INCOMPLETE_DISCRETE_GRAMMAR_OR_SYMMETRY] Refuted `The finite saturation fit covers the constant-rate correction in practice.` (witness: `The constant-rate stress curves returned stiffness errors of 404–442%, while the corresponding finite-saturation curves stayed within 3.15%.`) -> Pivot `Test whether coherent unloading curvature is resolvable above sensor noise, and use the exact hold-rate correction when it is not.`
- **`node_auto_09_dispersive_wave_tomography`**: Phase-invariant late-envelope energy plus a wall-arrival mask was not a reliable localizer: the three non-twin normalized errors were 0.1826, 0.4397, and 0.4120, versus 0.0051 after matched-twin subtraction. The non-twin solver separates the direct packet and first wall echoes before matching residual packets with free complex amplitudes, and scores direct attenuation with standardized sensor gains under a 1.8 ridge so an unpenalized gain fit cannot freely explain away a near-sensor shadow. Full waveform subtraction is reserved for matching geometry, aligned records and pair channels, and late-coda correlation strictly above 0.78; otherwise the reference still supplies an empirical pulse envelope and a group-slowness prior transferred by the square root of the thickness ratio when available. Carrier-period timing and plate-span geometry make the operator covariant under coordinated frequency and length changes, while nonparallel ray support and the residualized packet energy govern how much localization evidence each modality contributes. [REGIME SPLIT GUARD `median_late_coda_correlation_on_geometry_matched_aligned_paths > 0.78` — Counter-Regime Warning: Correlation alone cannot authorize subtraction across geometries. If the twin gate fails, retain reference pulse and timing calibration when available, fit direct and wall packets, and combine residual scatter with ridge-adjusted shadow evidence.]
  - *Wave-1 Consolidated Lesson*: [RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM] Refuted `Raw late-envelope energy after image-source wall gating is a reliable localization observable for non-twin cases.` (witness: `Coda-only errors were 0.1826, 0.4397, and 0.4120 in the three non-twin cases, versus 0.0051 after matched-twin subtraction.`) -> Pivot `Keep matched-twin subtraction and compare early packet residuals after separately estimating the direct shape and wall background.`

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`12948 ms` LLM)
- **Rationale**: Inspect the public input, upstream interface, and apparatus log to identify the active boundary rule and avoid composing the solver in raw coordinates.

```bash
ls -l /holdout_task /tmp/hegel_continual_archive; cat /holdout_task/{batch_solver.py,_verified_upstream.py,_upstream_param_payload.json,validate_public_schema.py}; cat /tmp/hegel_continual_archive/archive_apparatus_log.json; PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'ridge nuisance profiling'; cd /holdout_task; python3 - <<'PY'
import pickle
b=pickle.load(open('public_batch.pkl','rb')); print('items',len(b))
for item in b[:1]:
 print(item.keys(), 'diagnostics', type(item['campaign_diagnostics']))
 for k,v in item['study'].items(): print(k,type(v),getattr(v,'shape',None),list(v) if isinstance(v,dict) else v if isinstance(v,(str,int,float)) else '')
PY
```
- **Container Exit Code**: `0` (`400 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9418,
        0.86553,
        2.62025,
        0.70698,
        -0.40437
      ],
      [
        -0.33522,
        0.837,
        0.75835,
        -0.83301,
        0.63613,
        2.62455,
        0.47886,
        0.02862
      ],
      [
        -0.27426,
        0.63565,
        0.85017,
        -0.83184,
        0.33211,
        2.56179,
        0.28134,
        0.49257
      ],
      [
        -0.22849,
        0.32274,
        0.46096,
        -0.85886,
        -0.14425,
        2.59234,
        0.06056,
        0.71407
      ],
      [
        -0.4223,
        -0.14882,
        -0.14751,
        -0.76186,
        -0.56544,
        2.46866,
        -0.14676,
        0.90611
      ],
      [
        -0.48914,
        -0.58374,
        -0.67011,
        -0.66905,
        -0.74056,
        2.49207,
        -0.37695,
        0.75193
      ],
      [
        -0.43396,
        -0.83681,
        -0.80377,
        -0.48374,
        -0.86967,
        2.57447,
        -0.52445,
        0.48803
      ]
    ]
  }
]autogenous_sparse_ridge_certification@770f6644 | solve_and_certify_l0(X: np.ndarray, y: np.ndarray, lam0: float, lam2: float, M: float, eps: float=1e-06, *, max_leaves: int=90) -> dict | Certify boxed sparse ridge regression with at most 16 real predictors | REQ: {'field': 'n_samples', 'op': '>=', 'value': 1}; {'field': 'n_features', 'op': '<=', 'val | --describe autogenous_sparse_ridge_certification.solve_and_certify_l0
items 1
dict_keys(['study', 'campaign_diagnostics']) diagnostics <class 'list'>
Lx <class 'float'> None 2.4
Ly <class 'float'> None 2.0
nx <class 'int'> None 14
ny <class 'int'> None 12
kappa_x <class 'numpy.ndarray'> (14,) 
kappa_y <class 'numpy.ndarray'> (12,) 
vel_x <class 'numpy.ndarray'> (14,) 
vel_y <class 'numpy.ndarray'> (12,) 
robin_x <class 'float'> None 1.0240161106792895
robin_y <class 'float'> None 1.1758119992635399
V_reaction_2d <class 'numpy.ndarray'> (14, 12) 
mu_shifts <class 'numpy.ndarray'> (3,) 
source_modes <class 'numpy.ndarray'> (8, 14, 12) 
sensor_ij <class 'numpy.ndarray'> (32, 2) 
row_bank_id <class 'numpy.ndarray'> (96,) 
sensor_calibration_sweeps <class 'dict'> None ['ref_rosette', 'raw_rosette', 'single_bank_sweeps']
Z_nuisance <class 'numpy.ndarray'> (96, 3) 
y_obs_raw <class 'numpy.ndarray'> (96,) 
aux_manifold_telemetry <class 'numpy.ndarray'> (24, 4) 
nuisance_ridge <class 'float'> None 0.08
source_ridge <class 'float'> None 0.04
u_box <class 'float'> None 0.42
total_budget <class 'float'> None 1.4
```

### Turn 2 — `execute_in_remote_docker` (`13843 ms` LLM)
- **Rationale**: The public item has tracer pulses. Inspect the calibration interface and archived boundary formulas so the next test can distinguish operator mismatch from calibration error.

```bash
cd /holdout_task; cat _verified_upstream.py validate_public_schema.py; cat /tmp/hegel_continual_archive/archive_manifest.json; for f in /tmp/hegel_continual_archive/*solver.py; do echo ====; grep -nE '^def |robin|boundary|diag\[' "$f"; done; python3 - <<'PY'
import json,pickle,numpy as np
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
a=json.load(open(p)); print(type(a),len(a));
for x in a:
 print({k: (v if not isinstance(v,(dict,list)) or len(v)<8 else (np.shape(v) if isinstance(v,list) else list(v))) for k,v in x.items()})
i=pickle.load(open('public_batch.pkl','rb'))[0]; print('pulses',len(i['campaign_diagnostics'])); print([(k,np.shape(v)) for k,v in i['campaign_diagnostics'][0].items()]); print(json.load(open('_upstream_param_payload.json')))
PY
```
- **Container Exit Code**: `0` (`538 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
"terminal_field",
      "weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py"
  },
  {
    "episode_short_name": "beta_m99_c03_batch_01",
    "task_id": "beta_m99_c03_batch_01__blind_continual_campaign",
    "entrypoint": "solve_batch",
    "output_schema_keys": [
      "kkt_certificate",
      "terminal_field",
      "weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/beta_m99_c03_batch_01_solver.py"
  },
  {
    "episode_short_name": "beta_m99_c03_batch_02",
    "task_id": "beta_m99_c03_batch_02__blind_continual_campaign",
    "entrypoint": "solve_downstream",
    "output_schema_keys": [
      "kkt_certificate",
      "terminal_field",
      "weights"
    ],
    "solver_path": "/tmp/hegel_continual_archive/beta_m99_c03_batch_02_solver.py"
  }
]====
8:def _axis_operator(kappa, velocity, length):
23:def _forward(study):
41:def _calibrated_observations(study):
57:def _box_simplex(H, c, cap, budget):
103:def solve_batch(study: dict) -> dict:
====
8:def _axis_operator(kappa, velocity, length):
19:def _forward(study):
36:def _box_simplex(H, c, cap, budget):
82:def solve_downstream(study: dict) -> dict:
====
8:def _axis_operator(kappa, velocity, robin, length):
16:    diag = np.r_[kappa[0] / h**2 + robin / h,
18:                 kappa[-1] / h**2 + robin / h]
22:def _forward(study):
25:                             study['robin_' + a], study['L' + a]) for a in 'xy']
40:def _calibrated_observations(study):
49:def _box_simplex(H, c, cap, budget):
95:def solve_batch(study: dict) -> dict:
<class 'list'> 6
{'completed_campaign_index': 0, 'completed_campaign_short_name': 'facility_init', 'facility_telemetry_trace': (24, 8)}
{'completed_campaign_index': 1, 'completed_campaign_short_name': 'beta_m99_c03_s00', 'facility_telemetry_trace': (24, 8)}
{'completed_campaign_index': 2, 'completed_campaign_short_name': 'beta_m99_c03_batch_01', 'facility_telemetry_trace': (24, 8)}
{'completed_campaign_index': 3, 'completed_campaign_short_name': 'beta_m99_c03_batch_02', 'facility_telemetry_trace': (24, 8)}
{'completed_campaign_index': 4, 'completed_campaign_short_name': 'beta_m99_c03_batch_03', 'facility_telemetry_trace': (24, 8)}
{'completed_campaign_index': 5, 'completed_campaign_short_name': 'beta_m99_c03_batch_04', 'facility_telemetry_trace': (24, 8)}
pulses 3
[('w_pulse', (8,)), ('y_pulse_raw', (96,))]
{'gains': [1.2699989, 0.82999928, 0.78000042, 1.31001083], 'offsets': [0.05999695, -0.0500034, -0.07000476, 0.08000568]}
```

### Turn 3 — `execute_in_remote_docker` (`14483 ms` LLM)
- **Rationale**: Recover the exact calibrated observation key and archived wall stencils before comparing pulse predictions; summarize the log rather than dumping the traces.

```bash
cd /holdout_task; tail -22 _verified_upstream.py; cat /tmp/hegel_continual_archive/beta_m99_c03_batch_02_solver.py; sed -n '1,42p' /tmp/hegel_continual_archive/beta_m99_c03_batch_01_solver.py; python3 - <<'PY'
import json,pickle,numpy as np
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 x=np.array(e['facility_telemetry_trace']);print(e['completed_campaign_short_name'], 'mean',x.mean(0).round(4),'std',x.std(0).round(4))
i=pickle.load(open('public_batch.pkl','rb'))[0]; s=i['study']; print('aux',s['aux_manifold_telemetry'][:4]); print('pulses',[(p['w_pulse'], np.min(p['y_pulse_raw']),np.max(p['y_pulse_raw'])) for p in i['campaign_diagnostics']]); print('banks',s['row_bank_id'][:24]);print('sweep bank type',type(s['sensor_calibration_sweeps']['single_bank_sweeps']))
PY
```
- **Container Exit Code**: `0` (`408 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
'nx'], study['ny']
    ax, ay = [_axis_operator(study['kappa_' + a], study['vel_' + a],
                             study['L' + a]) for a in 'xy']
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    i, j = study['sensor_ij'].T
    sample = i + nx * j
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        blocks.append(fields[sample])
    return np.vstack(blocks), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _calibrated_observations(study):
    sweep = study['sensor_calibration_sweeps']
facility_init mean [-0.3592  0.0044 -0.0078  0.0081  0.0064 -0.3899 -0.0092 -0.0077] std [0.0787 0.6097 0.6019 0.604  0.5932 0.0742 0.6147 0.6056]
beta_m99_c03_s00 mean [ 2.4211e+00 -5.2000e-03 -2.0000e-04  7.2000e-03  3.1000e-03 -3.9510e-01
  3.4000e-03  2.9000e-03] std [0.0541 0.6064 0.5984 0.5998 0.5966 0.0723 0.6029 0.6081]
beta_m99_c03_batch_01 mean [-3.5000e-01  1.1000e-03  4.0000e-03 -1.1200e-02 -1.0400e-02  2.5607e+00
 -1.5000e-03  1.3100e-02] std [0.0655 0.6039 0.6003 0.6031 0.6    0.062  0.5993 0.6146]
beta_m99_c03_batch_02 mean [-0.3659  0.0078 -0.0037 -0.0049 -0.0116 -0.3913 -0.0055  0.0054] std [0.0679 0.6031 0.6009 0.6016 0.6003 0.0747 0.6007 0.5988]
beta_m99_c03_batch_03 mean [ 2.4434e+00 -2.2000e-03  8.4000e-03  6.2000e-03  5.5000e-03 -4.0680e-01
  1.2100e-02 -3.0000e-04] std [0.0679 0.6034 0.594  0.6083 0.6025 0.0623 0.6    0.6112]
beta_m99_c03_batch_04 mean [-3.5000e-01  1.1000e-03  4.0000e-03 -1.1200e-02 -1.0400e-02  2.5607e+00
 -1.5000e-03  1.3100e-02] std [0.0655 0.6039 0.6003 0.6031 0.6    0.062  0.5993 0.6146]
aux [[-0.31520507 -0.72500208 -0.10174833  0.23966342]
 [ 0.02856165  0.29844167 -0.01745721  0.41013462]
 [-1.29964573 -1.37432794 -0.96509407 -1.24391238]
 [ 0.18033435  0.43530967  1.52261487 -0.08554798]]
pulses [(array([1. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ]), -0.01811847847165435, 0.6437238002406888), (array([0. , 1. , 0. , 0. , 0.5, 0. , 0. , 0. ]), -0.0206532278691914, 0.6594418503429198), (array([0. , 0. , 1. , 0. , 0. , 0.5, 0. , 0. ]), -0.029023950213852998, 0.7654611376056216)]
banks [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
sweep bank type <class 'dict'>
```

### Turn 4 — `execute_in_remote_docker` (`28747 ms` LLM)
- **Rationale**: Compare open, reflecting, periodic, and mixed endpoint stencils against the calibrated pulses. Report both raw and nuisance-projected residuals so nuisance alignment cannot mask a wrong wall operator.
- **Alternatives Weighed**: `Use the latest logged wall regime without comparing pulses` (rejected: The log describes the preceding campaign; calibrated pulses directly test the current operator.)

```bash
cat > /tmp/chamber_pulses.py <<'PY'
import sys
sys.path.insert(0,'/holdout_task')
import pickle
from itertools import product
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from _verified_upstream import calibrate_upstream

def axis(s,a,ends):
    k,v=s['kappa_'+a],s['vel_'+a]; n=len(k); h=s['L'+a]/(n-1)
    A=np.zeros((n,n)); ii=np.arange(1,n-1)
    def up(i):
        l,r=(i-1)%n,(i+1)%n
        m=(k[i]+k[l])/(2*h*h)+np.maximum(v[i],0)/h
        p=(k[i]+k[r])/(2*h*h)+np.maximum(-v[i],0)/h
        A[i,l]=-m; A[i,r]=-p; A[i,i]=m+p
    up(ii)
    for side,bc in enumerate(ends):
        i=(n-1)*side; j=1 if side==0 else n-2
        if bc=='P': up(i)
        else:
            A[i,j]=-k[i]/h**2; A[i,i]=-A[i,j]+(0 if bc=='N' else (s['robin_'+a] if bc=='R' else bc)/h)
    return csc_matrix(A)

def forward(s,bcs):
    nx,ny=s['nx'],s['ny']; ax,ay=[axis(s,a,b) for a,b in zip('xy',bcs)]
    L=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); I=eye(nx*ny,format='csc')
    rhs=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); i,j=s['sensor_ij'].T
    return np.vstack([splu(L+t*I).solve(rhs)[i+nx*j] for t in s['mu_shifts']])

if __name__=='__main__':
    with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
    s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
    W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Y=np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]); Z=s['Z_nuisance']
    q=np.linalg.qr(Z,mode='reduced')[0]
    scores=[]
    for b in product('RNP',repeat=4):
        r=(forward(s,(b[:2],b[2:]))@W-Y); proj=r-q@(q.T@r)
        rm=np.sqrt(np.mean(r.reshape(len(s['mu_shifts']),-1,W.shape[1])**2,axis=1)); scores.append((np.sqrt(np.mean(r*r)),''.join(b),np.sqrt(np.mean(proj*proj)),rm.max(),rm.round(7).tolist()))
    for row in sorted(scores)[:12]: print(row)
    print('uniform candidates',[(b,r,p,ws) for r,b,p,ws,_ in scores if len(set(b))==1])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/chamber_pulses.py
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --describe autogenous_sparse_ridge_certification.solve_and_certify_l0 | head -35
```
- **Container Exit Code**: `0` (`658 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
91718, 0.0428794], [0.0024055, 0.0062768, 0.0237342], [0.0019812, 0.0041008, 0.0152345]])
(0.019394616686564108, 'NNPP', 0.019215723414747633, 0.04424857991133938, [[0.0086086, 0.0172126, 0.0442486], [0.0046697, 0.0118819, 0.0242133], [0.0030574, 0.0077412, 0.0154433]])
(0.020354862874335675, 'PNPP', 0.020185136837135652, 0.04208482412265894, [[0.0076477, 0.0298114, 0.0420848], [0.0043303, 0.0135614, 0.0231196], [0.002754, 0.0068944, 0.0147733]])
(0.021730546621824463, 'NPPP', 0.0215060986247409, 0.045499533360683256, [[0.0144312, 0.0275985, 0.0454995], [0.0060763, 0.0144905, 0.0249494], [0.0034228, 0.0085863, 0.0159687]])
(0.02403122702896323, 'NPPN', 0.023741887373436424, 0.045656979656576335, [[0.0288077, 0.0225754, 0.045657], [0.0133017, 0.014045, 0.0294794], [0.006831, 0.0087076, 0.0201914]])
(0.026944708272583204, 'NNPN', 0.02650559753528742, 0.06220581560609947, [[0.0163585, 0.0182381, 0.0622058], [0.0103645, 0.0118312, 0.0350216], [0.0063844, 0.0075591, 0.0221935]])
(0.027901172524861455, 'PPPN', 0.027517511001407195, 0.0688889706472096, [[0.005337, 0.0149739, 0.068889], [0.0034992, 0.0097919, 0.0366703], [0.0028496, 0.0060968, 0.0225778]])
(0.028695289284736667, 'NPNP', 0.028319946745878857, 0.06885603541280054, [[0.0073214, 0.0308383, 0.068856], [0.0035585, 0.0150962, 0.0319484], [0.0027437, 0.0088373, 0.0178362]])
uniform candidates [('RRRR', 0.12027677159990062, 0.11917319761602413, 0.21085364595769285), ('NNNN', 0.009915819450639107, 0.009807403778559732, 0.017728576159514252), ('PPPP', 0.01762643607233804, 0.017459800978363534, 0.0428794291058855)]
{
  "ok": true,
  "target": "autogenous_sparse_ridge_certification.solve_and_certify_l0",
  "access_mode": "executable",
  "execution_available": true,
  "matches": [
    {
      "module_stem": "autogenous_sparse_ridge_certification",
      "granularity_tier": "L2_domain_operator",
      "compiled_code_sha256": "770f6644bb1fd8eb49dbacf51ffa3405a9ca876974e0021d66415d9014d8dfb9",
      "qualification_status": "public_episode",
      "unit_test_verified": true,
      "mathematical_preconditions": [
        {
          "field": "n_samples",
          "op": ">=",
          "value": 1
        },
        {
          "field": "n_features",
          "op": "<=",
          "value": 16
        },
        {
          "field": "lam0",
          "op": ">=",
          "value": 0
        },
        {
          "field": "lam2",
          "op": ">=",
          "value": 0
        },
        {
          "field": "M",
```

### Turn 5 — `execute_in_remote_docker` (`50920 ms` LLM)
- **Rationale**: Implement the pulse-identified operator and the specified ridge projection and KKT certificate. Keep the staged batch wrapper unchanged because it already calls the verified dynamic calibration before the downstream solver.
- **Alternatives Weighed**: `Reuse the archived wholly periodic or wholly reflecting operator` (rejected: Both miss the calibrated tracer readings by orders of magnitude.); `Use the boxed L0 ridge skill` (rejected: It solves a sparsity-penalized problem, not the equality-constrained convex problem specified here.)
- **ATMS NOGOOD DECLARED**: Refuted `The same wall stencil can be used on both axes.` | Witness: `The mixed PPNN stencil has pooled calibrated pulse RMSE 5.62e-7; the best uniform stencil has RMSE 9.92e-3. Nuisance projection preserves the separation.` | Blanket Diagnosis: `The mismatch is in the endpoint rows, not the interior transport or nuisance projection.` | Sublated Pivot: `Keep the interior upwind operator and use periodic x with reflecting y.`

```bash
cat > /holdout_task/downstream_solver.py <<'PY'
"""Transport with periodic x and reflecting y, followed by source inversion."""
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_operator(kappa, velocity, length, periodic):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n) if periodic else np.arange(1, n - 1)
    left, right = (i - 1) % n, (i + 1) % n
    minus = (kappa[i] + kappa[left]) / (2 * h**2) + np.maximum(velocity[i], 0) / h
    plus = (kappa[i] + kappa[right]) / (2 * h**2) + np.maximum(-velocity[i], 0) / h
    a = np.zeros((n, n))
    a[i, i], a[i, left], a[i, right] = minus + plus, -minus, -plus
    if not periodic:
        a[0, 0] = kappa[0] / h**2
        a[-1, -1] = kappa[-1] / h**2
        a[0, 1], a[-1, -2] = -a[0, 0], -a[-1, -1]
    return csc_matrix(a)


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis_operator(study['kappa_x'], study['vel_x'], study['Lx'], True)
    ay = _axis_operator(study['kappa_y'], study['vel_y'], study['Ly'], False)
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    i, j = study['sensor_ij'].T
    samples = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        samples.append(fields[i + nx * j])
    return np.vstack(samples), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(hessian, linear, cap, budget):
    n = len(linear)
    if budget == 0:
        return np.zeros(n), float(linear.min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float((hessian @ w + linear).max())

    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)  # -1 lower, +1 upper, 0 free
    for _ in range(100 * n):
        g = hessian @ w + linear
        free = np.flatnonzero(active == 0)
        if not len(free):
            lo, hi = np.flatnonzero(active < 0), np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[i] >= g[j] - 1e-12:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            continue

        # Minimize on the current face and stop at the first blocking bound.
        k = len(free)
        system = np.ones((k + 1, k + 1))
        system[:k, :k] = hessian[np.ix_(free, free)]
        system[-1, -1] = 0
        step_and_multiplier = np.linalg.solve(system, np.r_[-g[free], 0.0])
        step = step_and_multiplier[:-1]
        target = w[free] + step
        limits = np.ones(k)
        lower, upper = target < -1e-14, target > cap + 1e-14
        limits[lower] = -w[free[lower]] / step[lower]
        limits[upper] = (cap - w[free[upper]]) / step[upper]
        block = np.argmin(limits)
        alpha = limits[block]
        w[free] += alpha * step
        if alpha < 1:
            i = free[block]
            active[i] = -1 if lower[block] else 1
            w[i] = 0.0 if active[i] < 0 else cap
            continue

        nu = float(-step_and_multiplier[-1])
        reduced = hessian @ w + linear - nu
        violation = active * reduced
        i = np.argmax(violation)
        if violation[i] <= 1e-12:
            return np.clip(w, 0, cap), nu
        active[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_downstream(cal_item: dict) -> dict:
    study = cal_item
    response, last = _forward(study)
    z = study['Z_nuisance']
    y = study['y_obs_raw']  # Canonical coordinates from the upstream transducer.
    ridge, cap, budget = (study[key] for key in ('source_ridge', 'u_box', 'total_budget'))
    projection = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    s = response - z @ (projection @ response)
    r = y - z @ (projection @ y)
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ r, cap, budget)
    residual = s @ w - r
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = s.T @ residual + ridge * w
    reduced = g - nu
    dual = float(objective - g @ w + nu * budget + cap * np.minimum(0, reduced).sum())
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0, reduced).tolist(),
            'lam_upper': np.maximum(0, -reduced).tolist(),
            'primal_objective': objective, 'dual_lower_bound': dual,
        },
    }
PY
python3 -I /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py
```
- **Container Exit Code**: `1` (`321 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/downstream_solver.py']`
- **Stdout (tail)**:
```text
[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5e9a871cafbc6b77 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5e9a871c (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/holdout_task/validate_public_schema.py", line 6, in <module>
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/holdout_task/batch_solver.py", line 3, in <module>
    import _verified_upstream as _up
ModuleNotFoundError: No module named '_verified_upstream'
```

### Turn 6 — `execute_in_remote_docker` (`14807 ms` LLM)
- **Rationale**: Run the public schema check through both launch paths. Giving the isolated verifier the artifact directory explicitly distinguishes a launcher path issue from an actual wrapper or calibration defect.
- **Alternatives Weighed**: `Change the staged wrapper's imports immediately` (rejected: The observed failure occurred under Python isolation, which drops the script directory. First test the documented launcher and check that the wrapper matches the staged calibration signature.)

```bash
cd /holdout_task; cat batch_solver.py; sed -n '1,12p' validate_public_schema.py; python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py; python3 -I - <<'PY'
import runpy, sys
root = '/holdout_task'
sys.path.insert(0, root)
sys.argv = [root + '/validate_public_schema.py', root + '/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
```
- **Container Exit Code**: `0` (`549 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
from __future__ import annotations
import inspect, json
import _verified_upstream as _up
import downstream_solver as _dn

_PAY = json.load(open('/holdout_task/_upstream_param_payload.json', 'r'))

def solve_batch(item: dict) -> dict:
    fn_u = getattr(_up, 'calibrate_upstream')
    cal = fn_u(item, _PAY) if len(inspect.signature(fn_u).parameters) >= 2 else fn_u(item)
    return _dn.solve_downstream(cal)
import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]
st = item["study"]
out = mod.solve_batch(st)
m_m = st["source_modes"].shape[0]; nx = int(st["nx"]); ny = int(st["ny"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
{"schema_valid": true, "errors": []}
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5e9a871cafbc6b77 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5e9a871c (directory).
```

### Turn 7 — `execute_in_remote_docker` (`39869 ms` LLM)
- **Rationale**: The schema check does not test the wall stencil, terminal indexing, or KKT accuracy. Hold out each pulse to test the stencil choice, compare the terminal field with a dense solve, and test KKT accuracy at ordinary and saturated budgets. A failure would identify which component needs a correction.

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import sys, pickle
from itertools import product
import numpy as np
sys.path[:0] = ['/holdout_task', '/tmp']
from batch_solver import solve_batch
from downstream_solver import _box_simplex, _forward
from _verified_upstream import calibrate_upstream
from chamber_pulses import forward
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
worst_pulse = worst_gap = worst_comp = worst_feas = worst_stat = worst_field = 0.
for item in items:
    study = item['study']
    pulses = item.get('campaign_diagnostics')
    s = calibrate_upstream(dict(study, reference_tracer_pulses=pulses))
    out = solve_batch(study)
    S, U = _forward(s)
    reference_S = forward(s, ('PP', 'NN'))
    np.testing.assert_allclose(S, reference_S, atol=1e-12, rtol=0)
    if pulses:
        calibrated = s['reference_tracer_pulses']
        W = np.column_stack([p['w_pulse'] for p in calibrated])
        Y = np.column_stack([p['y_pulse_raw'] for p in calibrated])
        candidates = {''.join(b): forward(s, (b[:2], b[2:])) @ W - Y for b in product('RNP', repeat=4)}
        for j in range(W.shape[1]):
            train = np.arange(W.shape[1]) != j
            best = min(candidates, key=lambda b: np.mean(candidates[b][:, train] ** 2))
            assert best == 'PPNN', (j, best)
            rmse = np.sqrt(np.mean(candidates[best][:, j].reshape(len(s['mu_shifts']), -1)**2, axis=1))
            worst_pulse = max(worst_pulse, rmse.max())
    cert = out['kkt_certificate']; w = out['weights']; cap = s['u_box']; budget = s['total_budget']
    z = s['Z_nuisance']; y = s['y_obs_raw']; ridge = s['source_ridge']
    P = np.eye(len(y)) - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    T = P @ reference_S; r = T @ w - P @ y
    obj = .5 * (r @ r + ridge * (w @ w)); g = T.T @ r + ridge * w
    lo, hi = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper']); nu = cert['nu_sum']
    exact_dual = obj - g @ w + nu * budget + cap * np.minimum(0, g-nu).sum()
    np.testing.assert_allclose([cert['primal_objective'], cert['dual_lower_bound']], [obj, exact_dual], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(w, cert['weights'])
    assert min(lo.min(), hi.min()) >= 0 and obj - exact_dual >= -1e-12
    worst_feas = max(worst_feas, abs(w.sum()-budget), -w.min(), w.max()-cap)
    worst_stat = max(worst_stat, np.max(abs(g-nu-lo+hi)))
    worst_comp = max(worst_comp, max(abs(lo*w)), max(abs(hi*(cap-w))))
    worst_gap = max(worst_gap, abs(obj-exact_dual))
    from chamber_pulses import axis
    nx, ny = s['nx'], s['ny']
    L = np.kron(np.eye(ny), axis(s, 'x', 'PP').toarray()) + np.kron(axis(s, 'y', 'NN').toarray(), np.eye(nx)) + np.diag(s['V_reaction_2d'].ravel(order='F'))
    rhs = np.einsum('k,kij->ij', w, s['source_modes']).ravel(order='F')
    field = np.linalg.solve(L + s['mu_shifts'][-1] * np.eye(nx*ny), rhs).reshape((nx,ny), order='F')
    worst_field = max(worst_field, np.max(abs(field - out['terminal_field'])))
assert worst_pulse < 1e-4 and max(worst_feas,worst_stat,worst_comp,worst_gap,worst_field) < 1e-9
rng = np.random.default_rng(558); worst_stress = 0.
for case in range(120):
    n = 8; A = rng.normal(size=(n,n)); H = A.T @ A + .04*np.eye(n); c = rng.normal(size=n)*10; cap=.42
    for b in [0., cap*1e-5, 2*cap, 1.4, cap*(n-1e-5), n*cap]:
        w,nu = _box_simplex(H,c,cap,b); reduced = H@w+c-nu
        err = max(abs(w.sum()-b), -w.min(), w.max()-cap, max(w*np.maximum(0,reduced)), max((cap-w)*np.maximum(0,-reduced)))
        worst_stress = max(worst_stress, err)
assert worst_stress < 1e-9
print(f'{len(items)} public items; held-pulse worst shift RMSE={worst_pulse:.3g}; terminal max error={worst_field:.3g}; feasibility={worst_feas:.3g}; stationarity={worst_stat:.3g}; complementarity={worst_comp:.3g}; gap={worst_gap:.3g}; 720 QP stress cases worst error={worst_stress:.3g}')
PY
```
- **Container Exit Code**: `0` (`812 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1 public items; held-pulse worst shift RMSE=7.03e-07; terminal max error=1.53e-16; feasibility=0; stationarity=1.77e-16; complementarity=1.51e-16; gap=7.72e-16; 720 QP stress cases worst error=2.22e-15

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5e9a871cafbc6b77 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5e9a871c (directory).
```

### Turn 8 — `submit_completed_task` (`11095 ms` LLM)
- **Rationale**: Every leave-one-pulse-out fold selected the mixed boundary stencil. The verified bundle also passed schema validation, an independent dense terminal-field comparison, and KKT stress tests without artifact changes.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`10` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4695'}): Do NOT lock prematurely with `10` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Verified both solver files. The public schema passes, the worst held-out tracer RMSE was 7.03e-7, and the worst KKT error across 720 stress cases was 2.22e-15.`

### Turn 9 — `execute_in_remote_docker` (`31194 ms` LLM)
- **Rationale**: Check whether a sensor bank or the nuisance projection drives selection of the mixed wall stencil. A fold reversal or small margin would warrant reopening the stencil choice; a constant-mode failure would warrant correcting the transport operator.
- **Alternatives Weighed**: `Test classification, relay, or Fokker–Planck alternatives` (rejected: This task has no class labels or relay states and specifies a steady spatial transport operator. Sensor-bank holdout and constant-mode preservation test the relevant nuisance and algebraic failure modes.)

```bash
set -e
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import pickle, runpy, sys
from itertools import product
import numpy as np
root = '/holdout_task'
sys.path[:0] = [root, '/tmp']
from _verified_upstream import calibrate_upstream
from chamber_pulses import forward
from downstream_solver import _axis_operator
with open(root + '/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
worst, margin, conservation = 0., float('inf'), 0.
folds = 0
for item in items:
    s = calibrate_upstream(dict(item['study'], reference_tracer_pulses=item['campaign_diagnostics']))
    for a, periodic in [('x', True), ('y', False)]:
        A = _axis_operator(s['kappa_'+a], s['vel_'+a], s['L'+a], periodic)
        conservation = max(conservation, np.max(np.abs(A @ np.ones(A.shape[0]))))
    pulses = s['reference_tracer_pulses']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    Y = np.column_stack([p['y_pulse_raw'] for p in pulses])
    Z = s['Z_nuisance']
    banks = np.asarray(s['row_bank_id'])
    shifts = np.repeat(np.arange(len(s['mu_shifts'])), len(s['sensor_ij']))
    names = [''.join(b) for b in product('RNP', repeat=4)]
    errors = np.stack([forward(s, (b[:2], b[2:])) @ W - Y for b in names])
    for p in range(W.shape[1]):
        for bank in np.unique(banks):
            train_rows = banks != bank
            train_cols = np.arange(W.shape[1]) != p
            R = errors[:, train_rows, :][:, :, train_cols]
            z = Z[train_rows]
            profile = np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
            T = np.eye(len(z)) - z @ profile
            for residual in (R, T @ R):
                score = np.sqrt(np.mean(residual**2, axis=(1,2)))
                best, second = np.argsort(score)[:2]
                assert names[best] == 'PPNN', (p, bank, names[best])
                margin = min(margin, score[second] / score[best])
                for k in np.unique(shifts):
                    held = errors[best, (banks == bank) & (shifts == k), p]
                    assert held.size
                    worst = max(worst, np.sqrt(np.mean(held**2)))
                folds += 1
assert conservation < 1e-10 and worst < 1e-4 and margin > 100
print(f'{folds} pulse/bank folds (with and without nuisance profiling); worst held-bank/pulse/shift RMSE={worst:.3g}; minimum training runner-up ratio={margin:.3g}; transport constant-mode residual={conservation:.3g}')
sys.argv = [root + '/validate_public_schema.py', root + '/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
```
- **Container Exit Code**: `0` (`641 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
24 pulse/bank folds (with and without nuisance profiling); worst held-bank/pulse/shift RMSE=8.32e-07; minimum training runner-up ratio=8.67e+03; transport constant-mode residual=3.55e-15
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5e9a871cafbc6b77 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5e9a871c (directory).
```

### Turn 10 — `submit_completed_task` (`12246 ms` LLM)
- **Rationale**: All 24 pulse and sensor-bank folds selected the mixed stencil with and without nuisance profiling. The same verified bundle passed the isolated schema check, dense terminal-field comparison, and KKT stress tests.
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`8` turns remaining; artifact status={'/holdout_task/batch_solver.py': 'FILE:408', '/holdout_task/downstream_solver.py': 'FILE:4695'}): Do NOT lock prematurely with `8` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Both solver files are verified. The solver uses periodic boundaries in x and reflecting boundaries in y. The public schema passed, the worst held-out tracer RMSE was 8.32e-7, and the worst KKT error across 720 stress cases was 2.22e-15.`

### Turn 11 — `execute_in_remote_docker` (`33507 ms` LLM)
- **Rationale**: The discrete audit cannot distinguish the exact stencil from a nearby one with weak leakage or altered wrap coupling. Fit those terms on training pulses only; reopen the stencil only if a meaningful correction improves held-out predictions. The existing sensor-bank audit already tests nuisance sensitivity, and this task has no class labels or relay states.

```bash
set -e
cat > /tmp/audit_wall_coefficients.py <<'PY'
import pickle, runpy, sys

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

root = '/holdout_task'
sys.path.insert(0, root)
from _verified_upstream import calibrate_upstream
from downstream_solver import _axis_operator

worst_fixed = worst_fit = worst_delta = 0.
min_rank = 8
folds = 0
with open(root + '/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = calibrate_upstream(dict(item['study'], reference_tracer_pulses=item['campaign_diagnostics']))
    nx, ny = s['nx'], s['ny']
    pulses = s['reference_tracer_pulses']
    w = np.column_stack([p['w_pulse'] for p in pulses])
    y = np.column_stack([p['y_pulse_raw'] for p in pulses])
    rows = s['sensor_ij'] @ np.array([1, nx])
    rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ w
    axes = [_axis_operator(s['kappa_'+a], s['vel_'+a], s['L'+a], a == 'x').toarray() for a in 'xy']
    ident = eye(nx*ny, format='csc')
    reaction = diags(s['V_reaction_2d'].ravel(order='F'), format='csc')

    def residual(theta):
        perturbed = []
        for base, params, a in zip(axes, theta.reshape(2, 2, 2), 'xy'):
            op = base.copy()
            h = s['L'+a] / (len(base)-1)
            for (i, j), (wrap, leak) in zip([(0, -1), (-1, 0)], params):
                op[i, j] -= wrap / h**2
                op[i, i] += wrap / h**2 + leak / h
            perturbed.append(csc_matrix(op))
        ax, ay = perturbed
        L = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
        return np.vstack([splu(L + shift * ident).solve(rhs)[rows] for shift in s['mu_shifts']]) - y

    zero = np.zeros(8)
    base_error = residual(zero)
    for p in range(w.shape[1]):
        train = np.arange(w.shape[1]) != p
        fit = least_squares(lambda t: residual(t)[:, train].ravel(), zero, bounds=(-0.05, 0.05), gtol=1e-12)
        assert fit.success, fit.message
        min_rank = min(min_rank, np.linalg.matrix_rank(fit.jac))
        worst_delta = max(worst_delta, np.max(abs(fit.x)))
        for e, fitted in [(base_error, False), (residual(fit.x), True)]:
            rmse = np.sqrt(np.mean(e[:, p].reshape(len(s['mu_shifts']), -1)**2, axis=1)).max()
            if fitted:
                worst_fit = max(worst_fit, rmse)
            else:
                worst_fixed = max(worst_fixed, rmse)
        folds += 1
print(f'boundary coefficient audit: {folds} held-pulse folds; minimum Jacobian rank={min_rank}/8; max absolute fitted perturbation={worst_delta:.3g}; worst held-out shift RMSE fixed={worst_fixed:.3g}, fitted={worst_fit:.3g}')
sys.argv = [root + '/validate_public_schema.py', root + '/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_wall_coefficients.py
```
- **Container Exit Code**: `0` (`673 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
boundary coefficient audit: 3 held-pulse folds; minimum Jacobian rank=8/8; max absolute fitted perturbation=2.82e-05; worst held-out shift RMSE fixed=7.03e-07, fitted=6.1e-07
{"schema_valid": true, "errors": []}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5e9a871cafbc6b77 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5e9a871c (directory).
```

### Turn 12 — `submit_completed_task` (`12752 ms` LLM)
- **Rationale**: All 24 pulse and sensor-bank folds selected the mixed stencil, with or without nuisance profiling. The unchanged bundle passed isolated schema validation, an independent dense terminal-field comparison, and KKT stress tests.
- **Alternatives Weighed**: `Fit additional endpoint leakage and wrap coupling` (rejected: The fitted perturbations were at most 2.82e-5 and reduced worst held-pulse shift RMSE by only 9.3e-8, with no demonstrated benefit for the required outputs.)
- **Submitted Final Answer / Summary**: `Verified both solver files. The solver uses periodic boundaries in x and reflecting boundaries in y. The public schema passed, the worst held-out tracer RMSE was 8.32e-7, and the worst KKT error across 720 stress cases was 2.22e-15.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
