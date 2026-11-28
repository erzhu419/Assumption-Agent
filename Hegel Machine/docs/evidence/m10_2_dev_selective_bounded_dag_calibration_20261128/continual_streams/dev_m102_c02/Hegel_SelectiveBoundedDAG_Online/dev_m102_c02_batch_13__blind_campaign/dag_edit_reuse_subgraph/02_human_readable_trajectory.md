# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `dev_m102_c02_batch_13__blind_campaign__reuse_subgraph`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_2_selective_bounded_dag_discovery_h16` / `m102_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `7 / 12` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `160,166` (`156,088` / `88,704` / `4,078`) |
| **Total Episode Wall Time (s)** | `142.64s` |

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

### Turn 1 — `execute_in_remote_docker` (`16147 ms` LLM)
- **Rationale**: Read the frozen consumer, scoring harness, public data shapes, and archive inventory before fitting or editing the sole target node.
- **Alternatives Weighed**: `Copy an archived calibrator outright` (rejected: The reported panel statistics point to different archived regimes; inspect the contracts and data before choosing a form.)

```bash
cd /holdout_task && cat _dag_spec.json && cat _frozen_nodes/node_01.py && cat test_dag_candidate.py && cat public_asset_index.json; echo '--- archive ---'; du -ah /tmp/hegel_continual_archive | sort -h | tail -60; echo '--- public payload ---'; python3 - <<'PY'
import pickle
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
def inspect(x,d=0):
 if hasattr(x,'shape'): print(type(x).__name__,x.shape,x.dtype,str(x.reshape(-1)[:8]));return
 if isinstance(x,dict):
  print('keys:',list(x))
  for k,v in x.items(): print(' '*d,k,end=': ');inspect(v,d+1)
 elif isinstance(x,(list,tuple)): print(type(x).__name__,len(x));inspect(x[0],d+1) if x else None
 else: print(type(x).__name__,str(x)[:160])
inspect(p)
PY
```
- **Container Exit Code**: `0` (`416 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
},
    "/holdout_task/test_dag_candidate.py": {
      "bytes": 8018
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2825
    }
  },
  "schema": "hegel_public_asset_index_v1"
}--- archive ---
4.0K	/tmp/hegel_continual_archive/archive_manifest.json
4.0K	/tmp/hegel_continual_archive/dev_m102_c02_batch_00_solver.py
4.0K	/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py
4.0K	/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
20K	/tmp/hegel_continual_archive
--- public payload ---
list 1
keys: ['study']
  study: keys: ['eval_times', 'panel_a', 'panel_b', 'panel_c', 'train', 'test', 'telemetry_vec']
   eval_times: ndarray (8,) float64 [0.25       0.64285714 1.03571429 1.42857143 1.82142857 2.21428571
 2.60714286 3.        ]
   panel_a: keys: ['u', 'v']
    u: ndarray (48, 4) float64 [-0.96884946  0.1361999  -0.51466993  0.32769074  1.23215695  0.77158548
 -0.16033005 -0.35966109]
    v: ndarray (48, 4) float64 [-1.07991001  0.3521788  -0.82032396  0.48971513  1.41734141  0.86371285
 -0.1893915  -0.94293145]
   panel_b: keys: ['u', 'v']
    u: ndarray (24, 4) float64 [-1.19700371 -0.42070824  0.56411732  0.88312795 -0.38277514 -0.92110618
 -0.52694723  0.93490248]
    v: ndarray (24, 4) float64 [ 0.0730301  -0.81702803 -0.56768778  0.33217904  0.61090157 -0.06477124
  0.23244554 -0.87841852]
   panel_c: keys: ['u', 'v1', 'v2']
    u: ndarray (24, 4) float64 [-0.32234982 -0.80332075  0.99084971 -0.0379777  -0.19717566 -1.0703229
 -0.95920922  0.96916947]
    v1: ndarray (24, 4) float64 [ 0.00999202  0.89450064 -0.79436297 -0.32234653  1.24724789 -0.62657683
 -0.69692709 -0.05825742]
    v2: ndarray (24, 4) float64 [ 0.54784972 -0.142308   -0.48430819 -0.11581889 -0.59851296  0.06408367
 -0.57129189 -0.30790972]
   train: keys: ['x', 'y', 'traj', 'summary']
    x: ndarray (72, 4) float64 [-0.17866927 -0.94861909 -0.08477855  0.35139842 -0.65356767  0.46033
  0.82161182  0.23984129]
    y: ndarray (72,) float64 [ 1.35843039 -0.98796855  0.88518207 -1.97335696 -1.13137497 -0.71452249
  0.90088549  1.0358094 ]
    traj: ndarray (72, 8) float64 [0.32389295 0.66281208 0.9242194  1.13541699 1.31927988 1.48299565
 1.62521737 1.7646412 ]
    summary: ndarray (4,) float64 [ 0.03578862  1.20276014 -0.00625586  1.1170146 ]
   test: keys: ['x']
    x: ndarray (36, 4) float64 [-0.02775249 -0.42254545 -0.86161989 -0.27756937  0.54739271  1.34083406
  0.53152185 -0.77354313]
   telemetry_vec: ndarray (4,) float64 [0.22 1.25 1.25 2.15]
```

### Turn 2 — `execute_in_remote_docker` (`8495 ms` LLM)
- **Rationale**: Recover the exact downstream contract, archived transforms, and fold construction; the first inspection truncated these details.

```bash
cd /holdout_task; tail -30 _frozen_nodes/node_01.py; cat test_dag_candidate.py; for f in /tmp/hegel_continual_archive/*; do echo ==== $f; cat "$f"; done
```
- **Container Exit Code**: `0` (`306 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
,b,y,n,2.,.001)+.25*_kernel(f,b,y,n,1.,.03)).ravel()
    sf=_lin(np.c_[z,b,_cross(z,b)])
    state=_ridge(sf[:n],np.asarray(tr['traj'],float),sf[n:],.08)
    # The terminal state carries much of the scalar readout's interaction.
    # Fit the difference on measured training states, then add it to the
    # predicted terminal state. Keep the independent readout as a hedge.
    diff = y[:,0] - np.asarray(tr['traj'], float)[:,-1]
    z5 = z**5
    gf = _lin(np.c_[b, z5, _cross(b,z5)])
    u = np.asarray(study['panel_c']['u'],float)
    poly = lambda x: np.c_[_lin(x),x*x,x**3]
    d = poly(z) @ _ls(poly(u),np.asarray(study['panel_c']['v2'],float))
    df = _lin(np.c_[b,d])
    correction = (.65*_ridge(gf[:n],diff,gf[n:],.12)
                  +.35*_ridge(df[:n],diff,df[n:],.4))
    y_hat = .40*y_hat + .60*(state[:,-1]+correction)
    return {'y_hat':y_hat,'state_pred':state,'diagnostics_vec':np.array([y_hat.mean(),y_hat.std(),state[:,-1].mean(),state[:,-1].std()])}
==== /tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json
{
  "accepted": false,
  "schema_valid": true,
  "execution_status": "completed",
  "failure_kind": "scientific_or_numerical",
  "errors": [
    "public_cv_y_mae_exceeded:0.604387",
    "public_cv_traj_mae_exceeded:0.393279"
  ],
  "public_input_sha256": "df36f78b6e615b036d65eff658bddeef754d81475497c8808c4480f9422369b9",
  "candidate_raw_sha256": "3a89b772182f513ae03bd8287d687b3dfbf35221f0c868cbded1a61b6ed8450c",
  "setup_source_sha256": "298106ec07c3942a8de45718e297bc8051141b92e7cd1ae43836a9ffff8e051e",
  "cv_metrics": {
    "public_cv_y_mae": 0.604387,
    "public_cv_traj_mae": 0.393279
  },
  "dag_runtime_trace": {
    "active_node_count": 2,
    "all_declared_edges_consumed": true,
    "all_exec_count_eq_1": true,
    "dag_id": "dag_13",
    "depth": 2,
    "edge_consumption_checks": {
      "node_00": [
        "__input__"
      ],
      "node_01": [
        "__input__",
        "node_00"
      ]
    },
    "exec_counts": {
      "node_00": 1,
      "node_01": 1
    },
    "max_fanout": 1,
    "node_code_shas": {
      "node_00": "06015f8a4b5f2e67022d0aece12f93a5c25dadc1332c81cf005970345dd99a7e",
      "node_01": "3897aadc3485493706942a28d95b3581750c475bc9de1fbb4d52cb16227c584c"
    },
    "non_identity_checks": {
      "node_00": true
    },
    "shared_fanout_nodes": [],
    "topo_order": [
      "node_00",
      "node_01"
    ]
  },
  "kernel_version": "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
  "wall_ms": 2301.27
}
```

### Turn 3 — `execute_in_remote_docker` (`14216 ms` LLM)
- **Rationale**: Retrieve the archived calibration formulas and fold definitions that were truncated in the preceding inspection.

```bash
cd /holdout_task; for f in /tmp/hegel_continual_archive/*solver.py; do echo ====; sed -n '1,54p' "$f"; done; echo ====; grep -nE 'fold|split|def |train|test' test_dag_candidate.py | tail -42; ls /tmp/hegel_wormhole_skills 2>/dev/null
```
- **Container Exit Code**: `0` (`296 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3][keep],v[keep,j]) for j in range(4)])
    def forward(z):
        w = np.roll(z,-1)
        return coeff[:,0]+coeff[:,1]*z+coeff[:,2]*w+coeff[:,3]*z*w+coeff[:,4]*z**3
    return np.array([root(lambda z:forward(z)-x,x).x for x in points])


def _op(p,k,z,mode):
    u,v = np.asarray(p['u'],float),np.asarray(p[k],float)
    def design(x,a):
        if mode=='b':
            return np.c_[_lin(x/(1+a*x*x)),x]
        return np.c_[_lin(np.arctan(a*x)),x,x**3]
    def error(a):
        d=design(u,a)
        h=np.sum(np.linalg.qr(d,mode='reduced')[0]**2,axis=1)
        return np.mean(np.abs((v-d@_ls(d,v))/(1-h[:,None])))
    a=minimize_scalar(error,bounds=(.15,4),method='bounded').x
    return design(z,a)@_ls(design(u,a),v)


def _cross(a,b):
    return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)


def _ridge(a,y,q,r):
    scale=np.maximum(a.std(0),1e-4)
    scale[0]=1
    p=np.eye(a.shape[1]);p[0,0]=0
    return (q/scale)@_ls(np.r_[a/scale,r*p],np.r_[y,np.zeros((a.shape[1],)+y.shape[1:])])


def _kernel(f, b, y, n, ell, reg):
    b=b/np.maximum(b[:n].std(0),1e-8)
====
9:def _canon_digest(obj) -> str:
11:    def _feed(v):
37:def _collect_subtree_digests(obj) -> set[str]:
45:def _collect_numeric_leaf_digests(obj) -> set[str]:
58:    def __init__(self, data: dict, src_key: str, tracker: set):
62:    def __getitem__(self, k):
65:    def get(self, k, default=None):
68:    def __iter__(self):
71:    def __len__(self): return len(self._data)
72:    def keys(self):
75:    def items(self):
78:    def values(self):
81:    def __contains__(self, k):
99:def run_dag(study: dict):
135:n_te, k_t = len(st["test"]["x"]), len(st["eval_times"])
152:    sp_idx = max(40, len(st["train"]["x"]) - 20)
153:    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
154:    st_cv["train"] = {
155:        "x": st["train"]["x"][:sp_idx].copy(),
156:        "y": st["train"]["y"][:sp_idx].copy(),
157:        "traj": st["train"]["traj"][:sp_idx].copy(),
158:        "summary": np.array([float(np.mean(st["train"]["y"][:sp_idx])), float(np.std(st["train"]["y"][:sp_idx])), float(np.mean(st["train"]["traj"][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
160:    st_cv["test"] = {"x": vx}
__init__.py
__main__.py
_skill_contracts.json
autogenous_conservative_elliptic_multigrid.py
autogenous_robust_network_assimilation.py
autogenous_sparse_ridge_certification.py
```

### Turn 4 — `execute_in_remote_docker` (`25770 ms` LLM)
- **Rationale**: Test the archived coupled sensor basis and compare frequency-profiled operators by panel leave-one-out error and trajectory compatibility.
- **Alternatives Weighed**: `Fit downstream targets directly` (rejected: Independent panel measurements can identify the coordinate transform without using held-out targets.)

```bash
PYTHONPATH=/tmp python3 -m hegel_wormhole_skills --search 'reference panel calibration robust regression variable projection' || true
cat >/tmp/probe14.py <<'PY'
import numpy as np,pickle
from scipy.optimize import minimize_scalar,root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def mat(*x):return np.column_stack(x)
def aff(x):return mat(np.ones(len(x)),x)
def a_design(u,j):
 x=u[:,j];w=u[:,(j+1)%4];return mat(np.ones(len(x)),x,w,x*w,x**3)
u=s['panel_a']['u'];v=s['panel_a']['v'];co=[]
for j in range(4):
 d=a_design(u,j);k=np.arange(len(u))
 for _ in range(4):
  b=ls(d[k],v[k,j]);e=abs(d@b-v[:,j]);k=np.argsort(e)[:-3]
 print('A',j,b,np.sort(e)[[0,20,38,43,44,45,46,47]],np.argsort(e)[-4:]);co.append(b)
co=np.array(co)
def zfun(points):
 def f(z):
  w=np.roll(z,-1);return np.sum(co*np.array([np.ones(4),z,w,z*w,z**3]).T,1)
 return np.array([root(lambda z:f(z)-x,x).x for x in points])
z=zfun(s['train']['x']); tr=s['train']['traj']; y=s['train']['y']; print('traj sv',np.linalg.svd(tr,compute_uv=False));print('raw z trajectory mae',np.mean(abs(aff(z)@ls(aff(z),tr)-tr)))
for panel in ['panel_b','panel_c']:
 u=s[panel]['u'];print(panel,'range',u.min(),u.max()); w=np.roll(u,-1,axis=1)
 for key,v in s[panel].items():
  if key=='u':continue
  for base in [lambda x:np.ones((len(x),1)),aff,lambda x:mat(aff(x),x*np.roll(x,-1,axis=1)),lambda x:mat(aff(x),x*x),lambda x:mat(aff(x),x**3)]:
   for name,g in [('cos',lambda x,a:np.cos(a*x)),('sin',lambda x,a:np.sin(a*x)),('rational',lambda x,a:x/(1+a*x*x)),('atan',lambda x,a:np.arctan(a*x))]:
    def d(x,a):return mat(base(x),g(x,a))
    def obj(a):
     x=d(u,a);h=np.sum(np.linalg.qr(x,mode='reduced')[0]**2,axis=1);return np.mean(abs((x@ls(x,v)-v)/(1-h[:,None])))
    aa=minimize_scalar(obj,bounds=(.15,4),method='bounded');c=d(z,aa.x)@ls(d(u,aa.x),v);print(key,base(u).shape[1],name,round(aa.x,4),round(aa.fun,6),'traj',round(np.mean(abs(aff(c)@ls(aff(c),tr)-tr)),6))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/probe14.py
```
- **Container Exit Code**: `0` (`615 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
.03805004
  0.03620771  0.03384468]
raw z trajectory mae 0.6072957084446657
panel_b range -1.2027979481274162 1.243991625686757
v 1 cos 2.6001 9.1e-05 traj 0.003942
v 1 sin 4.0 0.610492 traj 0.605389
v 1 rational 4.0 0.634116 traj 0.606189
v 1 atan 0.15 0.638773 traj 0.607295
v 5 cos 2.6001 0.000102 traj 0.003942
v 5 sin 4.0 0.724826 traj 0.603563
v 5 rational 4.0 0.731187 traj 0.599752
v 5 atan 4.0 0.732917 traj 0.59799
v 9 cos 2.6002 0.000138 traj 0.003943
v 9 sin 4.0 0.937568 traj 0.617633
v 9 rational 4.0 0.92127 traj 0.608885
v 9 atan 4.0 0.925594 traj 0.607835
v 9 cos 2.5989 0.000125 traj 0.003947
v 9 sin 4.0 0.258107 traj 0.192188
v 9 rational 4.0 0.244373 traj 0.199539
v 9 atan 4.0 0.244628 traj 0.199553
v 9 cos 2.5999 0.000122 traj 0.003941
v 9 sin 4.0 0.905708 traj 0.60107
v 9 rational 4.0 0.853524 traj 0.596621
v 9 atan 4.0 0.850447 traj 0.593471
panel_c range -1.2058782633892333 1.241130832344953
v1 1 cos 2.9583 0.597121 traj 0.052334
v1 1 sin 1.5028 0.033459 traj 0.607541
v1 1 rational 0.6673 0.019105 traj 0.607679
v1 1 atan 2.1032 0.00458 traj 0.607745
v1 5 cos 2.9406 0.133209 traj 0.601504
v1 5 sin 3.4261 0.011747 traj 0.607599
v1 5 rational 1.7829 0.005282 traj 0.607653
v1 5 atan 2.313 0.003363 traj 0.607688
v1 9 cos 0.3141 0.169161 traj 0.605546
v1 9 sin 3.3884 0.015143 traj 0.607575
v1 9 rational 1.7911 0.006592 traj 0.607636
v1 9 atan 2.3105 0.004142 traj 0.60768
v1 9 cos 4.0 0.13922 traj 0.60064
v1 9 sin 3.3582 0.015223 traj 0.606873
v1 9 rational 1.7158 0.007074 traj 0.607393
v1 9 atan 2.3102 0.004424 traj 0.607621
v1 9 cos 0.15 0.050462 traj 0.601922
v1 9 sin 4.0 0.012071 traj 0.607574
v1 9 rational 3.3883 0.002818 traj 0.60772
v1 9 atan 2.9753 0.002031 traj 0.607739
v2 1 cos 2.4979 0.308728 traj 0.013136
v2 1 sin 0.15 0.095372 traj 0.607295
v2 1 rational 0.15 0.10027 traj 0.607302
v2 1 atan 0.15 0.095453 traj 0.607295
v2 5 cos 1.2637 0.000392 traj 0.576585
v2 5 sin 0.15 0.10448 traj 0.606811
v2 5 rational 0.15 0.104652 traj 0.606839
v2 5 atan 0.15 0.104489 traj 0.606815
v2 9 cos 1.2815 0.00038 traj 0.576693
v2 9 sin 0.15 0.127241 traj 0.609593
v2 9 rational 4.0 0.129761 traj 0.60726
v2 9 atan 0.15 0.127335 traj 0.609586
v2 9 cos 2.434 0.000116 traj 0.576547
v2 9 sin 2.5738 0.00616 traj 0.578685
v2 9 rational 0.4937 0.00617 traj 0.578667
v2 9 atan 0.9666 0.006173 traj 0.578655
v2 9 cos 1.2679 0.000492 traj 0.57665
v2 9 sin 2.672 0.117425 traj 0.604952
v2 9 rational 0.2104 0.117422 traj 0.604979
v2 9 atan 0.5561 0.117422 traj 0.604984
```

### Turn 5 — `execute_in_remote_docker` (`25339 ms` LLM)
- **Rationale**: Implement the coordinate maps using only independent calibration panels: trim the three gross sensor outliers, invert the coupled map, and estimate the shared cosine frequency. Then check both frozen readouts on held-out target rows.
- **Alternatives Weighed**: `Retain linear or higher-order terms in panel B` (rejected: The simpler cosine basis has lower leave-one-out error and already explains the trajectory to its observed noise floor.)

```bash
cat > /holdout_task/node_00.py <<'PY'
"""Invert the coupled sensor map and evaluate the cosine calibration panel."""
import numpy as np
from scipy.optimize import minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _sensor_basis(x):
    w = np.roll(x, -1, axis=-1)
    return np.stack((np.ones_like(x), x, w, x*w, x**3), axis=-1)


def _sensor_coordinates(panel, points):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    basis = _sensor_basis(u)
    rows = np.arange(len(u))
    for _ in range(5):
        coef = np.array([_ls(basis[rows, j], v[rows, j]) for j in range(u.shape[1])])
        err = np.linalg.norm(np.sum(basis * coef, axis=-1) - v, axis=1)
        rows = np.argsort(err)[:len(u)-3]
    coef = np.array([_ls(basis[rows, j], v[rows, j]) for j in range(u.shape[1])])

    def forward(z):
        return np.sum(_sensor_basis(z) * coef, axis=-1)

    calibrated = []
    for x in points:
        guess = (x - coef[:, 0]) / coef[:, 1]
        fit = root(lambda z: forward(z) - x, guess, tol=1e-10)
        calibrated.append(fit.x)
    return np.asarray(calibrated)


def _cosine_coordinates(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(x, frequency):
        return np.column_stack((np.ones(len(x)), np.cos(frequency*x)))

    def loss(frequency):
        a = design(u, frequency)
        residual = a @ _ls(a, v) - v
        return np.sum(residual**2)

    grid = np.linspace(0.15, 5.0, 61)
    i = int(np.argmin([loss(f) for f in grid]))
    frequency = minimize_scalar(loss, bounds=(grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]),
                                method='bounded', options={'xatol': 1e-10}).x
    return design(z, frequency) @ _ls(design(u, frequency), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.asarray(study['train']['x'], dtype=float)
    query = np.asarray(study['test']['x'], dtype=float)
    z = _sensor_coordinates(study['panel_a'], np.concatenate((x, query)))
    c = _cosine_coordinates(study['panel_b'], z)
    return {'c': c[:len(x)], 'query': c[len(x):]}
PY
OPENBLAS_NUM_THREADS=1 python3 /holdout_task/test_dag_candidate.py
```
- **Container Exit Code**: `0` (`599 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_00.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.004806, "public_cv_traj_mae": 0.008357}, "exec_counts": {"node_00": 1, "node_01": 1}}

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7cb07ebbd502199b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7cb07ebb (directory).
```

### Turn 6 — `execute_in_remote_docker` (`24937 ms` LLM)
- **Rationale**: The candidate passed the official gate with substantial margin. Verify the unchanged artifact in an isolated Python process and check blocked and extreme-sensor slices without refitting the model structure.

```bash
set -e
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/test_dag_candidate.py
python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cal = load('calibration', '/holdout_task/node_00.py')
readout = load('readout', '/holdout_task/_frozen_nodes/node_01.py')
study = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
train = study['train']
x, y, traj = (train[k] for k in ('x','y','traj'))
idx = np.arange(len(x))
folds = [(f'block_{i}', f) for i, f in enumerate(np.array_split(idx, 4))]
folds += [('largest_sensor_norm', np.argsort(np.linalg.norm(x,axis=1))[-18:])]
folds += [(f'sensor_{j}_{tail}', np.argsort(x[:,j])[sl]) for j in range(x.shape[1]) for tail, sl in [('low',slice(None,18)),('high',slice(-18,None))]]
worst = np.zeros(2)
for name, test in folds:
    keep = np.setdiff1d(idx,test)
    sub = dict(study, train={'x':x[keep], 'y':y[keep], 'traj':traj[keep]}, test={'x':x[test]})
    features = cal.run_node({'__input__':sub})
    assert features['c'].shape[0] == len(keep) and features['query'].shape[0] == len(test)
    pred = readout.run_node({'__input__':sub, 'node_00':features})
    metrics = np.array([np.mean(abs(pred['y_hat']-y[test])), np.mean(abs(pred['state_pred']-traj[test]))])
    assert np.isfinite(metrics).all()
    worst = np.maximum(worst, metrics)
    print(name, *np.round(metrics,6))
print('worst scalar / trajectory MAE', *np.round(worst,6))
assert (worst <= .038).all()
PY
```
- **Container Exit Code**: `0` (`1057 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.004806, "public_cv_traj_mae": 0.008357}, "exec_counts": {"node_00": 1, "node_01": 1}}
block_0 0.004521 0.007422
block_1 0.004302 0.008626
block_2 0.00369 0.007793
block_3 0.00487 0.008328
largest_sensor_norm 0.003639 0.006139
sensor_0_low 0.004887 0.006356
sensor_0_high 0.004092 0.00661
sensor_1_low 0.00419 0.008817
sensor_1_high 0.002863 0.007068
sensor_2_low 0.005348 0.006821
sensor_2_high 0.005084 0.007186
sensor_3_low 0.005206 0.005844
sensor_3_high 0.00412 0.00557
worst scalar / trajectory MAE 0.005348 0.008817

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=7cb07ebbd502199b (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_7cb07ebb (directory).
```

### Turn 7 — `submit_completed_task` (`11645 ms` LLM)
- **Rationale**: The hash-bound artifact passed the official test in an isolated Python process, executed each node exactly once, and kept both errors below tolerance on every blocked and extreme-sensor slice.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_00.py` without changing the frozen readout. The official test passed: scalar MAE 0.004806 and trajectory MAE 0.008357, both below 0.038. Across 13 stress slices, the worst MAEs were 0.005348 and 0.008817, respectively.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel_M10_1_UnknownCut_DAG_Discovery",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
