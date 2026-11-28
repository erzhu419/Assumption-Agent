# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `conf_m101_b01_batch_02__blind_campaign__reuse_subgraph`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel_M10_1_UnknownCut_DAG_Discovery` |
| **Domain** | `blind_multi_stage_scientific_computation_graph` |
| **Wave / Wormhole Family** | `m10_1_bounded_unknown_cut_dag_discovery` / `m101_blind_dag` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `7 / 12` |
| **ATMS Nogoods Recorded** | `2` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `166,082` (`159,002` / `48,384` / `7,080`) |
| **Total Episode Wall Time (s)** | `223.3s` |

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

### Turn 1 — `execute_in_remote_docker` (`15397 ms` LLM)
- **Rationale**: Inspect the frozen interfaces, scoring procedure, and numerical ranges to catch input-contract mismatches and identify the smallest basis worth testing.

```bash
cd /holdout_task && cat _dag_spec.json _frozen_nodes/node_00.py _frozen_nodes/node_01.py test_dag_candidate.py validate_public_schema.py public_asset_index.json && python3 - <<'PY'
import pickle,numpy as np
with open('public_batch.pkl','rb') as f: d=pickle.load(f)
def s(x,k=''):
 if isinstance(x,dict):
  for a,b in x.items(): s(b,k+'/'+str(a))
 elif isinstance(x,np.ndarray):
  print(k,x.shape,x.dtype,x[:2]);
  if x.size and np.issubdtype(x.dtype,np.number): print('range',np.min(x,axis=0),np.max(x,axis=0)) if x.ndim<3 and (x.ndim<2 or x.shape[-1]<12) else None
 else: print(k,type(x),str(x)[:300])
s(d)
PY
```
- **Container Exit Code**: `0` (`363 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
4
                ],
                "type": "ndarray"
              }
            },
            "panel_c": {
              "u": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              },
              "v1": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              },
              "v2": {
                "dtype": "float64",
                "shape": [
                  24,
                  4
                ],
                "type": "ndarray"
              }
            },
            "telemetry_vec": {
              "dtype": "float64",
              "shape": [
                4
              ],
              "type": "ndarray"
            },
            "test": {
              "x": {
                "dtype": "float64",
                "shape": [
                  36,
                  4
                ],
                "type": "ndarray"
              }
            },
            "train": {
              "summary": {
                "dtype": "float64",
                "shape": [
                  4
                ],
                "type": "ndarray"
              },
              "traj": {
                "dtype": "float64",
                "shape": [
                  72,
                  8
                ],
                "type": "ndarray"
              },
              "x": {
                "dtype": "float64",
                "shape": [
                  72,
                  4
                ],
                "type": "ndarray"
              },
              "y": {
                "dtype": "float64",
                "shape": [
                  72
                ],
                "type": "ndarray"
              }
            }
          }
        },
        "len": 1,
        "type": "list"
      }
    },
    "/holdout_task/test_dag_candidate.py": {
      "bytes": 5206
    },
    "/holdout_task/validate_public_schema.py": {
      "bytes": 2573
    }
  },
  "schema": "hegel_public_asset_index_v1"
} <class 'list'> [{'study': {'eval_times': array([0.25      , 0.64285714, 1.03571429, 1.42857143, 1.82142857,
       2.21428571, 2.60714286, 3.        ]), 'panel_a': {'u': array([[ 0.80036867, -0.76044789, -1.25289052, -1.16071049],
       [ 1.13700265,  1.15375462,  0.78592456, -1.22828447],
       [ 0.85872062,  1
```

### Turn 2 — `execute_in_remote_docker` (`22756 ms` LLM)
- **Rationale**: Read the exact validation split, inspect the study arrays, and compare affine versus quadratic leave-one-out errors per output to determine which stages need additional terms.
- **Alternatives Weighed**: `Fit a full cubic panel-C basis` (rejected: There are more coefficients than panel rows; examine leave-one-out residuals from identifiable bases first.)

```bash
cd /holdout_task && sed -n '35,160p' test_dag_candidate.py && python3 - <<'PY'
import pickle,itertools,numpy as np
np.set_printoptions(precision=4,suppress=True,linewidth=130)
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; t=s['train']; u=p['u']; n,d=u.shape
for k in ['eval_times','telemetry_vec']: print(k,s[k])
print('summary',t['summary']); print('panel c [u,v1,v2] rows\n',np.c_[u,p['v1'],p['v2']][:8]);print('training [x,y,traj] rows\n',np.c_[t['x'],t['y'],t['traj']][:6]); print('range x/y/traj',[(np.min(t[k],axis=0),np.max(t[k],axis=0)) for k in ('x','y','traj')])
a=np.c_[np.ones(n),u]; q=np.column_stack([a]+[(u[:,i]*u[:,j]) for i,j in itertools.combinations_with_replacement(range(d),2)])
for k in ['v1','v2']:
 v=p[k]; print(k,'std',v.std(0));
 for f in [a,q]:
  fit=np.linalg.lstsq(f,v,rcond=None)[0]; e=v-f@fit; loo=e/(1-(f*np.linalg.pinv(f).T).sum(1))[:,None]; print(f.shape[1], 'train RMSE',np.mean(e**2,0)**.5,'LOO RMSE',np.mean(loo**2,0)**.5); print('weights\n',fit)
PY
```
- **Container Exit Code**: `0` (`325 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.3189 -1.3965 -1.4469 -1.4911]
 [-0.4571  0.4204 -1.3834  1.1102 -0.3097 -0.2284 -0.4947 -0.6765 -0.8292 -0.9271 -1.0148 -1.0846 -1.1381]
 [-0.7509 -1.5154  0.9999 -0.6628  1.1713 -0.9008 -1.6737 -2.0201 -2.1833 -2.2372 -2.2289 -2.2304 -2.2209]
 [-0.4233 -0.3322  1.1412  0.7625  0.089   0.1436  0.2569  0.3015  0.3136  0.3151  0.3101  0.3008  0.2891]]
range x/y/traj [(array([-0.8753, -1.5163, -1.4657, -0.9246]), array([1.0004, 1.1662, 1.163 , 1.1631])), (-1.7795049340959586, 1.291047630084273), (array([-0.9008, -1.6737, -2.0201, -2.1833, -2.2372, -2.2289, -2.2846, -2.3444]), array([0.2928, 0.5306, 0.664 , 0.8314, 0.9589, 1.0858, 1.1811, 1.2541]))]
v1 std [0.6643 0.5051 0.5767 0.525 ]
5 train RMSE [0.5584 0.4656 0.5165 0.4899] LOO RMSE [0.7862 0.6579 0.7029 0.6997]
weights
 [[-0.0332 -0.2102  0.1721  0.2684]
 [ 0.3866 -0.05   -0.3282 -0.0494]
 [-0.2997 -0.1296 -0.2748  0.2111]
 [ 0.238   0.289  -0.0257 -0.0246]
 [-0.104  -0.0086 -0.1596  0.2326]]
15 train RMSE [0.0001 0.     0.     0.0001] LOO RMSE [0.0002 0.0001 0.0002 0.0002]
weights
 [[-0.024  -0.0015  0.1066  0.134 ]
 [ 0.0001  0.0002 -0.0001  0.0001]
 [ 0.      0.     -0.     -0.    ]
 [ 0.      0.0001 -0.      0.0001]
 [-0.     -0.0001 -0.     -0.    ]
 [-0.0001 -0.0002  0.0002 -0.    ]
 [-1.1559  0.3081 -0.1567  0.1681]
 [-0.0001 -0.0001 -0.     -0.    ]
 [ 0.3649  0.9236 -0.1323  0.6926]
 [-0.0001 -0.0001  0.     -0.0001]
 [-0.0674 -0.6808  0.2139  0.9847]
 [ 0.0001 -0.0002  0.     -0.0001]
 [-0.0001 -0.0002  0.0001 -0.0001]
 [ 0.1002 -0.2677 -1.1813  0.0784]
 [-0.     -0.0001 -0.      0.    ]]
v2 std [0.3182 0.3442 0.3749 0.3682]
5 train RMSE [0.0791 0.0937 0.0592 0.074 ] LOO RMSE [0.1043 0.1227 0.0756 0.0958]
weights
 [[ 0.2405 -0.0889 -0.0719  0.1202]
 [-0.0458 -0.0516 -0.4714  0.3074]
 [ 0.0876 -0.4903  0.001  -0.0623]
 [ 0.1766  0.0561 -0.2595 -0.3453]
 [ 0.4312  0.0271  0.1408  0.1788]]
15 train RMSE [0.0017 0.0042 0.0027 0.004 ] LOO RMSE [0.0057 0.0175 0.01   0.0167]
weights
 [[ 0.0852  0.0153  0.0499  0.1282]
 [-0.018  -0.0194 -0.3894  0.2746]
 [ 0.0637 -0.4746  0.0057 -0.026 ]
 [ 0.1978  0.0408 -0.2597 -0.3513]
 [ 0.4374  0.0491  0.0997  0.1731]
 [-0.0094 -0.0195 -0.1813  0.1224]
 [ 0.0081 -0.0063  0.0017  0.004 ]
 [-0.0094  0.0028 -0.0057  0.0017]
 [-0.0047 -0.0057 -0.0027 -0.0006]
 [ 0.0306 -0.2131 -0.0017 -0.0047]
 [ 0.0021 -0.004   0.0079  0.0034]
 [ 0.0054  0.0001  0.0006 -0.0037]
 [ 0.0887  0.0136 -0.1219 -0.1608]
 [-0.0058 -0.003   0.0012  0.0046]
 [ 0.2005  0.0227  0.0445  0.0792]]
```

### Turn 3 — `execute_in_remote_docker` (`37695 ms` LLM)
- **Rationale**: Compare held-out panel residuals across additive transforms to distinguish v2 curvature from interpolation, then test whether the recovered coordinates explain each training target before choosing the sink model.
- **Alternatives Weighed**: `Retain all quadratic Panel C v1 columns` (rejected: The observed square and linear coefficients are at the noise scale; the sparse basis leaves more panel rows to constrain the fit.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine basis is sufficient for Panel C v1.` | Witness: `Affine leave-one-out RMSE ranges from 0.658 to 0.786; quadratic leave-one-out RMSE is about 0.0002, with only four cross-products carrying substantial coefficients.` | Blanket Diagnosis: `The missing terms are in Panel C v1; the result does not require changes to either frozen upstream stage.` | Sublated Pivot: `Fit the four observed cross-products and evaluate v2 and the two downstream targets independently.`

```bash
cd /holdout_task && sed -n '83,130p' test_dag_candidate.py && cat > /tmp/sink_probe.py <<'PY'
import pickle,numpy as np,itertools
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; t=s['train']; u=p['u']; v=p['v2']
def design(*xs): return np.column_stack((np.ones(len(xs[0])),*xs))
def fit(f,v): return np.linalg.lstsq(f,v,rcond=None)[0]
def check(name,f,v):
 w=fit(f,v); e=v-f@w; loo=e/(1-(f*np.linalg.pinv(f).T).sum(1)).reshape((-1,)+(1,)*(e.ndim-1)); print(name,f.shape[1], 'rmse',np.sqrt(np.mean(e*e,axis=0)), 'loo mae',np.mean(abs(loo),axis=0));return w
pairs=((0,1),(0,3),(1,2),(2,3))
def cross(z): return np.column_stack([z[:,i]*z[:,j] for i,j in pairs])
print('panel ranges',u.min(0),u.max(0))
for i in range(1,6):
 w=check('panel additive powers '+str(i),design(*[u**k for k in range(1,i+1)]),v)
 if 2<=i<=4: print(w)
for a in (.25,.35,.45,.5,.55,.65,1):
 for name,f in [('expm1',np.expm1(a*u)),('sin+quad',np.sin(u)+a*u*u),('tanh+quad',np.tanh(u)+a*u*u),('quad',u+a*u*u)]: check((name,a),design(f),v)
for i in (0,1):
 ns={};exec(open('/holdout_task/_frozen_nodes/node_0%d.py'%i).read(),ns);out=ns['run_node']({'__input__':s} if i==0 else {'node_00':out})
z=out['x'][:len(t['y'])]; c=design(cross(z))@fit(design(cross(u)),p['v1']); w=fit(design(u,u*u,u**3,u**4),v); b=design(z,z*z,z**3,z**4)@w
print('stage range',z.min(0),z.max(0),'stage v1 range',c.min(0),c.max(0),'stage v2 range',b.min(0),b.max(0))
y=t['y']; tr=t['traj']; print('traj singvals',np.linalg.svd(tr,compute_uv=False)); print('traj means',tr.mean(0));
for name,f in [('z',z),('v1',c),('v2',b),('v1v2',np.c_[c,b]),('z+v1+v2',np.c_[z,c,b]),('tanh(v1)',np.tanh(c)),('tanh(v2)',np.tanh(b)),('z powers 4',np.c_[z,z*z,z**3,z**4]),('v1 powers 4',np.c_[c,c*c,c**3,c**4]),('v2 powers 4',np.c_[b,b*b,b**3,b**4])]:
 print('train',name); wy=check('y',design(f),y); wt=check('traj',design(f),tr); print('wy',wy) if len(wy)<=9 else None
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/sink_probe.py
```
- **Container Exit Code**: `0` (`388 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
143
traj 5 rmse [0.20727 0.40192 0.51179 0.57997 0.62607 0.66018 0.69021 0.71444] loo mae [0.16543 0.32782 0.42495 0.48856 0.53357 0.56576 0.59357 0.61509]
wy [ 0.01267 -0.06199 -0.07909  0.19953 -0.15164]
train v1
y 5 rmse 0.0366726330662191 loo mae 0.028330440408066142
traj 5 rmse [0.0048  0.00567 0.00435 0.00512 0.00493 0.00385 0.00509 0.00464] loo mae [0.00399 0.00483 0.00373 0.00448 0.00439 0.00343 0.00446 0.00386]
wy [ 0.00297  0.91459 -0.53836 -0.59547  0.98351]
train v2
y 5 rmse 0.4882538703176351 loo mae 0.3842388548860953
traj 5 rmse [0.19724 0.38124 0.48473 0.549   0.59232 0.62464 0.65323 0.67614] loo mae [0.15805 0.31176 0.40208 0.46218 0.50401 0.53391 0.56046 0.5792 ]
wy [ 0.08984 -0.31611  0.10583 -0.22937 -0.21035]
train v1v2
y 9 rmse 0.03335838820090107 loo mae 0.02852512862893635
traj 9 rmse [0.00471 0.00524 0.0043  0.00491 0.00486 0.00372 0.00494 0.00456] loo mae [0.00421 0.00479 0.00397 0.00445 0.00452 0.00351 0.00459 0.00412]
wy [-0.00423  0.91811 -0.54588 -0.60504  0.98274  0.02703  0.00914 -0.0255   0.03639]
train z+v1+v2
y 13 rmse 0.026493359110805633 loo mae 0.02555495740522037
traj 13 rmse [0.00456 0.00519 0.00408 0.00482 0.00465 0.00352 0.0048  0.00446] loo mae [0.00442 0.00524 0.00392 0.00481 0.00474 0.00358 0.00464 0.00428]
train tanh(v1)
y 5 rmse 0.07480262227643059 loo mae 0.05609535189888488
traj 5 rmse [0.03636 0.07005 0.09195 0.10643 0.11549 0.12294 0.13018 0.13551] loo mae [0.0263  0.04975 0.06547 0.07522 0.08065 0.08625 0.09024 0.09394]
wy [ 0.00783  1.0502  -0.61899 -0.74342  1.10129]
train tanh(v2)
y 5 rmse 0.4869221392276821 loo mae 0.3820673489458165
traj 5 rmse [0.19565 0.37806 0.48074 0.54458 0.58768 0.61992 0.64844 0.6714 ] loo mae [0.15686 0.30964 0.39961 0.45907 0.50057 0.53019 0.55651 0.57509]
wy [ 0.09729 -0.35682  0.07229 -0.22033 -0.30487]
train z powers 4
y 17 rmse 0.3869155939155148 loo mae 0.4767447473347058
traj 17 rmse [0.16234 0.31966 0.41494 0.47812 0.52329 0.55931 0.58979 0.61567] loo mae [0.18462 0.3666  0.47947 0.55347 0.60822 0.65309 0.6896  0.72058]
train v1 powers 4
y 17 rmse 0.01910118822894981 loo mae 0.03975337368962861
traj 17 rmse [0.00429 0.00509 0.00388 0.00467 0.00452 0.00348 0.00452 0.00426] loo mae [0.00723 0.00715 0.00576 0.00763 0.00596 0.00581 0.00628 0.00627]
train v2 powers 4
y 17 rmse 0.3387301015498072 loo mae 0.3892058195341438
traj 17 rmse [0.12109 0.23784 0.30879 0.35834 0.39346 0.42232 0.44828 0.46808] loo mae [0.14812 0.29992 0.39723 0.46173 0.50739 0.54223 0.57184 0.59432]
```

### Turn 4 — `execute_in_remote_docker` (`38340 ms` LLM)
- **Rationale**: Create the sparse sink and run the official acceptance check. Probe residual terms separately to see whether a compact addition can materially improve the readout rather than fitting noise.
- **Alternatives Weighed**: `Include Panel C v2 in the baseline` (rejected: It did not improve readout LOO MAE and contributes little to trajectory prediction; test whether a specific residual term is warranted instead.)

```bash
cat > /holdout_task/node_02.py <<'PY'
import numpy as np


def _panel_features(x):
    a, b, c, d = x.T
    return np.column_stack((np.ones(len(x)), a*b, a*d, b*c, c*d))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    panel = study['panel_c']
    panel_map = np.linalg.lstsq(_panel_features(panel['u']), panel['v1'], rcond=None)[0]
    x = inputs['node_01']['x']
    design = np.column_stack((np.ones(len(x)), _panel_features(x) @ panel_map))
    n = len(train['y'])
    outputs = design[n:] @ np.linalg.lstsq(
        design[:n], np.column_stack((train['y'], train['traj'])), rcond=None
    )[0]
    y, state = outputs[:, 0], outputs[:, 1:]
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
PY
OPENBLAS_NUM_THREADS=1 python3 -I /holdout_task/test_dag_candidate.py
cat > /tmp/readout_probe.py <<'PY'
import pickle,numpy as np,itertools,runpy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; p=s['panel_c']; n=len(t['y']); y=t['y']; tr=t['traj']; times=s['eval_times']
z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s}); z=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']({'node_00':z})['x'][:n]; feat=runpy.run_path('/holdout_task/node_02.py')['_panel_features']; c=feat(z)@np.linalg.lstsq(feat(p['u']),p['v1'],rcond=None)[0]; g=feat(z)[:,1:]; u=p['u']; pow2=lambda a:np.c_[np.ones(len(a)),a,a*a]; b=pow2(z)@np.linalg.lstsq(pow2(u),p['v2'],rcond=None)[0]; f=np.c_[np.ones(n),g]; base=lambda f,t:np.linalg.lstsq(f,t,rcond=None)[0]; residual=lambda t:t-f@base(f,t); r=residual(y)
def score(q):
 w=base(q,y); e=y-q@w; lev=(q*np.linalg.pinv(q).T).sum(1); return np.sqrt(np.mean(e*e)),np.mean(abs(e/(1-lev))),np.mean(abs(y[-20:]-q[-20:]@base(q[:-20],y[:-20])))
print('y baseline RMSE/LOO MAE/tail MAE',score(f)); print('traj coeff',np.round(base(f,tr),4))
names=[]; cols=[]
def add(name,x): names.append(name); cols.append(x)
for tag,x in [('z',z),('b',b),('c',c),('g',g)]:
 for i,a in enumerate(x.T):
  for k,fn in [('lin',lambda a:a),('sq',np.square),('cub',lambda a:a**3),('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('sat',lambda a:a/(1+a*a)),('abs',abs),('exp',np.exp)]: add(f'{tag}{i}_{k}',fn(a))
 for i,j in itertools.combinations(range(x.shape[1]),2):
  a,bb=x[:,i],x[:,j]
  for k,v in [('prod',a*bb),('sinprod',np.sin(a*bb)),('cosprod',np.cos(a*bb)),('sin+',np.sin(a+bb)),('sin-',np.sin(a-bb)),('cos+',np.cos(a+bb)),('cos-',np.cos(a-bb)),('abs+',abs(a+bb)),('abs-',abs(a-bb)),('quot',a/(1+bb*bb)),('antiquot',bb/(1+a*a))]: add(f'{tag}{i}_{j}_{k}',v)
print('top single residual features'); print(*sorted((score(np.c_[f,v]),k) for k,v in zip(names,cols) if np.linalg.norm(residual(v))>1e-8)[:24],sep='\n')
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/readout_probe.py
```
- **Container Exit Code**: `0` (`621 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_02.py']`
- **Stdout (tail)**:
```text
-0.373  -0.3897]
 [ 0.0701  0.086   0.0483 -0.0124 -0.0715 -0.1342 -0.1882 -0.2313]
 [-0.4683 -0.8536 -1.0082 -1.0642 -1.0691 -1.0441 -1.0244 -0.9993]
 [-0.389  -0.8324 -1.1601 -1.4005 -1.5876 -1.7438 -1.8732 -1.986 ]
 [ 0.3849  0.7896  1.0514  1.2291  1.3593  1.4591  1.5442  1.6114]]
top single residual features
((0.015546614463972864, 0.012459208635975412, 0.010249502169821846), 'c1_3_prod')
((0.01707981797678424, 0.013023892919009336, 0.010306842928495458), 'c1_3_sinprod')
((0.019261326993113056, 0.016730561070337158, 0.014030877683276636), 'c1_3_cos+')
((0.021405096875328426, 0.018066457479232984, 0.01883853712162509), 'c1_3_abs+')
((0.024413959605549325, 0.019739148290835038, 0.013758216701932821), 'g1_cos')
((0.024413959605549325, 0.019739148290835038, 0.013758216701932821), 'z0_3_cosprod')
((0.02442365949049753, 0.019777030662260036, 0.014245201514255545), 'g1_sq')
((0.024645199667883155, 0.01985285619049218, 0.014763701779930222), 'g1_abs')
((0.025160188313980567, 0.02048961537473061, 0.015778133080771143), 'g1_exp')
((0.02521580220897893, 0.02041695743593748, 0.022968150959626146), 'g0_1_cos+')
((0.025223069094335856, 0.021544687249084697, 0.018096360909119497), 'c1_3_cosprod')
((0.02528551193898645, 0.020878051723872793, 0.01804870542784326), 'c1_exp')
((0.026074495778896527, 0.021771145130571413, 0.01820954392774974), 'c1_sq')
((0.026126446659357587, 0.02267397361190052, 0.02437639693941279), 'g1_2_cos+')
((0.02632236357715217, 0.020681418356594573, 0.016900026535686168), 'c1_3_sin+')
((0.02655891531508982, 0.02219261673953983, 0.018402492534254616), 'c1_cos')
((0.027105453763277516, 0.02352296069252241, 0.016739501353568394), 'c0_1_cosprod')
((0.02718000025903149, 0.022519737662831447, 0.015877057860743317), 'c1_3_quot')
((0.027605048981875588, 0.023029266726324056, 0.0211756559563444), 'g1_2_sin+')
((0.027818577952937646, 0.02213414420033167, 0.01893769202727831), 'c1_3_antiquot')
((0.028004922993925718, 0.02316479851238807, 0.029570519815707786), 'g0_1_abs+')
((0.028279956660118028, 0.023928503561169927, 0.01977583034296485), 'g1_cub')
((0.02830358616229904, 0.02353427969058539, 0.019165586723702673), 'c1_cub')
((0.028365149668617154, 0.02445825095533677, 0.02600035505726017), 'g1_2_abs+')

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d43119b66c32034a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d43119b6 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`30641 ms` LLM)
- **Rationale**: Keep the accepted sink unchanged while checking whether the interaction generalizes, and inspect its residuals for another compact term. Promote a challenger only if it improves the readout on difficult splits without changing the accurate trajectory path.
- **Alternatives Weighed**: `Submit the accepted affine baseline` (rejected: A single observed interaction cuts readout LOO MAE by more than half; test whether the gain holds on block and extrapolation splits.)

```bash
cat >/tmp/readout_refine.py <<'PY'
exec(open('/tmp/readout_probe.py').read().split("print('top single residual features')")[0])
from scipy.optimize import least_squares
base_c=np.c_[np.ones(n),c]; pair=c[:,1]*c[:,3]; f=np.c_[base_c,pair]
print('interaction',score(f),base(f,y))
res=y-f@base(f,y)
print('strongest second terms')
print(*sorted((score(np.c_[f,v]),k,base(np.c_[f,v],y)[-1]) for k,v in zip(names,cols) if np.linalg.norm(v-f@base(f,v))>1e-8)[:16],sep='\n')
q=np.column_stack([base_c]+[c[:,i]*c[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
print('full quadratic',score(q),base(q,y))
for name,extra in [('v2',b),('z',z),('both',np.c_[z,b]),('traj modes',np.linalg.svd(tr,full_matrices=False)[0][:,:2])]:
 h=np.c_[f,extra]; print(name,score(h),base(h,y))
sets=[]
for freq in (.25,.5,1.,1.5,2.,3.):
 for i,j in itertools.combinations(range(4),2):
  a,d=c[:,i],c[:,j]
  for tag,w in [('sin+',np.sin(freq*(a+d))),('sin-',np.sin(freq*(a-d))),('cos+',np.cos(freq*(a+d))),('cos-',np.cos(freq*(a-d))),('tanhprod',np.tanh(freq*a*d)),('sinprod',np.sin(freq*a*d)),('ratprod',a*d/(1+freq*(a*d)**2))]:
   sets.append((f'{tag}_{i}{j}_{freq}',np.c_[base_c,w]))
print('replace product',*sorted((score(h),key) for key,h in sets)[:10],sep='\n')
models={'affine':base_c,'product':f,'full quadratic':q,'product + z':np.c_[f,z],'product + v2':np.c_[f,b]}
for name,h in models.items():
 scores=[]; corners=[]
 for inds in np.array_split(np.random.default_rng(812).permutation(n),6)+[np.argsort(np.linalg.norm(z,axis=1))[-12:]]:
  keep=np.setdiff1d(np.arange(n),inds); e=abs(y[inds]-h[inds]@base(h[keep],y[keep])); scores.append(e.mean()); corners.append(e.max())
 print('stress',name,np.round(scores,5),'worst row',max(corners))
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/readout_refine.py
```
- **Container Exit Code**: `0` (`560 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
14, 0.01128533305182502) [ 2.42017312e-04  8.78415648e-01 -5.75738037e-01 -5.98479856e-01
  9.34402941e-01  1.90684292e-01  7.60112064e-03  7.43319265e-04
 -1.02562612e-02 -3.44950757e-03]
z (0.015009518149483737, 0.013307746088235354, 0.011066053123476942) [ 0.00251878  0.87748587 -0.57336711 -0.60045478  0.93282868  0.19063017
  0.00141529 -0.00235571  0.0076935   0.00527052]
both (0.01387291043996549, 0.013910383235514736, 0.010507454134006812) [ 0.01101842  0.87768275 -0.57463998 -0.5894705   0.93162431  0.20132754
 -0.00291232 -0.01296313  0.01099854  0.03632479 -0.06282772 -0.03121667
 -0.01321087 -0.01095135]
traj modes (0.015101752014309772, 0.012784027239658454, 0.010075132833228789) [-1.71459671e-04  6.78397174e-01 -9.63085862e-01 -2.06888062e+00
 -9.32897188e-01  1.93049500e-01 -8.50767312e+00 -7.71725608e-01]
replace product
((0.015542048793982433, 0.012461031637980936, 0.010207574797278351), 'sinprod_13_0.25')
((0.015545679726936799, 0.012463690760013947, 0.010168513151274655), 'tanhprod_13_0.25')
((0.015603072650722695, 0.012480489566774004, 0.010076894784392363), 'sinprod_13_0.5')
((0.015722980504188966, 0.012464439576211225, 0.009949009377904017), 'tanhprod_13_0.5')
((0.017017136332001647, 0.013001079601668967, 0.010623998196048316), 'ratprod_13_0.25')
((0.017079817976784248, 0.013023892919009293, 0.010306842928495402), 'sinprod_13_1.0')
((0.01722456596801089, 0.013114501549714705, 0.011068162298778944), 'tanhprod_13_1.0')
((0.01848998168777989, 0.016113073599866762, 0.013980244537488376), 'cos+_13_0.5')
((0.018555946321660223, 0.015961787137845216, 0.014008362634932334), 'cos+_13_0.25')
((0.01906555750972999, 0.013873980194725072, 0.01261580682024191), 'ratprod_13_0.5')
stress affine [0.03292 0.02621 0.0424  0.03177 0.01794 0.0237  0.0712 ] worst row 0.2649538692857041
stress product [0.01481 0.01641 0.01503 0.01016 0.00917 0.01204 0.03014] worst row 0.08771003614907324
stress full quadratic [0.0052  0.00386 0.00931 0.00391 0.00687 0.00599 0.01067] worst row 0.02905558635753569
stress product + z [0.0168  0.01987 0.01421 0.00949 0.01028 0.01339 0.03156] worst row 0.0710174417336964
stress product + v2 [0.01725 0.0198  0.01432 0.00963 0.01151 0.01345 0.03068] worst row 0.07483836729279325

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=d43119b66c32034a (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_d43119b6 (directory).
```

### Turn 6 — `execute_in_remote_docker` (`44235 ms` LLM)
- **Rationale**: The official test verifies the saved artifact and handoff. High-norm holdout checks extrapolation, nested selection checks whether the readout gain depends on seeing held-out labels, and per-horizon scoring catches trajectory failures that averaging can hide.
- **Alternatives Weighed**: `Use the full quadratic readout` (rejected: It adds eight unsupported terms and raises public-tail MAE from 0.00435 to 0.00605.)
- **ATMS NOGOOD DECLARED**: Refuted `An affine readout in Panel C v1 coordinates explains the data to the residual noise scale.` | Witness: `Adding two specific cross-products reduces readout LOO MAE from 0.02833 to 0.00485 and public-tail MAE from 0.02398 to 0.00435; the unrestricted quadratic has higher LOO MAE.` | Blanket Diagnosis: `The missing interactions are in the scalar readout; the affine trajectory already has near-noise residuals across all evaluation times.` | Sublated Pivot: `Use v1[0]·v1[2] and v1[1]·v1[3] in the readout only, and validate the saved sink on full-DAG splits with inner interaction selection.`

```bash
set -e
cat > /holdout_task/node_02.py <<'PY'
import numpy as np


def _panel_features(x):
    a, b, c, d = x.T
    return np.column_stack((np.ones(len(x)), a*b, a*d, b*c, c*d))


def _fit_predict(design, target):
    n = len(target)
    return design[n:] @ np.linalg.lstsq(design[:n], target, rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    panel = study['panel_c']
    panel_map = np.linalg.lstsq(_panel_features(panel['u']), panel['v1'], rcond=None)[0]
    v1 = _panel_features(inputs['node_01']['x']) @ panel_map
    affine = np.column_stack((np.ones(len(v1)), v1))
    a, b, c, d = v1.T
    readout = np.column_stack((affine, a*c, b*d))
    y = _fit_predict(readout, train['y'])
    state = _fit_predict(affine, train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
PY
export OPENBLAS_NUM_THREADS=1
python3 -I /holdout_task/test_dag_candidate.py
python3 -I - <<'PY'
import copy, itertools, pickle, runpy
import numpy as np
root='/holdout_task/'
study=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']
node0,node1=[runpy.run_path(root+'_frozen_nodes/node_0%d.py'%i)['run_node'] for i in range(2)]
node=runpy.run_path(root+'node_02.py'); train=study['train']; n=len(train['x'])
upstream=lambda s:node1({'node_00':node0({'__input__':s})})
features=node['_panel_features']; panel=study['panel_c']; p=features(panel['u']); v=panel['v1']
w=np.linalg.lstsq(p,v,rcond=None)[0]
loo=(v-p@w)/(1-(p*np.linalg.pinv(p).T).sum(1))[:,None]
print('Sparse panel v1 worst-channel LOO RMSE:', np.sqrt((loo**2).mean(0)).max())
assert np.sqrt((loo**2).mean(0)).max() < .001
z=upstream(study)['x'][:n]
slices=[(f'fold_{i+1}',ix) for i,ix in enumerate(np.array_split(np.random.default_rng(812).permutation(n),6))]
slices += [('highest_latent_norm', np.argsort(np.linalg.norm(z,axis=1))[-12:])]
results=[]
for name,ix in slices:
    keep=np.setdiff1d(np.arange(n),ix); s=copy.deepcopy(study)
    s['train']={k:train[k][keep] for k in ('x','y','traj')}
    y,t=s['train']['y'],s['train']['traj']
    s['train']['summary']=np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])
    s['test']={'x':train['x'][ix]}
    up=upstream(s); o=node['run_node']({'__input__':s,'node_01':up})
    ey=np.abs(o['y_hat']-train['y'][ix]); et=np.abs(o['state_pred']-train['traj'][ix])
    # Rediscover interactions using only the outer training rows and inner LOO.
    c=(features(up['x'][:len(keep)])@w); f=np.c_[np.ones(len(c)),c]
    pairs=list(itertools.combinations(range(4),2)); products=[c[:,i]*c[:,j] for i,j in pairs]
    def inner_score(ij):
        h=np.c_[f,products[ij[0]],products[ij[1]]]; inv=np.linalg.pinv(h)
        return np.mean(np.abs((y-h@(inv@y))/(1-(h*inv.T).sum(1))))
    chosen=min(itertools.combinations(range(len(pairs)),2),key=inner_score)
    held_c=features(up['x'][len(keep):])@w
    ii,jj=(pairs[i] for i in chosen)
    fit=np.c_[f,products[chosen[0]],products[chosen[1]]]
    held_fit=np.c_[np.ones(len(ix)),held_c,held_c[:,ii[0]]*held_c[:,ii[1]],held_c[:,jj[0]]*held_c[:,jj[1]]]
    nested=np.mean(np.abs(held_fit@np.linalg.lstsq(fit,y,rcond=None)[0]-train['y'][ix]))
    results.append((ey.mean(),et.mean(0).max(),nested))
    print(name,'readout MAE %.6f, worst-horizon MAE %.6f, nested readout MAE %.6f'%results[-1], 'selected pairs',ii,jj)
worst=np.max(results,axis=0)
print('Worst-slice MAE [readout, worst trajectory horizon, nested readout]:',worst)
assert np.all(worst < .038)
PY
```
- **Container Exit Code**: `0` (`578 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/node_02.py']`
- **Stdout (tail)**:
```text
{"accepted": true, "errors": [], "cv_metrics": {"public_cv_y_mae": 0.004353, "public_cv_traj_mae": 0.003983}, "exec_counts": {"node_00": 1, "node_01": 1, "node_02": 1}}
Sparse panel v1 worst-channel LOO RMSE: 0.00011833239754425902
fold_1 readout MAE 0.005476, worst-horizon MAE 0.005921, nested readout MAE 0.005476 selected pairs (0, 2) (1, 3)
fold_2 readout MAE 0.003719, worst-horizon MAE 0.005262, nested readout MAE 0.003719 selected pairs (0, 2) (1, 3)
fold_3 readout MAE 0.004852, worst-horizon MAE 0.005618, nested readout MAE 0.004852 selected pairs (0, 2) (1, 3)
fold_4 readout MAE 0.003427, worst-horizon MAE 0.005617, nested readout MAE 0.003427 selected pairs (0, 2) (1, 3)
fold_5 readout MAE 0.005835, worst-horizon MAE 0.005686, nested readout MAE 0.005835 selected pairs (0, 2) (1, 3)
fold_6 readout MAE 0.005593, worst-horizon MAE 0.005364, nested readout MAE 0.005593 selected pairs (0, 2) (1, 3)
highest_latent_norm readout MAE 0.004493, worst-horizon MAE 0.006422, nested readout MAE 0.004493 selected pairs (0, 2) (1, 3)
Worst-slice MAE [readout, worst trajectory horizon, nested readout]: [0.00583545 0.00642153 0.00583545]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=59b02c2074b17d79 (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v02_59b02c20 (directory).
```

### Turn 7 — `submit_completed_task` (`14958 ms` LLM)
- **Rationale**: The verified artifact clears the official thresholds and the high-norm holdout. Nested selection recovered the same two readout interactions in every outer split, and the sparse panel map had a worst-channel leave-one-out RMSE of 0.000118.
- **Alternatives Weighed**: `Keep the affine readout` (rejected: Its worst-slice MAE was 0.0712; the two-product readout reduced it to 0.005835.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/node_02.py` without changing either upstream node. The official test passed: readout MAE was 0.004353 and trajectory MAE was 0.003983, both below 0.038. Each node ran exactly once.`

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
