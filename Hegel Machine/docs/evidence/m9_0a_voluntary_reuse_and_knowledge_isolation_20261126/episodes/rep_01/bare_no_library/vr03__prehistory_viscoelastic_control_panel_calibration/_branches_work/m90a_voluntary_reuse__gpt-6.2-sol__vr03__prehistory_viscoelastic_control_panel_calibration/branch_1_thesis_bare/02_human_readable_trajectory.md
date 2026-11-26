# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Bare_ReAct_LiveGPT` | `vr03__prehistory_viscoelastic_control_panel_calibration`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Viscoelastic Hereditary Memory Extraction, Reference Control Panel Alignment & Simpson-Deconfounded Risk` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_cohort_deconfounding_and_reference_balancing` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `5 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `79,037` (`75,858` / `9,856` / `3,179`) |
| **Total Episode Wall Time (s)** | `479.68s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`12090 ms` LLM)
- **Rationale**: Inspect the public cohort schema and available numerical libraries before choosing the implementation.

```bash
ls -la /holdout_task; python - <<'PY'
import pickle,numpy as np
p='/holdout_task/public_viscoelastic_cohort.pkl'
x=pickle.load(open(p,'rb'))
def show(a,depth=0):
 if depth>4:return
 print(' '*depth,type(a).__name__,getattr(a,'shape',''),getattr(a,'dtype',''),len(a) if isinstance(a,(list,tuple,dict)) else (a if np.isscalar(a) else ''))
 if isinstance(a,dict):
  for k,v in a.items():
   print(' '*depth,k); show(v,depth+1)
 if isinstance(a,(list,tuple)) and len(a):show(a[0],depth+1)
show(x)
try: import scipy; print('scipy',scipy.__version__)
except ImportError: pass
PY
```
- **Container Exit Code**: `127` (`363 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1120
drwxr-xr-x 2 root root    4096 Nov 26 00:51 .
drwxr-xr-x 1 root root    4096 Nov 26 00:51 ..
-rw-r--r-- 1 root root 1136378 Nov 26 00:51 public_viscoelastic_cohort.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`12205 ms` LLM)
- **Rationale**: Use the available Python 3 executable to establish the cohort layout and numerical dependencies.

```bash
command -v python3; ls -la /pretrain_task; python3 - <<'PY'
import pickle, importlib.util
x=pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))
def show(a,depth=0):
 if depth>5:return
 print(' '*depth,type(a).__name__,getattr(a,'shape',''),getattr(a,'dtype',''), len(a) if isinstance(a,(list,tuple,dict)) else '')
 if isinstance(a,dict):
  for k,v in a.items():
   print(' '*depth,k); show(v,depth+1)
 if isinstance(a,(list,tuple)) and a: show(a[0],depth+1)
show(x)
print('scipy',importlib.util.find_spec('scipy') is not None)
PY
```
- **Container Exit Code**: `0` (`512 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
total 8
drwxr-xr-x 2 root root 4096 Nov 23 09:31 .
drwxr-xr-x 1 root root 4096 Nov 26 00:51 ..
 list   2
  dict   2
  cohort
   dict   3
   taus
    list   2
     float   
   train
    dict   6
    center_id
     ndarray (288,) int64 
    stratum
     ndarray (288,) int64 
    controls
     ndarray (288, 5) float64 
    missing_channel_mask
     ndarray (288, 3) bool 
    traces
     list   288
      dict   5
      past_timestamps
      past_strain_history
      timestamps
      strain_u
      stress_obs
    y
     ndarray (288,) int64 
   query
    dict   5
    center_id
     ndarray (180,) int64 
    stratum
     ndarray (180,) int64 
    controls
     ndarray (180, 5) float64 
    missing_channel_mask
     ndarray (180, 3) bool 
    traces
     list   180
      dict   5
      past_timestamps
      past_strain_history
      timestamps
      strain_u
      stress_obs
  gt
   dict   4
   Z_query
    ndarray (180, 3) float64 
   y_query
    ndarray (180,) int64 
   prob_query
    ndarray (180,) float64 
   beta
    ndarray (3,) float64 
scipy True
```

### Turn 3 — `execute_in_remote_docker` (`399973 ms` LLM)
- **Rationale**: Implement memory extraction and control alignment, then check public modulus error and center summaries. Large observed error would point to the extraction or alignment rather than the risk model.
- **Alternatives Weighed**: `Ordinary least squares over all timepoints` (rejected: Large telemetry glitches can distort all three fitted moduli.)

```bash
cat > /holdout_task/viscoelastic_panel_calibration.py <<'PY'
"""Fit hereditary moduli and calibrate failure risk across rheometry centers."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(trace, taus):
    t = np.concatenate((trace['past_timestamps'], trace['timestamps']))
    u = np.concatenate((trace['past_strain_history'], trace['strain_u']))
    start = len(trace['past_timestamps'])
    fraction = -np.expm1(-np.diff(t)[:, None] / taus)
    memory = np.full_like(taus, u[0])
    design = np.empty((len(t) - start, 4))
    design[:, 0] = 1
    design[:, 1] = u[start:]
    for k, f in enumerate(fraction, 1):
        memory += f * (u[k] - memory)
        if k >= start:
            design[k - start, 2:] = memory
    return design


def _robust_moduli(design, stress, samples):
    # Pick a least-median seed from minimal fits, then re-fit the consensus
    # so clean observations, not just the four seed points, determine moduli.
    x = design[samples]
    good = np.abs(np.linalg.det(x)) > 1e-12
    fits = np.linalg.solve(x[good], stress[samples[good], None]).squeeze(-1)
    ordinary = np.linalg.lstsq(design, stress, rcond=None)[0]
    fits = np.vstack((fits, ordinary))
    residuals = np.abs(stress[:, None] - design @ fits.T)
    fit = fits[np.argmin(np.median(residuals, axis=0))]
    for _ in range(4):
        residual = stress - design @ fit
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual - np.median(residual)) <= max(3.5 * mad, 1e-9)
        if np.linalg.matrix_rank(design[keep]) < design.shape[1]:
            break
        fit = np.linalg.lstsq(design[keep], stress[keep], rcond=None)[0]
    return fit[1:]


def _aligned(split, taus):
    rng = np.random.default_rng(2025)
    n_time = len(split['traces'][0]['timestamps'])
    samples = np.array([rng.choice(n_time, 4, replace=False) for _ in range(64)])
    z = np.array([_robust_moduli(_design(trace, taus), np.asarray(trace['stress_obs']), samples)
                  for trace in split['traces']])
    centers = np.asarray(split['center_id'])
    controls = np.asarray(split['controls'])
    for center in np.unique(centers):
        cell = centers == center
        panel = controls[cell]
        populated = np.sum(np.isfinite(panel), axis=0) >= 2
        panel = panel[:, populated]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        z[cell] = (z[cell] - offset) / gain
    z[np.asarray(split['missing_channel_mask'], dtype=bool)] = np.nan
    return z
PY
cd /holdout_task && OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle,numpy as np
from viscoelastic_panel_calibration import _aligned
for pack in pickle.load(open('public_viscoelastic_cohort.pkl','rb')):
 c, gt = pack['cohort'], pack['gt']; print('taus', c['taus'], 'beta',gt['beta']);
 for name in ['train','query']:
  d=c[name]; z=_aligned(d,np.asarray(c['taus'])); print(name,'means',np.round(np.nanmean(z,axis=0),2))
  for center in np.unique(d['center_id']):
   i=d['center_id']==center; print(center, np.bincount(d['stratum'][i]),'z',np.round(np.nanmean(z[i],axis=0),2),'y',round(np.mean(d.get('y',gt['y_query'])[i]),3));
  if name=='query': print('MAE observed',np.nanmean(abs(z-gt['Z_query'])), 'MAE cols',np.nanmean(abs(z-gt['Z_query']),axis=0))
PY
```
- **Container Exit Code**: `0` (`1110 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_panel_calibration.py']`
- **Stdout (tail)**:
```text
taus [0.55, 2.6] beta [ 2.15 -1.95  1.8 ]
train means [ 0.    0.02 -0.12]
0 [35 22 15] z [0.45 0.42 0.26] y 0.528
1 [41 17 14] z [0.34 0.64 0.33] y 0.458
2 [35 18 19] z [-0.45 -0.53 -0.61] y 0.347
3 [35 24 13] z [-0.35 -0.42 -0.44] y 0.389
query means [-0.17 -0.03 -0.12]
4 [53 24 13] z [0.28 0.27 0.22] y 0.489
5 [57 21 12] z [-0.62 -0.32 -0.47] y 0.278
MAE observed 0.03490765867241297 MAE cols [0.03170231 0.0368557  0.0360933 ]
taus [0.55, 2.6] beta [ 2.15 -1.95  1.8 ]
train means [0.01 0.11 0.23]
0 [42 19 11] z [0.67 0.86 0.64] y 0.569
1 [36 24 12] z [0.5  0.46 0.92] y 0.625
2 [33 27 12] z [-0.62 -0.53 -0.34] y 0.389
3 [39 22 11] z [-0.55 -0.4  -0.34] y 0.361
query means [ 0.05  0.15 -0.04]
4 [52 26 12] z [0.43 0.6  0.32] y 0.444
5 [50 22 18] z [-0.33 -0.3  -0.4 ] y 0.4
MAE observed 0.030170246808417713 MAE cols [0.02852259 0.02940091 0.03246532]
```

### Turn 4 — `execute_in_remote_docker` (`32183 ms` LLM)
- **Rationale**: Complete cell-balanced imputation and stratum-adjusted risk calibration, then check whether either public cohort misses a requested metric or output invariant.
- **Alternatives Weighed**: `Include center identity as a risk predictor` (rejected: The control panel already estimates center scanner effects, and the specified risk model has no center term.)

```bash
cat >> /holdout_task/viscoelastic_panel_calibration.py <<'PY'


def _risk_design(z, strata):
    # Reference-code stratum 0; the slopes are shared by all strata and centers.
    return np.column_stack((np.ones(len(z)), strata == 1, strata == 2, z))


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Return aligned query moduli, failure probabilities, and fitted slopes."""
    train, query = cohort['train'], cohort['query']
    taus = np.asarray(cohort['taus'], dtype=float)
    z_train, z_query = _aligned(train, taus), _aligned(query, taus)
    strata = np.asarray(train['stratum'])
    cells = np.column_stack((train['center_id'], strata))
    _, membership, counts = np.unique(cells, axis=0, return_inverse=True,
                                     return_counts=True)
    # Each training center–stratum cell contributes one mean per channel.
    means = np.array([np.nanmean(z_train[membership == k], axis=0)
                      for k in range(len(counts))])
    fill = np.nanmean(means, axis=0)
    z_train = np.where(np.isnan(z_train), fill, z_train)
    z_query = np.where(np.isnan(z_query), fill, z_query)

    design = _risk_design(z_train, strata)
    y = np.asarray(train['y'], dtype=float)
    weights = 1.0 / (len(counts) * counts[membership])
    ridge = 0.003  # Mild slope shrinkage for the finite, reweighted sample.

    def objective(coef):
        eta = design @ coef
        loss = weights @ (np.logaddexp(0, eta) - y * eta)
        grad = design.T @ (weights * (expit(eta) - y))
        slopes = coef[3:]
        grad[3:] += ridge * slopes
        return loss + 0.5 * ridge * (slopes @ slopes), grad

    coef = minimize(objective, np.zeros(design.shape[1]), jac=True,
                    method='BFGS', options={'gtol': 1e-9}).x
    probabilities = expit(_risk_design(z_query, np.asarray(query['stratum'])) @ coef)
    return {
        'aligned_query_moduli': z_query,
        'query_probabilities': np.clip(probabilities, np.finfo(float).eps,
                                       1.0 - np.finfo(float).eps),
        'modulus_weights': coef[3:],
    }
PY
cd /holdout_task && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 - <<'PY'
import pickle, numpy as np
from viscoelastic_panel_calibration import calibrate_viscoelastic_cohort
for n, pack in enumerate(pickle.load(open('public_viscoelastic_cohort.pkl','rb')),1):
 c, truth = pack['cohort'], pack['gt']
 r = calibrate_viscoelastic_cohort(c)
 z, p, b = (r[k] for k in ('aligned_query_moduli','query_probabilities','modulus_weights'))
 y, centers = truth['y_query'], c['query']['center_id']
 assert z.shape == (len(y),3) and p.shape == y.shape and b.shape == (3,)
 assert np.isfinite(z).all() and np.isfinite(b).all() and ((p>0)&(p<1)).all()
 aucs, wins, pairs = [], 0, 0
 for center in np.unique(centers):
  pc, yc = p[centers==center], y[centers==center]
  d = pc[yc==1,None] - pc[yc==0]
  w = np.sum(d>0)+.5*np.sum(d==0)
  wins+=w; pairs+=d.size; aucs.append(w/d.size)
 print(f'{n}: MAE={np.mean(abs(z-truth["Z_query"])):.4f} Brier={np.mean((p-y)**2):.4f} matched AUC={wins/pairs:.4f} per-center={np.round(aucs,4)} prob RMSE={np.mean((p-truth["prob_query"])**2)**.5:.4f} beta={b.round(4)}')
PY
```
- **Container Exit Code**: `0` (`1092 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_panel_calibration.py']`
- **Stdout (tail)**:
```text
1: MAE=0.0722 Brier=0.0928 matched AUC=0.9433 per-center=[0.9575 0.9255] prob RMSE=0.1036 beta=[ 1.9479 -1.5809  1.6001]
2: MAE=0.0871 Brier=0.1160 matched AUC=0.9057 per-center=[0.884 0.928] prob RMSE=0.1355 beta=[ 1.5611 -1.2639  1.3276]
```

### Turn 5 — `submit_completed_task` (`11685 ms` LLM)
- **Rationale**: The required file is on disk, the output shapes and probability bounds were verified, and both public cohorts met the requested thresholds.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. Both public cohorts met all targets: modulus MAE 0.0722 and 0.0871, within-center AUROC 0.9433 and 0.9057, and Brier scores 0.0928 and 0.1160.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-M90a-VoluntaryReuse-4",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
