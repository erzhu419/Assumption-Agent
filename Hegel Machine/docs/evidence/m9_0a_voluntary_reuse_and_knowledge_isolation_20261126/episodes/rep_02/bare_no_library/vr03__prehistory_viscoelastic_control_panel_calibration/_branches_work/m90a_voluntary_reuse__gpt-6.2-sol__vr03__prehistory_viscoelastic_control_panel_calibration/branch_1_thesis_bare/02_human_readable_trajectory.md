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
| **Total Tokens (Prompt / Cached / Out)** | `79,728` (`76,535` / `32,640` / `3,193`) |
| **Total Episode Wall Time (s)** | `114.03s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`12525 ms` LLM)
- **Rationale**: Inspect the public input structure and available numerical libraries before choosing the extraction and calibration implementation.

```bash
cd /holdout_task && ls -lh && python - <<'PY'
import pickle, numpy as np
p=pickle.load(open('public_viscoelastic_cohort.pkl','rb'))
def show(x, d=0):
 if isinstance(x,dict) and d<5:
  for k,v in x.items():
   print(' '*d, k, end=': ')
   if isinstance(v,dict): print()
   show(v,d+1)
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x));
  if len(x): show(x[0],d+1)
 else:
  print(type(x).__name__,getattr(x,'shape',''),getattr(x,'dtype',''), repr(x)[:120] if not isinstance(x,np.ndarray) or x.size<20 else '')
show(p)
import importlib.util
for name in ('numpy','scipy','sklearn'): print(name, bool(importlib.util.find_spec(name)))
PY
```
- **Container Exit Code**: `127` (`304 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1.1M
-rw-r--r-- 1 root root 1.1M Nov 26 00:51 public_viscoelastic_cohort.pkl
```
- **Stderr (tail)**:
```text
bash: line 1: python: command not found
```

### Turn 2 — `execute_in_remote_docker` (`12417 ms` LLM)
- **Rationale**: Use python3 after the missing python executable; inspect the actual schema and numerical dependencies to avoid implementing against guessed field names.

```bash
cd /holdout_task; command -v python3; ls -d /opt/*; python3 - <<'PY'
import pickle, importlib.util
for name in ('numpy','scipy','sklearn'): print(name, bool(importlib.util.find_spec(name)))
p=pickle.load(open('public_viscoelastic_cohort.pkl','rb'))
def show(x,d=0):
 if isinstance(x,dict):
  for k,v in x.items():
   print(' '*d,k,end=': ')
   if isinstance(v,dict): print()
   show(v,d+1)
 elif isinstance(x,(list,tuple)):
  print(type(x).__name__,len(x));
  if x:
   if isinstance(x[0],dict): show(x[0],d+1)
   else: print(str(x[:3])[:120])
 else: print(type(x).__name__,getattr(x,'shape',''),getattr(x,'dtype',''))
show(p)
PY
```
- **Container Exit Code**: `0` (`620 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
/usr/bin/python3
numpy True
scipy True
sklearn True
list 2
  cohort: 
   taus: list 2
[0.55, 2.6]
   train: 
    center_id: ndarray (288,) int64
    stratum: ndarray (288,) int64
    controls: ndarray (288, 5) float64
    missing_channel_mask: ndarray (288, 3) bool
    traces: list 288
      past_timestamps: ndarray (6,) float64
      past_strain_history: ndarray (6,) float64
      timestamps: ndarray (36,) float64
      strain_u: ndarray (36,) float64
      stress_obs: ndarray (36,) float64
    y: ndarray (288,) int64
   query: 
    center_id: ndarray (180,) int64
    stratum: ndarray (180,) int64
    controls: ndarray (180, 5) float64
    missing_channel_mask: ndarray (180, 3) bool
    traces: list 180
      past_timestamps: ndarray (6,) float64
      past_strain_history: ndarray (6,) float64
      timestamps: ndarray (36,) float64
      strain_u: ndarray (36,) float64
      stress_obs: ndarray (36,) float64
  gt: 
   Z_query: ndarray (180, 3) float64
   y_query: ndarray (180,) int64
   prob_query: ndarray (180,) float64
   beta: ndarray (3,) float64
```
- **Stderr (tail)**:
```text
ls: cannot access '/opt/*': No such file or directory
```

### Turn 3 — `execute_in_remote_docker` (`15079 ms` LLM)
- **Rationale**: Check actual cell prevalence and scanner statistics; these determine whether the specified cell balancing or control estimation needs special treatment. Trace ranges inform robust regression.

```bash
cd /holdout_task && python3 - <<'PY'
import pickle,numpy as np
for j,d in enumerate(pickle.load(open('public_viscoelastic_cohort.pkl','rb'))):
 print('case',j,'beta',d['gt']['beta']); co=d['cohort']
 for split in ['train','query']:
  a=co[split]; y=a.get('y',d['gt']['y_query']); print(split)
  for k in np.unique(a['center_id']):
   m=a['center_id']==k; x=a['controls'][m]; print(k,[(int(sum(ms:=m&(a['stratum']==s))),round(y[ms].mean(),2)) for s in range(3)],np.round(np.nanmean(x,axis=0),2),np.round(np.nanstd(x,axis=0),2))
 for t in co['train']['traces'][:4]:
  print('obs range',np.round(np.quantile(t['stress_obs'],[0,.1,.5,.9,1]),2),'strain range', np.round(np.quantile(t['strain_u'],[0,.5,1]),2))
PY
```
- **Container Exit Code**: `0` (`689 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
case 0 beta [ 2.15 -1.95  1.8 ]
train
0 [(35, 0.51), (22, 0.45), (15, 0.67)] [4.38 4.15 4.13 4.28 3.96] [1.43 1.4  1.36 1.37 1.49]
1 [(41, 0.37), (17, 0.47), (14, 0.71)] [3.59 3.86 3.92 3.72 3.87] [0.8  0.77 0.73 0.74 0.73]
2 [(35, 0.31), (18, 0.33), (19, 0.42)] [-3.91 -3.95 -3.86 -4.11 -4.42] [1.39 1.35 1.15 1.31 1.25]
3 [(35, 0.4), (24, 0.33), (13, 0.46)] [-3.62 -3.81 -3.71 -3.55 -3.79] [0.78 0.8  0.86 0.73 0.84]
query
4 [(53, 0.4), (24, 0.62), (13, 0.62)] [-3.38 -3.23 -3.49 -3.65 -3.58] [1.28 1.31 1.57 1.63 1.57]
5 [(57, 0.3), (21, 0.24), (12, 0.25)] [3.91 3.81 3.92 3.87 4.02] [0.75 0.79 0.58 0.79 0.67]
obs range [-12.    -2.63   9.41  15.93  22.34] strain range [-1.21  0.47  1.43]
obs range [-10.06  -7.42   7.14  12.14  16.84] strain range [-1.2   0.39  1.42]
obs range [-7.09 -3.21  6.99 16.38 23.09] strain range [-1.19  0.35  1.41]
obs range [-8.51 -6.69  7.79 13.19 14.93] strain range [-1.21  0.49  1.43]
case 1 beta [ 2.15 -1.95  1.8 ]
train
0 [(42, 0.55), (19, 0.68), (11, 0.45)] [3.9  4.4  3.96 3.98 3.97] [1.2  1.23 1.47 1.27 1.33]
1 [(36, 0.47), (24, 0.67), (12, 1.0)] [3.79 3.79 3.73 3.72 3.73] [0.84 0.69 0.84 0.68 0.7 ]
2 [(33, 0.21), (27, 0.56), (12, 0.5)] [-3.88 -3.9  -3.97 -3.92 -3.77] [1.45 1.22 1.28 1.25 1.32]
3 [(39, 0.28), (22, 0.36), (11, 0.64)] [-3.96 -3.76 -3.57 -3.58 -3.72] [0.81 0.75 0.96 0.88 0.75]
query
4 [(52, 0.42), (26, 0.42), (12, 0.58)] [-3.54 -3.14 -3.62 -3.66 -3.49] [1.4  1.11 1.58 1.58 1.52]
5 [(50, 0.38), (22, 0.36), (18, 0.5)] [4.01 3.74 3.9  3.83 3.89] [0.75 0.78 0.78 0.8  0.69]
obs range [-19.4  -12.88   3.93  11.63  16.22] strain range [-1.2   0.42  1.43]
obs range [-9.05 -6.66  2.35  8.59 15.02] strain range [-1.2   0.43  1.42]
obs range [-8.22 -4.72 10.02 17.62 19.96] strain range [-1.21  0.41  1.42]
obs range [-12.84  -4.73   9.45  15.3   23.1 ] strain range [-1.2   0.43  1.41]
```

### Turn 4 — `execute_in_remote_docker` (`52269 ms` LLM)
- **Rationale**: Implement both stages and check public output shapes, bounds, and accuracy; separate observed-channel and imputed-channel errors to locate any alignment failure.
- **Alternatives Weighed**: `Ordinary least squares on all stress observations` (rejected: Large telemetry glitches can move the fitted moduli; use a trimmed fit to identify clean observations before refitting.)

```bash
cd /holdout_task && cat > viscoelastic_panel_calibration.py <<'PY'
"""Extract hereditary moduli and fit risk in the control-aligned coordinates."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(trace, taus):
    past_t = np.asarray(trace['past_timestamps'], dtype=float)
    past_u = np.asarray(trace['past_strain_history'], dtype=float)
    t = np.asarray(trace['timestamps'], dtype=float)
    u = np.asarray(trace['strain_u'], dtype=float)
    times = np.concatenate((past_t, t))
    strain = np.concatenate((past_u, u))
    fraction = -np.expm1(-np.diff(times)[:, None] / taus)
    memory = np.full(taus.size, past_u[0])
    design = np.column_stack((np.ones(t.size), u, np.empty((t.size, taus.size))))
    for k, (f, value) in enumerate(zip(fraction, strain[1:]), start=1):
        memory += f * (value - memory)
        if k >= past_t.size:
            design[k - past_t.size, 2:] = memory
    return design


def _robust_moduli(design, stress, starts):
    """Concentrate least-trimmed squares, then refit on all clean points."""
    def fit(rows):
        return np.linalg.lstsq(design[rows], stress[rows], rcond=None)[0]

    n = stress.size
    h = int(0.8 * n)
    best_loss = np.inf
    for start in (slice(None), *starts):
        coef = fit(start)
        previous = None
        for _ in range(12):
            error = (stress - design @ coef) ** 2
            rows = np.sort(np.argpartition(error, h - 1)[:h])
            if np.array_equal(rows, previous):
                break
            coef = fit(rows)
            previous = rows
        error = stress - design @ coef
        loss = np.sum(np.partition(error ** 2, h - 1)[:h])
        if loss < best_loss:
            best_loss, best_error, best_coef = loss, error, coef
        # The specified glitches are at least 4.5 stress units. A fit
        # with this much agreement already isolates the clean trace.
        if np.count_nonzero(np.abs(error) < 1.0) >= h:
            break
    clean = np.abs(best_error) < 2.25
    return (fit(clean) if np.count_nonzero(clean) >= h else best_coef)[1:]


def _aligned(split, taus, starts):
    moduli = np.array([
        _robust_moduli(_design(trace, taus), np.asarray(trace['stress_obs']), starts)
        for trace in split['traces']
    ])
    moduli[np.asarray(split['missing_channel_mask'], dtype=bool)] = np.nan
    centers = np.asarray(split['center_id'])
    controls = np.asarray(split['controls'], dtype=float)
    for center in np.unique(centers):
        rows = centers == center
        panel = controls[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        moduli[rows] = (moduli[rows] - offset) / gain
    return moduli


def _imputation_means(z, centers, strata):
    cells = np.unique(np.column_stack((centers, strata)), axis=0)
    means = []
    for c, s in cells:
        cell = z[(centers == c) & (strata == s)]
        count = np.isfinite(cell).sum(axis=0)
        means.append(np.divide(np.nansum(cell, axis=0), count,
                               out=np.full(z.shape[1], np.nan), where=count > 0))
    return np.nanmean(means, axis=0)


def _risk_design(z, strata):
    return np.column_stack((np.eye(3)[strata], z))


def _fit_risk(design, y):
    # A small slope penalty prevents an unstable fit without penalizing
    # the three stratum intercepts.
    def objective(coef):
        eta = design @ coef
        loss = np.logaddexp(0, (1 - 2 * y) * eta).sum()
        grad = design.T @ (expit(eta) - y)
        loss += 0.25 * (coef[3:] @ coef[3:])
        grad[3:] += 0.5 * coef[3:]
        return loss, grad

    return minimize(objective, np.zeros(design.shape[1]), jac=True,
                    method='L-BFGS-B', options={'gtol': 1e-8, 'ftol': 1e-12}).x


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Return imputed aligned query moduli, failure probabilities and slopes."""
    train, query = cohort['train'], cohort['query']
    taus = np.asarray(cohort['taus'], dtype=float)
    rng = np.random.default_rng(0)
    n = len(train['traces'][0]['timestamps'])
    starts = tuple(rng.choice(n, 8, replace=False) for _ in range(24))
    z_train = _aligned(train, taus, starts)
    z_query = _aligned(query, taus, starts)
    strata = np.asarray(train['stratum'], dtype=int)
    means = _imputation_means(z_train, np.asarray(train['center_id']), strata)
    z_train = np.where(np.isnan(z_train), means, z_train)
    z_query = np.where(np.isnan(z_query), means, z_query)
    coef = _fit_risk(_risk_design(z_train, strata), np.asarray(train['y'], dtype=float))
    eta_query = _risk_design(z_query, np.asarray(query['stratum'], dtype=int)) @ coef
    return {'aligned_query_moduli': z_query,
            'query_probabilities': np.clip(expit(eta_query), 1e-12, 1 - 1e-12),
            'modulus_weights': coef[3:]}
PY
cat > /tmp/verify_visco.py <<'PY'
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from viscoelastic_panel_calibration import calibrate_viscoelastic_cohort
for i, item in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb'))):
    co, gt = item['cohort'], item['gt']
    out = calibrate_viscoelastic_cohort(co)
    z, p, b = (out[k] for k in ('aligned_query_moduli','query_probabilities','modulus_weights'))
    y, c = gt['y_query'], co['query']['center_id']
    n = len(y)
    assert z.shape==(n,3) and p.shape==(n,) and b.shape==(3,)
    assert np.isfinite(z).all() and np.isfinite(b).all() and ((p>0)&(p<1)).all()
    print(i, 'MAE', np.abs(z-gt['Z_query']).mean(), 'Brier', np.mean((y-p)**2), 'AUC', [roc_auc_score(y[c==j],p[c==j]) for j in np.unique(c)], 'beta', b)
    for j in np.unique(c):
        for m in [~co['query']['missing_channel_mask'], co['query']['missing_channel_mask']]:
            print(j,int(m.sum()),np.round(np.mean(np.abs(z[c==j][m[c==j]]-gt['Z_query'][c==j][m[c==j]])),4),end='; ')
    print()
PY
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/holdout_task python3 /tmp/verify_visco.py
```
- **Container Exit Code**: `0` (`1520 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_panel_calibration.py']`
- **Stdout (tail)**:
```text
0 MAE 0.0721827240016916 Brier 0.09285594254198691 AUC [0.9560276679841897, 0.9255384615384615] beta [ 1.99698743 -1.57463618  1.68653164]
4 514 0.0505; 4 26 0.6039; 5 514 0.0191; 5 26 0.9865; 
1 MAE 0.08707594057658564 Brier 0.11541459726863662 AUC [0.8825, 0.9279835390946503] beta [ 1.66440606 -1.40109393  1.44713955]
4 503 0.0255; 4 37 0.8858; 5 503 0.0347; 5 37 0.8354;
```

### Turn 5 — `submit_completed_task` (`9805 ms` LLM)
- **Rationale**: The required module is saved, and both public cohorts passed the output contract and all three performance targets.
- **Submitted Final Answer / Summary**: `Implemented `/holdout_task/viscoelastic_panel_calibration.py` and verified both public cohorts. Both met all targets: aligned-moduli MAE was 0.072–0.087, within-center AUROC was 0.883–0.956, and Brier score was 0.093–0.115.`

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
