# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Hegel_M6_9_Antithesis_LiveGPT` | `vr03__prehistory_viscoelastic_control_panel_calibration`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-M90a-VoluntaryReuse-4` |
| **Domain** | `Viscoelastic Hereditary Memory Extraction, Reference Control Panel Alignment & Simpson-Deconfounded Risk` |
| **Wave / Wormhole Family** | `m90a_voluntary_reuse` / `fam_cohort_deconfounding_and_reference_balancing` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Hegel_M6_9_Antithesis_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `10 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `248,567` (`238,367` / `65,152` / `10,200`) |
| **Total Episode Wall Time (s)** | `319.24s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`17340 ms` LLM)
- **Rationale**: Inspect only compact schema and dependency summaries to detect interface mismatches and identify the reference data available for center-held-out validation.
- **Alternatives Weighed**: `Implement against the prose schema immediately` (rejected: The actual field names, available truth arrays, and installed dependencies determine which implementation and validation are possible.)

```bash
ls -lh /holdout_task; python3 - <<'PY'
import pickle
import numpy as np
with open('/holdout_task/public_viscoelastic_cohort.pkl','rb') as f:
    obj=pickle.load(f)
def show(x, depth=0):
    p=' '*depth
    if isinstance(x,dict):
        print(p,'dict',len(x),list(x)[:18]);
        if depth<5:
            for k,v in list(x.items())[:18]:
                print(p,k,end=': '); show(v,depth+1)
    elif isinstance(x,(list,tuple)):
        print(p,type(x).__name__,len(x));
        if x and depth<5: show(x[0],depth+1)
    elif hasattr(x,'shape'): print(p,type(x).__name__,x.shape,x.dtype, x if x.size<=5 else '')
    else: print(p,type(x).__name__,repr(x)[:160])
show(obj)
for name in ('scipy','sklearn'):
    try:
        m=__import__(name); print(name,m.__version__)
    except ImportError: print(name,'not installed')
PY
```
- **Container Exit Code**: `0` (`1153 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 1.1M
-rw-r--r-- 1 root root 1.1M Nov 26 00:46 public_viscoelastic_cohort.pkl
 list 2
  dict 2 ['cohort', 'gt']
  cohort:    dict 3 ['taus', 'train', 'query']
   taus:     list 2
     float 0.55
   train:     dict 6 ['center_id', 'stratum', 'controls', 'missing_channel_mask', 'traces', 'y']
    center_id:      ndarray (288,) int64 
    stratum:      ndarray (288,) int64 
    controls:      ndarray (288, 5) float64 
    missing_channel_mask:      ndarray (288, 3) bool 
    traces:      list 288
      dict 5 ['past_timestamps', 'past_strain_history', 'timestamps', 'strain_u', 'stress_obs']
    y:      ndarray (288,) int64 
   query:     dict 5 ['center_id', 'stratum', 'controls', 'missing_channel_mask', 'traces']
    center_id:      ndarray (180,) int64 
    stratum:      ndarray (180,) int64 
    controls:      ndarray (180, 5) float64 
    missing_channel_mask:      ndarray (180, 3) bool 
    traces:      list 180
      dict 5 ['past_timestamps', 'past_strain_history', 'timestamps', 'strain_u', 'stress_obs']
  gt:    dict 4 ['Z_query', 'y_query', 'prob_query', 'beta']
   Z_query:     ndarray (180, 3) float64 
   y_query:     ndarray (180,) int64 
   prob_query:     ndarray (180,) float64 
   beta:     ndarray (3,) float64 [ 2.15 -1.95  1.8 ]
scipy 1.13.0
sklearn 1.5.1
```

### Turn 2 — `execute_in_remote_docker` (`31068 ms` LLM)
- **Rationale**: Measure residual separation and design conditioning to choose a robust fit, then compare center and stratum summaries to see which nuisance effects need explicit treatment.
- **Alternatives Weighed**: `Fit pooled logistic regression immediately` (rejected: Center prevalence may reflect latent composition or a center-specific baseline; the two cases require different calibration.)

```bash
cat > /tmp/inspect_visco.py <<'PY'
import pickle
import numpy as np
from scipy.special import logit
P='/holdout_task/public_viscoelastic_cohort.pkl'
def load():
    with open(P,'rb') as f: return pickle.load(f)
def design(trace,taus):
    tp=np.asarray(trace['past_timestamps']); u=np.asarray(trace['strain_u']); t=np.asarray(trace['timestamps']); hp=np.asarray(trace['past_strain_history']); taus=np.asarray(taus)
    x=np.r_[hp,u]; time=np.r_[tp,t]; m=np.empty((len(x),2)); m[0]=x[0]
    for k,a in enumerate(np.exp(-np.diff(time)[:,None]/taus),1): m[k]=a*m[k-1]+(1-a)*x[k]
    return np.column_stack((np.ones(len(t)), u, m[len(tp):]))
if __name__=='__main__':
    R=np.random.default_rng(518)
    subs=np.array([R.choice(36,4,replace=False) for _ in range(140)])
    for j,d in enumerate(load()):
        D,G=d['cohort'],d['gt']; print('case',j,'taus',D['taus'],'beta',G['beta'])
        for key,block in [(x,D[x]) for x in ('train','query')]:
            raw=[]; res=[]; cond=[]
            for tr in block['traces']:
                X=design(tr,D['taus']); y=np.array(tr['stress_obs']); cond.append(np.linalg.cond(X))
                cand=np.einsum('ijk,ik->ij',np.linalg.pinv(X[subs]),y[subs]); e=np.abs(y[:,None]-X@cand.T)
                fit=cand[np.argmin(np.median(e,axis=0))]; err=np.abs(y-X@fit); ok=err<=max(0.1,5*np.median(err)); fit=np.linalg.lstsq(X[ok],y[ok],rcond=None)[0]; raw.append(fit[1:]); res.append(np.sort(np.abs(y-X@fit)))
            raw=np.array(raw); res=np.array(res); keep=~block['missing_channel_mask']; controls=block['controls']; s=block['stratum']; cid=block['center_id']; y=block.get('y',G['y_query']); print(key,'condition',np.round(np.quantile(cond,[0,.5,.95,1]),2),'order residual median',np.round(np.median(res,axis=0),4),'mask',np.mean(~keep,axis=0))
            for c in np.unique(cid):
                i=cid==c; panel=controls[i]; off=np.median(np.nanmean(panel,axis=0)); gain=np.median(np.nanstd(panel,axis=0)); Z=(raw[i]-off)/gain
                print('center',c,'panel',np.round([off,gain],3),'strata n/p',[(int(sum(i&(s==t))),round(y[i&(s==t)].mean(),3)) for t in range(3)],'Z cell means',np.round([Z[s[i]==t].mean(0) for t in range(3)],3))
                if key=='query':
                    gt=G['Z_query'][i]; print('  observed MAE',np.abs(Z-gt)[keep[i]].mean(),'miss oracle panel MAE',np.abs(Z-gt)[~keep[i]].mean(),'latent affine residual',np.round(np.quantile((logit(G['prob_query'][i])-gt@G['beta']),[0,.1,.5,.9,1]),3),'strata logits', [np.round(np.unique(np.round((logit(G['prob_query'][i])-gt@G['beta'])[s[i]==t],6)),3).tolist()[:8] for t in range(3)])
PY
OPENBLAS_NUM_THREADS=1 python3 /tmp/inspect_visco.py
```
- **Container Exit Code**: `0` (`1173 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
MAE 0.01135132610777608 latent affine residual [-0.9  -0.9  -0.9   0.85  0.85] strata logits [[-0.9], [0.0], [0.85]]
case 1 taus [0.55, 2.6] beta [ 2.15 -1.95  1.8 ]
train condition [ 6.44 10.6  12.7  13.67] order residual median [4.0000e-04 9.0000e-04 1.5000e-03 2.0000e-03 2.5000e-03 3.1000e-03
 3.6000e-03 4.2000e-03 4.8000e-03 5.4000e-03 5.8000e-03 6.5000e-03
 7.1000e-03 7.7000e-03 8.5000e-03 9.2000e-03 9.7000e-03 1.0500e-02
 1.1400e-02 1.2000e-02 1.2900e-02 1.3800e-02 1.4600e-02 1.5600e-02
 1.6800e-02 1.8000e-02 1.8900e-02 2.0800e-02 2.2400e-02 2.4300e-02
 2.7400e-02 3.2400e-02 5.3977e+00 6.1880e+00 7.1498e+00 7.9202e+00] mask [0.04861111 0.05902778 0.05208333]
center 0 panel [3.969 1.273] strata n/p [(42, 0.548), (19, 0.684), (11, 0.455)] Z cell means [[0.749 0.833 0.584]
 [0.621 0.93  0.691]
 [0.593 0.834 0.585]]
center 1 panel [3.73  0.698] strata n/p [(36, 0.472), (24, 0.667), (12, 1.0)] Z cell means [[0.514 0.731 0.643]
 [0.323 0.238 1.181]
 [1.24  0.172 1.022]]
center 2 panel [-3.896  1.28 ] strata n/p [(33, 0.212), (27, 0.556), (12, 0.5)] Z cell means [[-0.565 -0.409 -0.605]
 [-0.572 -0.648 -0.002]
 [-0.773 -0.825 -0.463]]
center 3 panel [-3.717  0.81 ] strata n/p [(39, 0.282), (22, 0.364), (11, 0.636)] Z cell means [[-0.475 -0.305 -0.194]
 [-0.746 -0.475 -0.617]
 [-0.407 -0.655 -0.578]]
query condition [ 6.12 10.64 12.87 14.19] order residual median [4.0000e-04 1.0000e-03 1.5000e-03 2.0000e-03 2.6000e-03 3.2000e-03
 3.7000e-03 4.3000e-03 4.8000e-03 5.5000e-03 6.0000e-03 6.6000e-03
 7.1000e-03 7.9000e-03 8.5000e-03 9.1000e-03 9.8000e-03 1.0400e-02
 1.1200e-02 1.1900e-02 1.2800e-02 1.3700e-02 1.4800e-02 1.5600e-02
 1.6800e-02 1.7900e-02 1.9000e-02 2.0700e-02 2.2300e-02 2.5100e-02
 2.7800e-02 3.1800e-02 5.3937e+00 6.1850e+00 7.1309e+00 7.8350e+00] mask [0.09444444 0.07222222 0.03888889]
center 4 panel [-3.545  1.515] strata n/p [(52, 0.423), (26, 0.423), (12, 0.583)] Z cell means [[ 0.576  0.635  0.549]
 [ 0.198  0.728 -0.063]
 [ 0.138  0.062  0.119]]
  observed MAE 0.025497947515414475 miss oracle panel MAE 0.03374879864479603 latent affine residual [-0.9  -0.9  -0.9   0.85  0.85] strata logits [[-0.9], [0.0], [0.85]]
center 5 panel [3.89  0.777] strata n/p [(50, 0.38), (22, 0.364), (18, 0.5)] Z cell means [[-0.097 -0.395 -0.35 ]
 [-0.635 -0.019 -0.407]
 [-0.553 -0.513 -0.412]]
  observed MAE 0.03473312258657027 miss oracle panel MAE 0.035655380873113984 latent affine residual [-0.9  -0.9  -0.9   0.85  0.85] strata logits [[-0.9], [-0.0], [0.85]]
```

### Turn 3 — `execute_in_remote_docker` (`61088 ms` LLM)
- **Rationale**: The observed residual gap supports consensus refitting. Score modulus error separately for observed and missing channels, then use held-out centers to test whether alignment and cell weighting improve transportable risk rather than pooled performance alone.
- **Alternatives Weighed**: `Center-specific intercepts or center-centering the aligned moduli` (rejected: The public truth shows a shared query logit with stratum offsets; removing center means after panel alignment would also remove real differences in specimen composition.)

```bash
cat > /holdout_task/viscoelastic_panel_calibration.py <<'PY'
"""Recover viscoelastic moduli and calibrate risk on the control-panel scale."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(trace, taus):
    past = np.asarray(trace['past_strain_history'], dtype=float)
    strain = np.asarray(trace['strain_u'], dtype=float)
    time = np.concatenate((trace['past_timestamps'], trace['timestamps']))
    inputs = np.concatenate((past, strain))
    memory = np.empty((inputs.size, taus.size))
    memory[0] = inputs[0]
    # This includes the bridge from the last pre-window sample to the first observation.
    for k, alpha in enumerate(np.exp(-np.diff(time)[:, None] / taus), 1):
        memory[k] = alpha * memory[k - 1] + (1.0 - alpha) * inputs[k]
    return np.column_stack((np.ones(strain.size), strain, memory[past.size:]))


def _extract(block, taus, subsets):
    raw = np.empty((len(block['traces']), 3))
    for i, trace in enumerate(block['traces']):
        design = _design(trace, taus)
        stress = np.asarray(trace['stress_obs'], dtype=float)
        candidates = np.einsum('ijk,ik->ij', np.linalg.pinv(design[subsets]), stress[subsets])
        error = np.abs(stress[:, None] - design @ candidates.T)
        residual = error[:, np.argmin(np.median(error, axis=0))]
        # Identify a consensus before refitting; large glitches cannot exert leverage.
        inlier = residual <= max(0.1, 6.0 * np.median(residual))
        raw[i] = np.linalg.lstsq(design[inlier], stress[inlier], rcond=None)[0][1:]
    raw[np.asarray(block['missing_channel_mask'], dtype=bool)] = np.nan
    return raw


def _align(raw, block):
    center = np.asarray(block['center_id'])
    panels = np.asarray(block['controls'], dtype=float)
    z = np.empty_like(raw)
    for c in np.unique(center):
        rows = center == c
        panel = panels[rows]
        panel = panel[:, np.sum(np.isfinite(panel), axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        z[rows] = (raw[rows] - offset) / gain
    return z


def _cell_index(center, stratum):
    _, index, counts = np.unique(np.column_stack((center, stratum)), axis=0,
                                 return_inverse=True, return_counts=True)
    return index, counts


def _imputation(z, cells, n_cells):
    # Average cell means, not individual specimens. An unobserved cell/channel
    # contributes no estimate for that channel.
    means = np.full((n_cells, z.shape[1]), np.nan)
    for c in range(n_cells):
        group = z[cells == c]
        counts = np.isfinite(group).sum(axis=0)
        np.divide(np.nansum(group, axis=0), counts, out=means[c], where=counts > 0)
    return np.nanmean(means, axis=0)


def _risk_design(z, stratum):
    s = np.asarray(stratum)
    return np.column_stack((z, np.ones(len(s)), s == 1, s == 2))


def _fit_risk(x, y, weights, ridge=1.0):
    penalty = np.array([ridge, ridge, ridge, 0.0, 0.0, 0.0])

    def objective(b):
        logits = x @ b
        loss = weights @ np.logaddexp(0, (1 - 2 * y) * logits)
        return loss + 0.5 * np.dot(penalty * b, b), x.T @ (weights * (expit(logits) - y)) + penalty * b

    return minimize(objective, np.zeros(x.shape[1]), jac=True, method='BFGS').x


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Estimate query moduli and failure probabilities without query labels."""
    taus = np.asarray(cohort['taus'], dtype=float)
    train, query = cohort['train'], cohort['query']
    rng = np.random.default_rng(518)
    subsets = np.array([rng.choice(36, 4, replace=False) for _ in range(96)])
    z_train = _align(_extract(train, taus, subsets), train)
    z_query = _align(_extract(query, taus, subsets), query)
    cells, counts = _cell_index(train['center_id'], train['stratum'])
    means = _imputation(z_train, cells, len(counts))
    z_train = np.where(np.isnan(z_train), means, z_train)
    z_query = np.where(np.isnan(z_query), means, z_query)
    weights = (len(z_train) / len(counts)) / counts[cells]
    fitted = _fit_risk(_risk_design(z_train, train['stratum']), np.asarray(train['y']), weights)
    probabilities = expit(_risk_design(z_query, query['stratum']) @ fitted)
    eps = np.finfo(float).eps
    return {'aligned_query_moduli': z_query,
            'query_probabilities': np.clip(probabilities, eps, 1.0 - eps),
            'modulus_weights': fitted[:3]}
PY
cat > /tmp/eval_visco.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

spec=importlib.util.spec_from_file_location('calibration','/holdout_task/viscoelastic_panel_calibration.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_viscoelastic_cohort.pkl','rb') as f: data=pickle.load(f)

def score(y,p):
    if len(np.unique(y))<2:
        return np.array([np.nan, np.mean((y-p)**2), np.nan])
    return np.array([roc_auc_score(y,p), np.mean((y-p)**2), balanced_accuracy_score(y,p>=.5)])

for case,d in enumerate(data):
    cohort, gt=d['cohort'],d['gt']
    t,q=cohort['train'],cohort['query']
    out=m.calibrate_viscoelastic_cohort(cohort)
    p=out['query_probabilities']; z=out['aligned_query_moduli']; y=gt['y_query']; ctr=q['center_id']; mask=q['missing_channel_mask']; err=np.abs(z-gt['Z_query'])
    print('public',case,'MAE',err.mean(),'masked',err[mask].mean(),'unmasked',err[~mask].mean(),'by channel',err.mean(0),'weights',out['modulus_weights'],'pooled',score(y,p))
    for c in np.unique(ctr):
        rows=ctr==c
        print(' query center',c,'MAE',err[rows].mean(),'metrics',score(y[rows],p[rows]),'stratum metrics',[(s,np.round(score(y[rows&(q['stratum']==s)],p[rows&(q['stratum']==s)]),4)) for s in range(3)])
    rng=np.random.default_rng(518); subsets=np.array([rng.choice(36,4,replace=False) for _ in range(96)])
    raw=m._extract(t,np.asarray(cohort['taus']),subsets); aligned=m._align(raw,t); c=t['center_id']; s=t['stratum']; y=t['y']; scores={}
    for held in np.unique(c):
        test=c==held; train=~test; idx,counts=m._cell_index(c[train],s[train]); means=m._imputation(aligned[train],idx,len(counts)); arr=np.where(np.isnan(aligned),means,aligned); x=m._risk_design(arr,s)
        for power in (0,.5,1):
            w=counts[idx]**-power; w*=sum(train)/w.sum()
            for ridge in (.1,1,3):
                b=m._fit_risk(x[train],y[train],w,ridge)
                scores.setdefault((power,ridge),[]).append(score(y[test],expit(x[test]@b)))
        rawmean=m._imputation(raw[train],idx,len(counts)); xr=m._risk_design(np.where(np.isnan(raw),rawmean,raw),s); b=m._fit_risk(xr[train],y[train],np.ones(sum(train)))
        scores.setdefault('unaligned',[]).append(score(y[test],expit(xr[test]@b)))
    for variant, sc in scores.items():
        a=np.array(sc); print(' LOCO',variant,'mean',np.round(a.mean(0),4),'worst',np.round([a[:,0].min(),a[:,1].max(),a[:,2].min()],4))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/eval_visco.py
```
- **Container Exit Code**: `1` (`2048 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_panel_calibration.py']`
- **Stdout (tail)**:
```text
public 0 MAE 0.07218279630490507 masked 0.8098879169818509 unmasked 0.03486697308000118 by channel [0.08011627 0.06005166 0.07638045] weights [ 1.92125013 -1.55458075  1.57725082] pooled [0.94581538 0.09300777 0.88523306]
 query center 4 MAE 0.07507229491352119 metrics [0.95750988 0.08821867 0.89970356] stratum metrics [(0, array([0.942 , 0.1012, 0.8817])), (1, array([0.9556, 0.0794, 0.9111])), (2, array([1.    , 0.0515, 0.9375]))]
 query center 5 MAE 0.06929329769628893 metrics [0.92553846 0.09779688 0.86615385] stratum metrics [(0, array([0.9559, 0.0821, 0.8868])), (1, array([0.9625, 0.0701, 0.8688])), (2, array([0.6296, 0.221 , 0.7778]))]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a8fb6a8dcd30a3af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=False | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a8fb6a8d (directory).
```
- **Stderr (tail)**:
```text
Traceback (most recent call last):
  File "/tmp/eval_visco.py", line 31, in <module>
    w=counts[idx]**-power; w*=sum(train)/w.sum()
                           ^^^^^^^^^^^^^^^^^^^^^
numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```

### Turn 4 — `execute_in_remote_docker` (`15721 ms` LLM)
- **Rationale**: Fix the validation script's integer weight cast without editing the candidate. Complete the second public-cohort evaluation and compare aligned and unaligned fits on held-out centers; if alignment fails to transport, investigate weighting and scanner calibration before changing the estimator.
- **Alternatives Weighed**: `Change the risk model based on the weakest query center–stratum slice` (rejected: That small slice is below the center-level target but does not establish a transport failure; the second cohort and held-out-center comparisons are still missing.)

```bash
sed -i 's/w=counts\[idx\]\*\*-power/w=counts[idx].astype(float)**-power/' /tmp/eval_visco.py
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/eval_visco.py
```
- **Container Exit Code**: `0` (`1956 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nter 4 MAE 0.07507229491352119 metrics [0.95750988 0.08821867 0.89970356] stratum metrics [(0, array([0.942 , 0.1012, 0.8817])), (1, array([0.9556, 0.0794, 0.9111])), (2, array([1.    , 0.0515, 0.9375]))]
 query center 5 MAE 0.06929329769628893 metrics [0.92553846 0.09779688 0.86615385] stratum metrics [(0, array([0.9559, 0.0821, 0.8868])), (1, array([0.9625, 0.0701, 0.8688])), (2, array([0.6296, 0.221 , 0.7778]))]
 LOCO (0, 0.1) mean [0.9268 0.1096 0.8391] worst [0.9115 0.1179 0.8174]
 LOCO (0, 1) mean [0.9262 0.1092 0.839 ] worst [0.9091 0.1178 0.8174]
 LOCO (0, 3) mean [0.926  0.1104 0.8455] worst [0.9099 0.1192 0.8174]
 LOCO (0.5, 0.1) mean [0.9266 0.1095 0.8391] worst [0.9123 0.1179 0.8174]
 LOCO (0.5, 1) mean [0.9262 0.109  0.839 ] worst [0.9123 0.1176 0.8174]
 LOCO (0.5, 3) mean [0.9258 0.1101 0.8422] worst [0.9107 0.1188 0.8174]
 LOCO (1, 0.1) mean [0.9249 0.1095 0.8391] worst [0.9091 0.1181 0.8174]
 LOCO (1, 1) mean [0.9263 0.1089 0.839 ] worst [0.9115 0.1175 0.8174]
 LOCO (1, 3) mean [0.9263 0.1098 0.8422] worst [0.9107 0.1185 0.8174]
 LOCO unaligned mean [0.7541 0.2105 0.6509] worst [0.7257 0.2331 0.5341]
public 1 MAE 0.08707594057658555 masked 0.8613047619776036 unmasked 0.030124715145496708 by channel [0.08516438 0.1187041  0.05735934] weights [ 1.54499556 -1.24894509  1.31471048] pooled [0.91156377 0.11601847 0.83577935]
 query center 4 MAE 0.08603833943057193 metrics [0.8845     0.13654804 0.79      ] stratum metrics [(0, array([0.8985, 0.1306, 0.7924])), (1, array([0.8606, 0.1457, 0.7636])), (2, array([0.8286, 0.1425, 0.8286]))]
 query center 5 MAE 0.08811354172259916 metrics [0.92798354 0.09548891 0.87962963] stratum metrics [(0, array([0.8795, 0.1256, 0.82  ])), (1, array([0.9554, 0.079 , 0.9018])), (2, array([1.    , 0.0319, 1.    ]))]
 LOCO (0, 0.1) mean [0.9102 0.1188 0.8223] worst [0.8679 0.1446 0.7592]
 LOCO (0, 1) mean [0.9108 0.1182 0.8223] worst [0.8687 0.1424 0.7592]
 LOCO (0, 3) mean [0.9112 0.1186 0.8213] worst [0.8704 0.1407 0.7676]
 LOCO (0.5, 0.1) mean [0.9104 0.1186 0.8223] worst [0.8696 0.1447 0.7592]
 LOCO (0.5, 1) mean [0.9104 0.1181 0.8223] worst [0.8704 0.1423 0.7592]
 LOCO (0.5, 3) mean [0.9114 0.1185 0.8228] worst [0.8704 0.1403 0.7676]
 LOCO (1, 0.1) mean [0.9104 0.1186 0.8189] worst [0.8696 0.1451 0.74  ]
 LOCO (1, 1) mean [0.9106 0.118  0.821 ] worst [0.8696 0.1424 0.7483]
 LOCO (1, 3) mean [0.9116 0.1186 0.8133] worst [0.8696 0.1401 0.7483]
 LOCO unaligned mean [0.7584 0.1979 0.6997] worst [0.7333 0.208  0.6575]
```

### Turn 5 — `execute_in_remote_docker` (`47200 ms` LLM)
- **Rationale**: Verify the exact artifact in an isolated process. The independent hereditary-state test can expose a recurrence or glitch-handling error; the public contract assertions can expose interface or target failures. Compare the narrowest AUROC margin against oracle risk on the same imputed features to distinguish a fitting bottleneck from information lost through masking.
- **Alternatives Weighed**: `Retune regularization or cell weights on public query outcomes` (rejected: The held-out-center comparisons show only small, inconsistent gains, and selecting on the query outcomes would turn validation into tuning.)

```bash
OPENBLAS_NUM_THREADS=1 python3 -I - <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

path = '/holdout_task/viscoelastic_panel_calibration.py'
spec = importlib.util.spec_from_file_location('viscoelastic_panel_calibration', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as f:
    cases = pickle.load(f)

# Exact reconstruction with a nonconstant prehistory, unequal gaps, and four large glitches.
rng = np.random.default_rng(237)
for i in range(30):
    taus = np.array([0.2 + i / 100, 1.7 + i / 40])
    n = 3 + i % 12
    time = np.cumsum(rng.uniform(0.04, 0.9, n + 36))
    time[n:] += (i % 5) * 0.3
    u = rng.normal(size=n + 36)
    expected = rng.normal(size=3) + (4 if i % 2 else -4)
    memory = []
    for k in range(n, len(time)):
        a = np.exp(-np.diff(time[:k + 1])[:, None] / taus)
        memory.append(np.exp(-(time[k] - time[0]) / taus) * u[0] + np.sum((1 - a) * np.exp(-(time[k] - time[1:k + 1])[:, None] / taus) * u[1:k + 1, None], axis=0))
    stress = 0.8 + np.column_stack((u[n:], memory)) @ expected
    stress[rng.choice(36, 4, replace=False)] += rng.uniform(4.5, 9.0, 4) * rng.choice([-1, 1], 4)
    selected = np.random.default_rng(518)
    subsets = np.array([selected.choice(36, 4, replace=False) for _ in range(96)])
    trace = dict(past_timestamps=time[:n], past_strain_history=u[:n], timestamps=time[n:], strain_u=u[n:], stress_obs=stress)
    mask = np.arange(3)[None, :] == i % 3
    extracted = mod._extract(dict(traces=[trace], missing_channel_mask=mask), taus, subsets)
    np.testing.assert_array_equal(np.isnan(extracted), mask)
    np.testing.assert_allclose(extracted[~mask], expected[~mask[0]], atol=1e-9, rtol=0)
print('Exact hereditary-state/glitch/mask stress: 30/30')

worst = [0.0, 1.0, 0.0]
for j, case in enumerate(cases):
    cohort, gt = case['cohort'], case['gt']
    out = mod.calibrate_viscoelastic_cohort(cohort)
    assert set(out) == {'aligned_query_moduli', 'query_probabilities', 'modulus_weights'}
    q = cohort['query']
    y, latent, b = gt['y_query'], gt['Z_query'], gt['beta']
    p, z, fit = (out[k] for k in ('query_probabilities', 'aligned_query_moduli', 'modulus_weights'))
    for a, shape in ((p, (len(y),)), (z, (len(y), 3)), (fit, (3,))):
        assert isinstance(a, np.ndarray) and a.shape == shape and a.dtype.kind == 'f' and np.all(np.isfinite(a))
    assert np.all((p > 0) & (p < 1))
    mask = q['missing_channel_mask']
    for d in range(3):
        assert np.ptp(z[mask[:, d], d]) == 0.0 if mask[:, d].any() else True
    intercept = logit(gt['prob_query']) - latent @ b
    # Diagnostic only: oracle weights/intercepts on the same output features isolate model-fit headroom.
    oracle_same = expit(intercept + z @ b)
    for c in np.unique(q['center_id']):
        rows = q['center_id'] == c
        mae = np.abs(z[rows] - latent[rows]).mean()
        auc = roc_auc_score(y[rows], p[rows])
        brier = np.mean((p[rows] - y[rows]) ** 2)
        ref_auc = roc_auc_score(y[rows], oracle_same[rows])
        full_auc = roc_auc_score(y[rows], gt['prob_query'][rows])
        print(f'cohort {j}, center {c}: MAE {mae:.5f}, AUROC {auc:.5f}, Brier {brier:.5f}; oracle risk on same moduli AUROC {ref_auc:.5f}, oracle with unmasked truth {full_auc:.5f}')
        assert mae <= 0.22 and auc >= 0.88 and brier <= 0.160
        worst = [max(worst[0], mae), min(worst[1], auc), max(worst[2], brier)]
print('Worst public query-center MAE, AUROC, Brier:', worst)
PY
```
- **Container Exit Code**: `0` (`1563 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
Exact hereditary-state/glitch/mask stress: 30/30
cohort 0, center 4: MAE 0.07507, AUROC 0.95751, Brier 0.08822; oracle risk on same moduli AUROC 0.95800, oracle with unmasked truth 0.96047
cohort 0, center 5: MAE 0.06929, AUROC 0.92554, Brier 0.09780; oracle risk on same moduli AUROC 0.92492, oracle with unmasked truth 0.94892
cohort 1, center 4: MAE 0.08604, AUROC 0.88450, Brier 0.13655; oracle risk on same moduli AUROC 0.87600, oracle with unmasked truth 0.90900
cohort 1, center 5: MAE 0.08811, AUROC 0.92798, Brier 0.09549; oracle risk on same moduli AUROC 0.92798, oracle with unmasked truth 0.94496
Worst public query-center MAE, AUROC, Brier: [0.08811354172259916, 0.8845, 0.13654803613976527]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a8fb6a8dcd30a3af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a8fb6a8d (directory).
```

### Turn 6 — `submit_completed_task` (`15422 ms` LLM)
- **Rationale**: The exact submitted artifact passed isolated-process verification, all public query-center thresholds, and independent tests of prehistory, glitch handling, and masking. Further tuning on public query outcomes would compromise validation.
- **Alternatives Weighed**: `Retune the risk model to widen the narrowest public AUROC margin` (rejected: The held-out training-center comparisons showed no consistent improvement, and oracle weights on the same imputed moduli did not improve the weakest public query center.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/viscoelastic_panel_calibration.py': 'FILE:4401'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. All four public query centers met the targets: worst MAE 0.0881, lowest AUROC 0.8845, and worst Brier score 0.1365. All 30 hereditary reconstruction stress tests passed. In leave-one-center-out validation on the training data, the lowest AUROC was 0.8696.`

### Turn 7 — `execute_in_remote_docker` (`33690 ms` LLM)
- **Rationale**: Audit the known candidate without editing it. Check whether residuals outside the glitch tail warrant revisiting extraction, and compare class-specific recalls using identical cell weights to isolate the effect of scanner alignment. If aligned recall collapses in a held-out center, inspect whether the issue is ranking or threshold calibration before proposing a change.
- **Alternatives Weighed**: `Add nonlinear terms or unrelated relay and drift operators` (rejected: The specified stress law is affine in strain and the two exact memory states; the residual audit can show whether that basis is leaving errors outside the glitch tail.)

```bash
cat > /tmp/audit_visco.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('calibration', '/holdout_task/viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as f:
    cases = pickle.load(f)
rng = np.random.default_rng(518)
subsets = np.array([rng.choice(36, 4, replace=False) for _ in range(96)])
for k, case in enumerate(cases):
    cohort = case['cohort']
    taus = np.asarray(cohort['taus'])
    t = cohort['train']
    c, s, y = (t[key] for key in ('center_id', 'stratum', 'y'))
    raw = m._extract(t, taus, subsets)
    aligned = m._align(raw, t)
    # The exact affine basis predicts 32 clean observations. A large residual
    # before the glitch tail would motivate investigating the assumed basis or fit.
    order = []
    for block in (t, cohort['query']):
        restored = m._extract(dict(block, missing_channel_mask=np.zeros_like(block['missing_channel_mask'])), taus, subsets)
        for trace, b in zip(block['traces'], restored):
            r = trace['stress_obs'] - m._design(trace, taus)[:, 1:] @ b
            r = np.sort(np.abs(r - np.median(r)))
            order.append(r[[30, 31, 32]])
    print('cohort', k, 'residual orders 31..33 median/max', np.round(np.median(order, axis=0), 4), np.round(np.max(order, axis=0), 4))
    results = {}
    for h in np.unique(c):
        test = c == h
        train = ~test
        idx, counts = m._cell_index(c[train], s[train])
        w = train.sum() / (len(counts) * counts[idx])
        for name, z in [('raw', raw), ('panel', aligned)]:
            mean = m._imputation(z[train], idx, len(counts))
            x = m._risk_design(np.where(np.isnan(z), mean, z), s)
            p = expit(x[test] @ m._fit_risk(x[train], y[train], w))
            yh, sh = y[test], s[test]
            recalls = [np.mean((p[yh == cl] >= .5) == cl) for cl in (0, 1)]
            cells = []
            for level in np.unique(sh):
                r = sh == level
                if np.unique(yh[r]).size == 2:
                    cells.append((int(level), int(r.sum()), *[np.mean((p[r & (yh == cl)] >= .5) == cl) for cl in (0, 1)]))
            results.setdefault(name, []).append((int(h), roc_auc_score(yh, p), np.mean((yh-p)**2), *recalls))
            print(' ', name, 'held center', h, 'stratum (s,n,recall0,recall1)', [(a,b,round(d,3),round(e,3)) for a,b,d,e in cells])
    for name, values in results.items():
        print(' ', name, '(center,AUC,Brier,recall0,recall1)', np.round(values, 4).tolist())
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/audit_visco.py
```
- **Container Exit Code**: `0` (`2685 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
1.0)]
  raw held center 2 stratum (s,n,recall0,recall1) [(0, 35, 0.917, 0.364), (1, 18, 0.833, 0.5), (2, 19, 1.0, 0.75)]
  panel held center 2 stratum (s,n,recall0,recall1) [(0, 35, 0.917, 0.636), (1, 18, 0.833, 0.833), (2, 19, 1.0, 0.75)]
  raw held center 3 stratum (s,n,recall0,recall1) [(0, 35, 0.905, 0.286), (1, 24, 0.812, 0.0), (2, 13, 0.714, 0.5)]
  panel held center 3 stratum (s,n,recall0,recall1) [(0, 35, 0.905, 0.857), (1, 24, 0.938, 0.375), (2, 13, 1.0, 0.833)]
  raw (center,AUC,Brier,recall0,recall1) [[0.0, 0.7407, 0.2166, 0.7941, 0.6579], [1.0, 0.7257, 0.2344, 0.5385, 0.7879], [2.0, 0.8162, 0.1689, 0.9149, 0.52], [3.0, 0.7476, 0.214, 0.8409, 0.25]]
  panel (center,AUC,Brier,recall0,recall1) [[0.0, 0.942, 0.1024, 0.9118, 0.7895], [1.0, 0.9394, 0.1066, 0.8205, 0.9091], [2.0, 0.9115, 0.1089, 0.9149, 0.72], [3.0, 0.9123, 0.1175, 0.9318, 0.7143]]
cohort 1 residual orders 31..33 median/max [0.0282 0.033  5.3967] [0.0466 0.0643 7.6794]
  raw held center 0 stratum (s,n,recall0,recall1) [(0, 42, 0.684, 0.739), (1, 19, 0.5, 0.923), (2, 11, 0.167, 1.0)]
  panel held center 0 stratum (s,n,recall0,recall1) [(0, 42, 0.842, 0.826), (1, 19, 1.0, 0.846), (2, 11, 0.5, 1.0)]
  raw held center 1 stratum (s,n,recall0,recall1) [(0, 36, 0.789, 0.765), (1, 24, 0.125, 0.938)]
  panel held center 1 stratum (s,n,recall0,recall1) [(0, 36, 1.0, 0.824), (1, 24, 0.75, 1.0)]
  raw held center 2 stratum (s,n,recall0,recall1) [(0, 33, 0.962, 0.143), (1, 27, 0.75, 0.4), (2, 12, 1.0, 0.667)]
  panel held center 2 stratum (s,n,recall0,recall1) [(0, 33, 0.962, 0.429), (1, 27, 0.75, 0.733), (2, 12, 1.0, 0.667)]
  raw held center 3 stratum (s,n,recall0,recall1) [(0, 39, 0.964, 0.273), (1, 22, 0.571, 0.625), (2, 11, 0.5, 0.714)]
  panel held center 3 stratum (s,n,recall0,recall1) [(0, 39, 0.929, 0.818), (1, 22, 0.714, 0.5), (2, 11, 0.25, 0.714)]
  raw (center,AUC,Brier,recall0,recall1) [[0.0, 0.7262, 0.2109, 0.5484, 0.8293], [1.0, 0.7654, 0.1963, 0.5926, 0.8444], [2.0, 0.7857, 0.1831, 0.9091, 0.3929], [3.0, 0.7508, 0.2079, 0.8043, 0.5]]
  panel (center,AUC,Brier,recall0,recall1) [[0.0, 0.9079, 0.1257, 0.8065, 0.8537], [1.0, 0.9712, 0.0768, 0.9259, 0.9333], [2.0, 0.8937, 0.1273, 0.9091, 0.6429], [3.0, 0.8696, 0.1424, 0.8043, 0.6923]]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a8fb6a8dcd30a3af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a8fb6a8d (directory).
```

### Turn 8 — `submit_completed_task` (`17407 ms` LLM)
- **Rationale**: The focused audit confirmed that panel alignment eliminated the zero class recalls seen in some unaligned held-out center–stratum cells. The unchanged artifact passed isolated-process verification and all public query-center thresholds.
- **Alternatives Weighed**: `Add nonlinear terms to the stress model` (rejected: Across both public cohorts, the 32nd ordered absolute residual stayed below 0.065, while the median 33rd residual exceeded 5.39, supporting the specified affine basis and a separate glitch tail.); `Retune the risk model to widen the narrowest public AUROC margin` (rejected: Training-center validation showed no consistent gain from alternate weights or penalties, and oracle risk on the same imputed moduli scored lower in the weakest public query center.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`20` turns remaining; artifact status={'/holdout_task/viscoelastic_panel_calibration.py': 'FILE:4401'}): Do NOT lock prematurely with `20` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. All four public query-center evaluations met the targets: worst MAE 0.0881, lowest AUROC 0.8845, and worst Brier score 0.1365. All 30 reconstruction stress tests passed. The lowest AUROC in training leave-one-center-out validation was 0.8696.`

### Turn 9 — `execute_in_remote_docker` (`31888 ms` LLM)
- **Rationale**: Masking accounts for much of the narrowest public AUROC margin, but the prior oracle comparison did not test whether fitting on imputed training inputs contributes to the problem. Use training-only leave-one-center-out comparisons to test that possibility. Keep the verified artifact unless the complete-case fit materially improves the weakest held-center ranking without worsening masked-case calibration.
- **Alternatives Weighed**: `Repeat the verified extraction and scanner-confounding checks` (rejected: The residual gap already supports the specified stress basis, and the raw-versus-aligned audit showed that panel alignment removed observed zero-class recalls.); `Replace the required cell-balanced query imputation` (rejected: The contract fixes the imputation rule; this comparison keeps it identical for both estimators at prediction time.)

```bash
cat > /tmp/visco_mask_loco.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('calibration', '/holdout_task/viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as f:
    cases = pickle.load(f)
rng = np.random.default_rng(518)
subsets = np.array([rng.choice(36, 4, replace=False) for _ in range(96)])
for j, case in enumerate(cases):
    cohort = case['cohort']
    t = cohort['train']
    c, s, y = (t[key] for key in ('center_id', 'stratum', 'y'))
    z = m._align(m._extract(t, np.asarray(cohort['taus']), subsets), t)
    complete = np.isfinite(z).all(axis=1)
    metrics = {'imputed': [], 'complete fit': []}
    masked_losses = {key: [] for key in metrics}
    for held in np.unique(c):
        test = c == held
        train = ~test
        idx, counts = m._cell_index(c[train], s[train])
        mean = m._imputation(z[train], idx, len(counts))
        x = m._risk_design(np.where(np.isnan(z), mean, z), s)
        for name, eligible in [('imputed', train), ('complete fit', train & complete)]:
            idx, counts = m._cell_index(c[eligible], s[eligible])
            if len(counts) != 9:
                print(j, held, name, 'missing complete-case training cells', len(counts))
            w = eligible.sum() / (len(counts) * counts[idx])
            b = m._fit_risk(x[eligible], y[eligible], w)
            p = expit(x[test] @ b)
            error = (p - y[test]) ** 2
            masked_losses[name].extend(error[~complete[test]])
            metrics[name].append((roc_auc_score(y[test], p), error.mean()))
    for name, rows in metrics.items():
        a = np.asarray(rows)
        print(f'cohort {j} {name}: held-center AUC/Brier={np.round(a, 5).tolist()}, worst={a[:, 0].min():.5f}/{a[:, 1].max():.5f}, masked Brier={np.mean(masked_losses[name]):.5f} (n={len(masked_losses[name])})')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/visco_mask_loco.py
```
- **Container Exit Code**: `0` (`1249 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cohort 0 imputed: held-center AUC/Brier=[[0.94195, 0.10241], [0.93939, 0.10664], [0.91149, 0.10888], [0.91234, 0.11755]], worst=0.91149/0.11755, masked Brier=0.17683 (n=49)
cohort 0 complete fit: held-center AUC/Brier=[[0.93498, 0.10453], [0.93706, 0.11015], [0.91234, 0.1121], [0.91396, 0.11454]], worst=0.91234/0.11454, masked Brier=0.19859 (n=49)
cohort 1 imputed: held-center AUC/Brier=[[0.90795, 0.12572], [0.97119, 0.07682], [0.89367, 0.12727], [0.86957, 0.14237]], worst=0.86957/0.14237, masked Brier=0.21015 (n=43)
cohort 1 complete fit: held-center AUC/Brier=[[0.90008, 0.13811], [0.9786, 0.066], [0.89123, 0.12823], [0.86957, 0.14273]], worst=0.86957/0.14273, masked Brier=0.21860 (n=43)

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=a8fb6a8dcd30a3af (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_a8fb6a8d (directory).
```

### Turn 10 — `submit_completed_task` (`15605 ms` LLM)
- **Rationale**: The measured residual gap supports the specified hereditary stress basis, and panel alignment eliminated the zero class recalls observed in some unaligned held-out center–stratum cells. The latest comparison did not improve the weakest ranking slice, so retain the candidate that passed isolated-process verification.
- **Alternatives Weighed**: `Fit risk on complete training cases only` (rejected: It left the lowest held-out-center AUROC unchanged at 0.8696 and worsened Brier scores on masked specimens in both cohorts.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. All four public query centers met the targets: worst MAE 0.0881, lowest AUROC 0.8845, and worst Brier score 0.1365. All 30 reconstruction stress tests passed. The lowest training leave-one-center-out AUROC was 0.8696.`

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
