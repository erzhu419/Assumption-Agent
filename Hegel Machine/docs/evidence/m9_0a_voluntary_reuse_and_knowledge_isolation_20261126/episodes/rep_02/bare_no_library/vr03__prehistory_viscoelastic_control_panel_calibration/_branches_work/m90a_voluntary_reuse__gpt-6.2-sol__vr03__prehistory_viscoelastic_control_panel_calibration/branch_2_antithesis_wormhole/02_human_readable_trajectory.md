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
| **Total Tokens (Prompt / Cached / Out)** | `246,340` (`236,073` / `51,584` / `10,267`) |
| **Total Episode Wall Time (s)** | `337.48s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`20485 ms` LLM)
- **Rationale**: Inspect the public cohort using summary output only; field nesting determines the input adapter, and label availability determines which splits can support out-of-fold validation.
- **Alternatives Weighed**: `Implement against an assumed pickle schema` (rejected: A compact inspection will identify actual field names, available validation labels, and supported numerical dependencies before implementation.)

```bash
cd /holdout_task && python3 -I - <<'PY'
import pickle
from collections import Counter
import numpy as np
with open('public_viscoelastic_cohort.pkl','rb') as f:
    data = pickle.load(f)
def inspect(x, name='root', depth=0):
    if isinstance(x, dict):
        print(name, 'keys', list(x)[:25])
        if depth < 3:
            for k,v in list(x.items())[:20]: inspect(v, f'{name}.{k}', depth+1)
    elif isinstance(x, (list,tuple)):
        print(name, type(x).__name__, len(x))
        if x and depth < 3: inspect(x[0], name+'[0]', depth+1)
    elif isinstance(x,np.ndarray):
        details = ''
        if x.dtype.kind in 'biuf' and x.size:
            vals = x[np.isfinite(x)]; details = f' finite={vals.size}/{x.size}'
            if vals.size: details += f' range=({vals.min():.5g},{vals.max():.5g})'
            if x.ndim == 1 and vals.size and np.unique(vals).size <= 12: details += f' counts={dict(Counter(vals.tolist()))}'
        print(name, x.shape, x.dtype, details)
    elif isinstance(x,(int,float,str,type(None))): print(name, repr(x)[:90])
    else: print(name,type(x).__name__)
inspect(data)
try:
    import scipy
    print('scipy', scipy.__version__)
except ImportError:
    print('scipy unavailable')
PY
```
- **Container Exit Code**: `0` (`664 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
root list 2
root[0] keys ['cohort', 'gt']
root[0].cohort keys ['taus', 'train', 'query']
root[0].cohort.taus list 2
root[0].cohort.train keys ['center_id', 'stratum', 'controls', 'missing_channel_mask', 'traces', 'y']
root[0].cohort.query keys ['center_id', 'stratum', 'controls', 'missing_channel_mask', 'traces']
root[0].gt keys ['Z_query', 'y_query', 'prob_query', 'beta']
root[0].gt.Z_query (180, 3) float64  finite=540/540 range=(-3.1306,2.9809)
root[0].gt.y_query (180,) int64  finite=180/180 range=(0,1) counts={0: 111, 1: 69}
root[0].gt.prob_query (180,) float64  finite=180/180 range=(4.9744e-06,0.99963)
root[0].gt.beta (3,) float64  finite=3/3 range=(-1.95,2.15) counts={2.15: 1, -1.95: 1, 1.8: 1}
scipy 1.13.0
```

### Turn 2 — `execute_in_remote_docker` (`58804 ms` LLM)
- **Rationale**: Implement both stages, then separate alignment error on observed channels from missing-channel error and compare held-out center discrimination and calibration against the public oracle.
- **Alternatives Weighed**: `Fit center identifiers as transferable predictors` (rejected: Training centers do not overlap query centers, and raw center prevalence can reflect the scanner and cohort composition rather than a transferable effect.)

```bash
cat >/holdout_task/viscoelastic_panel_calibration.py <<'PY'
"""Hereditary stress extraction and control-panel risk calibration."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(trace, taus):
    t = np.r_[trace['past_timestamps'], trace['timestamps']]
    u = np.r_[trace['past_strain_history'], trace['strain_u']]
    past = len(trace['past_timestamps'])
    alpha = -np.expm1(-np.diff(t)[:, None] / taus)
    state = np.full(len(taus), u[0])
    x = np.empty((len(t) - past, 1 + len(taus)))
    for k, a in enumerate(alpha, start=1):
        state += a * (u[k] - state)
        if k >= past:
            x[k - past, 0] = u[k]
            x[k - past, 1:] = state
    return np.column_stack((np.ones(len(x)), x))


def _robust_fit(a, y, subsets):
    # Whiten the design for stable minimal-subset fits, even when the
    # different Prony modes have very different observed amplitudes.
    q, r = np.linalg.qr(a, mode='reduced')
    sub = q[subsets]
    good = np.abs(np.linalg.det(sub)) > 1e-10
    idx = subsets[good]
    starts = np.linalg.solve(sub[good], y[idx][..., None])[..., 0]
    starts = np.vstack((starts, q.T @ y))
    trimmed = int(0.72 * len(y))
    resid = y[:, None] - q @ starts.T
    error = np.partition(resid**2, trimmed - 1, axis=0)[:trimmed].sum(axis=0)
    b = starts[error.argmin()]
    for _ in range(5):
        keep = np.argsort(np.abs(y - q @ b))[:trimmed]
        b = np.linalg.lstsq(q[keep], y[keep], rcond=None)[0]
    for _ in range(4):
        resid = y - q @ b
        med = np.median(resid)
        scale = 1.4826 * np.median(np.abs(resid - med))
        keep = np.abs(resid - med) <= max(4.5 * scale, 1e-10)
        b = np.linalg.lstsq(q[keep], y[keep], rcond=None)[0]
    return np.linalg.solve(r, b)[1:]


def _aligned(split, taus):
    center = np.asarray(split['center_id'])
    controls = np.asarray(split['controls'], dtype=float)
    rng = np.random.default_rng(185)
    subsets = np.array([rng.choice(36, 4, replace=False) for _ in range(96)])
    x = np.array([_robust_fit(_design(tr, taus), np.asarray(tr['stress_obs']), subsets)
                  for tr in split['traces']])
    for c in np.unique(center):
        rows = center == c
        panel = controls[rows]
        loc = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0))
        x[rows] = (x[rows] - loc) / scale
    x[np.asarray(split['missing_channel_mask'], dtype=bool)] = np.nan
    return x


def _cell_stats(x, center, stratum):
    means, groups = [], []
    for c in np.unique(center):
        for s in np.unique(stratum[center == c]):
            group = np.flatnonzero((center == c) & (stratum == s))
            means.append(np.nanmean(x[group], axis=0))
            groups.append(group)
    weights = np.empty(len(x))
    for group in groups:
        weights[group] = len(x) / (len(groups) * len(group))
    return np.nanmean(means, axis=0), weights


def _logistic(x, y, weight):
    def objective(b):
        eta = x @ b
        loss = weight @ (np.logaddexp(0, eta) - y * eta)
        grad = x.T @ (weight * (expit(eta) - y))
        return loss, grad
    return minimize(objective, np.zeros(x.shape[1]), jac=True, method='BFGS',
                    options={'gtol': 1e-7}).x


def _predict(x, xq, train, query):
    center = np.asarray(train['center_id'])
    strata = np.asarray(train['stratum'])
    means, weight = _cell_stats(x, center, strata)
    x = np.where(np.isnan(x), means, x)
    xq = np.where(np.isnan(xq), means, xq)
    b = _logistic(np.column_stack((x, np.eye(3)[strata])), np.asarray(train['y']), weight)
    pred = expit(xq @ b[:3] + b[3:][np.asarray(query['stratum'])])
    return xq, np.clip(pred, 1e-15, 1 - 1e-15), b[:3]


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Align raw moduli and predict failure without using query outcomes."""
    taus = np.asarray(cohort['taus'], dtype=float)
    train, query = cohort['train'], cohort['query']
    x, xq = _aligned(train, taus), _aligned(query, taus)
    moduli, probability, weights = _predict(x, xq, train, query)
    return {'aligned_query_moduli': moduli, 'query_probabilities': probability,
            'modulus_weights': weights}
PY
cat >/tmp/visco_eval.py <<'PY'
import importlib.util, pickle
import numpy as np
from scipy.stats import rankdata
from scipy.special import logit
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('vp',p+'viscoelastic_panel_calibration.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def auc(y,v):
    n=sum(y); k=len(y)-n
    return (rankdata(v)[y==1].sum()-n*(n+1)/2)/(n*k) if n*k else np.nan
with open(p+'public_viscoelastic_cohort.pkl','rb') as f: data=pickle.load(f)
for i,case in enumerate(data):
    c,g=case['cohort'],case['gt']; tr,q=c['train'],c['query']; r=m.calibrate_viscoelastic_cohort(c)
    z,y,prob=g['Z_query'],g['y_query'],g['prob_query']; x,v,b=r.values()
    print('public',i,'taus',c['taus'],'beta fit/true',b,g['beta'],'MAE/Brier/AUC', np.mean(abs(x-z)),np.mean((v-y)**2),auc(y,v),'oracle AUC/Brier',auc(y,prob),np.mean((prob-y)**2))
    implied=logit(prob)-z@g['beta']; print('query true logit offsets by center/stratum:')
    for cc in np.unique(q['center_id']):
        w=q['center_id']==cc; mask=q['missing_channel_mask'][w]; e=(x-z)[w]; print(cc,w.sum(),'mae / unmasked / auc / Brier',np.abs(e).mean(),np.abs(e[~mask]).mean(),auc(y[w],v[w]),np.mean((v[w]-y[w])**2),'oracle auc',auc(y[w],prob[w]),'offset',[(np.mean(implied[w&(q['stratum']==s)]),np.std(implied[w&(q['stratum']==s)])) for s in range(3)],'loc error',e.mean(0))
    xt=m._aligned(tr,np.asarray(c['taus']))
    print('train center counts/rates/latent means',[(k,[(int(w.sum()),round(tr['y'][w].mean(),3)) for w in [(tr['center_id']==k)&(tr['stratum']==s) for s in range(3)]],np.round(np.nanmean(xt[tr['center_id']==k],axis=0),3)) for k in np.unique(tr['center_id'])])
    for k in np.unique(tr['center_id']):
        w=tr['center_id']==k
        sub=lambda w:{a:np.asarray(tr[a])[w] for a in ('center_id','stratum','y')}
        _, pv, bv=m._predict(xt[~w],xt[w],sub(~w),sub(w)); print('LOCO',k,'AUC/Brier',auc(tr['y'][w],pv),np.mean((pv-tr['y'][w])**2),'beta',bv)
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/visco_eval.py
```
- **Container Exit Code**: `0` (`1676 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/viscoelastic_panel_calibration.py']`
- **Stdout (tail)**:
```text
array([0.344, 0.64 , 0.334])), (2, [(35, 0.314), (18, 0.333), (19, 0.421)], array([-0.454, -0.525, -0.614])), (3, [(35, 0.4), (24, 0.333), (13, 0.462)], array([-0.348, -0.419, -0.441]))]
LOCO 0 AUC/Brier 0.9404024767801857 0.0997323072312663 beta [ 1.89776271 -1.76803878  1.71991138]
LOCO 1 AUC/Brier 0.9378399378399378 0.10884061856000792 beta [ 2.33118039 -1.84431183  2.08318763]
LOCO 2 AUC/Brier 0.9123404255319149 0.11222165343527236 beta [ 2.39577004 -1.94788223  1.84115588]
LOCO 3 AUC/Brier 0.9082792207792207 0.11846960495613874 beta [ 2.31243681 -1.78499935  1.7705572 ]
public 1 taus [0.55, 2.6] beta fit/true [ 1.68423478 -1.37546608  1.4293179 ] [ 2.15 -1.95  1.8 ] MAE/Brier/AUC 0.08708528696385848 0.1151493118521004 0.9115637651821862 oracle AUC/Brier 0.930668016194332 0.09799772901758218
query true logit offsets by center/stratum:
4 90 mae / unmasked / auc / Brier 0.08605696615839474 0.02551803217492846 0.883 0.1372748155088838 oracle auc 0.909 offset [(-0.900000000000001, 2.7367045194873433e-14), (4.880511059393364e-15, 2.392189228783425e-14), (0.8499999999999948, 1.3741336715310729e-14)] loc error [-0.00106576 -0.01068441  0.02316237]
5 90 mae / unmasked / auc / Brier 0.08811360776932224 0.034733122586571734 0.9295267489711934 0.09302380819531696 oracle auc 0.9449588477366255 offset [(-0.9000000000000002, 3.4454139822928303e-15), (-2.7124767078910075e-17, 2.6709136651359064e-16), (0.849999999999999, 4.062398269647891e-15)] loc error [0.0466216  0.07829049 0.0275906 ]
train center counts/rates/latent means [(0, [(42, 0.548), (19, 0.684), (11, 0.455)], array([0.674, 0.859, 0.644])), (1, [(36, 0.472), (24, 0.667), (12, 1.0)], array([0.504, 0.461, 0.92 ])), (2, [(33, 0.212), (27, 0.556), (12, 0.5)], array([-0.622, -0.53 , -0.337])), (3, [(39, 0.282), (22, 0.364), (11, 0.636)], array([-0.55 , -0.399, -0.34 ]))]
LOCO 0 AUC/Brier 0.9071597167584579 0.1262957345238396 beta [ 1.82731514 -1.46097285  1.51685143]
LOCO 1 AUC/Brier 0.9720164609053498 0.07421877233560314 beta [ 1.49910243 -1.20964247  1.26858056]
LOCO 2 AUC/Brier 0.8928571428571429 0.12871846711583115 beta [ 1.79011547 -1.41370587  1.53391493]
LOCO 3 AUC/Brier 0.8687290969899666 0.14554513267199365 beta [ 1.90719103 -1.66151363  1.62963699]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b736cb34e04f7ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b736cb3 (directory).
```

### Turn 3 — `execute_in_remote_docker` (`31353 ms` LLM)
- **Rationale**: Compare prespecified weighting and complete-case fits with imputation means fitted inside each held-out-center fold. This distinguishes attenuation from missing predictors from sensitivity to cell weighting, without tuning on query labels.
- **Alternatives Weighed**: `Adjust the rheometry extraction` (rejected: Observed-channel MAE is at most 0.051 per public query center; the smaller margin is in risk discrimination.)

```bash
cat >/tmp/visco_risk_compare.py <<'PY'
import contextlib, io, runpy
with contextlib.redirect_stdout(io.StringIO()):
    env = runpy.run_path('/tmp/visco_eval.py')
import numpy as np
from scipy.special import expit
m, data, auc = (env[k] for k in ('m','data','auc'))
for i, case in enumerate(data):
    co,g = case['cohort'],case['gt']; tr,q = co['train'],co['query']
    x,xq = [m._aligned(d,np.asarray(co['taus'])) for d in (tr,q)]
    c,s,y = [np.asarray(tr[k]) for k in ('center_id','stratum','y')]
    cq,sq = [np.asarray(q[k]) for k in ('center_id','stratum')]
    print('cohort',i,'missing train/query',np.isnan(x).mean(),np.isnan(xq).mean(),'impute means',m._cell_stats(x,c,s)[0])
    for test in list(np.unique(c))+['query']:
        iq = test == 'query'; w = np.ones(len(x),bool) if iq else c!=test
        xv,sv,yv = (xq,sq,g['y_query']) if iq else (x[~w],s[~w],y[~w])
        cv = cq if iq else c[~w]
        mu,weight = m._cell_stats(x[w],c[w],s[w]); xt=x[w]
        at=np.column_stack((np.where(np.isnan(xt),mu,xt),np.eye(3)[s[w]])); av=np.column_stack((np.where(np.isnan(xv),mu,xv),np.eye(3)[sv])); complete=~np.isnan(xt).any(1)
        print(' fold',test,'complete-vs-missing errors')
        for label,ww,keep in [('cell',weight,np.ones(len(xt),bool)),('pooled',np.ones(len(xt)),np.ones(len(xt),bool)),('complete-cell',weight,complete),('complete-pooled',np.ones(len(xt)),complete)]:
            b=m._logistic(at[keep],y[w][keep],ww[keep]); p=expit(av@b)
            scores=[(round(auc(yv[z],p[z]),4),round(np.mean((p[z]-yv[z])**2),4)) for z in [cv==cc for cc in np.unique(cv)]]
            missing=np.isnan(xv).any(1); mb=[round(np.mean((p[z]-yv[z])**2),4) for z in (~missing,missing) if z.any()]
            print(' ',label,scores,'Brier by missing status',mb,'beta',b[:3].round(3))
        if iq:
            p=g['prob_query']; print(' oracle brier by missing', [round(np.mean((p[z]-yv[z])**2),4) for z in (~missing,missing)])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/visco_risk_compare.py
```
- **Container Exit Code**: `0` (`2825 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
9072, 0.1263)] Brier by missing status [0.0942, 0.3507] beta [ 1.827 -1.461  1.517]
  pooled [(0.9087, 0.1289)] Brier by missing status [0.0966, 0.355] beta [ 1.902 -1.709  1.636]
  complete-cell [(0.9048, 0.1392)] Brier by missing status [0.107, 0.3648] beta [ 1.925 -2.032  1.725]
  complete-pooled [(0.904, 0.1393)] Brier by missing status [0.1072, 0.3642] beta [ 1.853 -1.989  1.715]
 fold 1 complete-vs-missing errors
  cell [(0.972, 0.0742)] Brier by missing status [0.0707, 0.0961] beta [ 1.499 -1.21   1.269]
  pooled [(0.9737, 0.0724)] Brier by missing status [0.0695, 0.0905] beta [ 1.573 -1.27   1.346]
  complete-cell [(0.9778, 0.0629)] Brier by missing status [0.0595, 0.0836] beta [ 1.74  -1.691  1.563]
  complete-pooled [(0.9778, 0.0628)] Brier by missing status [0.0605, 0.0773] beta [ 1.697 -1.565  1.531]
 fold 2 complete-vs-missing errors
  cell [(0.8929, 0.1287)] Brier by missing status [0.1301, 0.1194] beta [ 1.79  -1.414  1.534]
  pooled [(0.8904, 0.1295)] Brier by missing status [0.1301, 0.1252] beta [ 1.903 -1.585  1.616]
  complete-cell [(0.8856, 0.1343)] Brier by missing status [0.1312, 0.1553] beta [ 2.143 -2.294  2.115]
  complete-pooled [(0.888, 0.1321)] Brier by missing status [0.1295, 0.1505] beta [ 2.156 -2.147  1.989]
 fold 3 complete-vs-missing errors
  cell [(0.8687, 0.1455)] Brier by missing status [0.1134, 0.2675] beta [ 1.907 -1.662  1.63 ]
  pooled [(0.8679, 0.145)] Brier by missing status [0.1137, 0.2638] beta [ 1.822 -1.566  1.657]
  complete-cell [(0.8671, 0.148)] Brier by missing status [0.1135, 0.2791] beta [ 2.079 -1.903  1.796]
  complete-pooled [(0.8662, 0.1471)] Brier by missing status [0.1138, 0.2735] beta [ 1.906 -1.732  1.784]
 fold query complete-vs-missing errors
  cell [(0.883, 0.1373), (0.9295, 0.093)] Brier by missing status [0.106, 0.1518] beta [ 1.684 -1.375  1.429]
  pooled [(0.8815, 0.138), (0.929, 0.0922)] Brier by missing status [0.1057, 0.1528] beta [ 1.754 -1.481  1.52 ]
  complete-cell [(0.8765, 0.1436), (0.9275, 0.0918)] Brier by missing status [0.106, 0.1645] beta [ 1.93  -1.923  1.747]
  complete-pooled [(0.8785, 0.1426), (0.9275, 0.0921)] Brier by missing status [0.1059, 0.1629] beta [ 1.879 -1.811  1.711]
 oracle brier by missing [0.1072, 0.0613]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b736cb34e04f7ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b736cb3 (directory).
```

### Turn 4 — `execute_in_remote_docker` (`28601 ms` LLM)
- **Rationale**: Missing channels account for the largest observed calibration errors. Test whether integrating their uncertainty around the required mean improves the weakest held-out centers without changing the moduli or selecting on query outcomes; retain the baseline if it does not.
- **Alternatives Weighed**: `Replace the cell-balanced mean with conditional imputation` (rejected: The specification requires cell-balanced training means for the returned aligned moduli.); `Use complete-case logistic regression directly` (rejected: It worsened the weakest public center AUROC from 0.8830 to 0.8765 and increased Brier error on missing rows in several held-out centers.)

```bash
cat >/tmp/visco_missing_check.py <<'PY'
import contextlib, io, runpy
import numpy as np
from scipy.special import expit
with contextlib.redirect_stdout(io.StringIO()):
    e = runpy.run_path('/tmp/visco_eval.py')
m, data, auc = (e[k] for k in ('m', 'data', 'auc'))
# Integrate missing predictor uncertainty around the mandated mean, without changing the imputed moduli.
nodes, ghw = np.polynomial.hermite.hermgauss(16)
for i, case in enumerate(data):
    co = case['cohort']; tr = co['train']
    x = m._aligned(tr, np.asarray(co['taus']))
    c,s,y = [np.asarray(tr[k]) for k in ('center_id','stratum','y')]
    for cc in np.unique(c):
        w = c != cc; xv=x[~w]; yv=y[~w]
        mu, weight = m._cell_stats(x[w],c[w],s[w]); xt = x[w]
        at = np.column_stack((np.where(np.isnan(xt),mu,xt),np.eye(3)[s[w]]))
        av = np.column_stack((np.where(np.isnan(xv),mu,xv),np.eye(3)[s[~w]]))
        missing = np.isnan(xv); complete = np.isfinite(xt).all(1)
        bv,cw = m._cell_stats(xt[complete],c[w][complete],s[w][complete]); del bv
        b0 = m._logistic(at,y[w],weight); b1=m._logistic(at[complete],y[w][complete],cw)
        var = np.nanvar(xt, axis=0)
        scenarios = [('baseline', expit(av@b0))]
        for name,b in [('complete+marginal',b1),('all+marginal',b0)]:
            sd = np.sqrt(missing@(var*b[:3]**2))
            scenarios.append((name,expit((av@b)[:,None]+np.sqrt(2)*sd[:,None]*nodes)@ghw/np.sqrt(np.pi)))
        print(i, cc, 'missing rows',missing.any(1).sum(), [(name,round(auc(yv,p),4),round(np.mean((p-yv)**2),4)) for name,p in scenarios])
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/visco_missing_check.py
```
- **Container Exit Code**: `0` (`2391 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
0 0 missing rows 13 [('baseline', 0.9404, 0.0997), ('complete+marginal', 0.9381, 0.1018), ('all+marginal', 0.9497, 0.0999)]
0 1 missing rows 15 [('baseline', 0.9378, 0.1088), ('complete+marginal', 0.9386, 0.1081), ('all+marginal', 0.9386, 0.1056)]
0 2 missing rows 8 [('baseline', 0.9123, 0.1122), ('complete+marginal', 0.9234, 0.1054), ('all+marginal', 0.9209, 0.1041)]
0 3 missing rows 13 [('baseline', 0.9083, 0.1185), ('complete+marginal', 0.9075, 0.1191), ('all+marginal', 0.8961, 0.1224)]
1 0 missing rows 9 [('baseline', 0.9072, 0.1263), ('complete+marginal', 0.9095, 0.1374), ('all+marginal', 0.9166, 0.1242)]
1 1 missing rows 10 [('baseline', 0.972, 0.0742), ('complete+marginal', 0.9778, 0.0661), ('all+marginal', 0.9704, 0.077)]
1 2 missing rows 9 [('baseline', 0.8929, 0.1287), ('complete+marginal', 0.8831, 0.1328), ('all+marginal', 0.8896, 0.1296)]
1 3 missing rows 15 [('baseline', 0.8687, 0.1455), ('complete+marginal', 0.8813, 0.1376), ('all+marginal', 0.8821, 0.1366)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b736cb34e04f7ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b736cb3 (directory).
```

### Turn 5 — `execute_in_remote_docker` (`38601 ms` LLM)
- **Rationale**: Retain the model that meets all public query-center thresholds. Verify the exact artifact in a clean process, using independent convolution to catch recurrence errors and known-coefficient replay on real sampling grids to catch robust-fit failures. A replay failure would trigger a targeted extraction fix rather than another risk-model adjustment.
- **Alternatives Weighed**: `Marginalize missing predictors in the risk model` (rejected: It improved the weakest LOCO center but reduced AUROC or worsened Brier in other folds, and it departs from the specified mean-imputation model.)

```bash
cat >/tmp/verify_visco.py <<'PY'
import importlib.util
import pickle
import numpy as np
from scipy.stats import rankdata

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('viscoelastic_panel_calibration', p+'viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(p+'public_viscoelastic_cohort.pkl', 'rb') as f:
    cases = pickle.load(f)

def auc(y, v):
    n = int(y.sum())
    return (rankdata(v)[y == 1].sum() - n*(n+1)/2)/(n*(len(y)-n))

rng = np.random.default_rng(821)
errors = []
for i, case in enumerate(cases):
    co, g = case['cohort'], case['gt']
    tr, q = co['train'], co['query']
    r = m.calibrate_viscoelastic_cohort(co)
    assert set(r) == {'aligned_query_moduli', 'query_probabilities', 'modulus_weights'}
    x, v, b = (r[k] for k in ('aligned_query_moduli','query_probabilities','modulus_weights'))
    y = np.asarray(g['y_query'])
    n = len(y)
    assert x.shape == (n,3) and v.shape == (n,) and b.shape == (3,)
    assert all(a.dtype.kind == 'f' and np.isfinite(a).all() for a in (x,v,b))
    assert ((v > 0) & (v < 1)).all()
    print('cohort',i,'weight error',np.round(b-g['beta'],4))
    for c in np.unique(q['center_id']):
        w = q['center_id'] == c
        metrics = np.abs(x[w]-g['Z_query'][w]).mean(),auc(y[w],v[w]),np.mean((v[w]-y[w])**2)
        print('query center',c,'MAE/AUROC/Brier',np.round(metrics,5))
        assert metrics[0] <= .22 and metrics[1] >= .88 and metrics[2] <= .16
    mask = np.asarray(q['missing_channel_mask'])
    xt = m._aligned(tr, np.asarray(co['taus']))
    groups = [(tr['center_id']==c)&(tr['stratum']==s) for c in np.unique(tr['center_id']) for s in range(3)]
    expected = np.nanmean([np.nanmean(xt[g],axis=0) for g in groups],axis=0)
    np.testing.assert_allclose(x[mask],np.broadcast_to(expected,x.shape)[mask])
    taus = np.asarray(co['taus'])
    subset_rng = np.random.default_rng(185)
    subsets = np.array([subset_rng.choice(36,4,replace=False) for _ in range(96)])
    for trace in q['traces']:
        tp = np.asarray(trace['past_timestamps']); up = np.asarray(trace['past_strain_history'])
        t = np.asarray(trace['timestamps']); u = np.asarray(trace['strain_u'])
        # Independent direct convolution using right-endpoint-held strain.
        ts = np.r_[tp,t]; us = np.r_[up,u]; modes = []
        for tau in taus:
            z = []
            for k in range(len(tp),len(ts)):
                z.append(us[0]*np.exp((ts[0]-ts[k])/tau) + us[1:k+1]@(np.exp((ts[1:k+1]-ts[k])/tau)-np.exp((ts[:k]-ts[k])/tau)))
            modes.append(z)
        a = np.column_stack((np.ones(len(t)),u,*modes))
        actual = m._design(trace,taus)
        np.testing.assert_allclose(actual,a,rtol=1e-11,atol=1e-12)
        for count in (4,6):
            coeff = rng.normal(size=4)
            signal = a@coeff
            j = rng.choice(len(t),count,replace=False)
            signal[j] += rng.choice([-1,1],size=count)*rng.uniform(4.5,9.,size=count)
            errors.append(np.max(np.abs(m._robust_fit(actual,signal,subsets)-coeff[1:])))
print('known coefficient replay: fits',len(errors),'worst absolute error',max(errors))
assert max(errors) < 1e-6
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_visco.py
```
- **Container Exit Code**: `0` (`2308 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cohort 0 weight error [ 0.0227  0.1548 -0.0074]
query center 4 MAE/AUROC/Brier [0.07507 0.95751 0.08694]
query center 5 MAE/AUROC/Brier [0.06929 0.92738 0.09727]
cohort 1 weight error [-0.4658  0.5745 -0.3707]
query center 4 MAE/AUROC/Brier [0.08606 0.883   0.13727]
query center 5 MAE/AUROC/Brier [0.08811 0.92953 0.09302]
known coefficient replay: fits 720 worst absolute error 5.268008251846368e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b736cb34e04f7ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b736cb3 (directory).
```

### Turn 6 — `submit_completed_task` (`29224 ms` LLM)
- **Rationale**: The saved artifact passed isolated-process checks for output shape, probability bounds, cell-balanced imputation, and all public query-center thresholds. Independent convolution agreed with the recurrence, and 720 glitch-injected replays recovered the known coefficients to within 5.3e-14.
- **Alternatives Weighed**: `Marginalize missing predictors in the risk model` (rejected: It improved the weakest training fold but worsened other held-out centers and departed from direct mean-imputed logistic prediction.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`22` turns remaining; artifact status={'/holdout_task/viscoelastic_panel_calibration.py': 'FILE:4189'}): Do NOT lock prematurely with `22` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. Across both public cohorts, every query center met all three targets: worst MAE 0.0881, lowest AUROC 0.8830, and worst Brier score 0.1373. The lowest training leave-one-center-out AUROC was 0.8687.`

### Turn 7 — `execute_in_remote_docker` (`29849 ms` LLM)
- **Rationale**: Compare raw, panel-aligned, and center-adjusted slope fits on the same held-out centers. Separate class recalls expose nuisance shortcuts that pooled AUROC can hide. Promote a slope adjustment only if it improves the weak center without introducing a material calibration regression.
- **Alternatives Weighed**: `Add nonlinear stress terms, drift operators, or relay modes` (rejected: The specified law is linear in the strain and two ZOH memory states; independent convolution and known-coefficient replay already verify that basis on public sampling grids. The input has no relay or drift state.)

```bash
cat >/tmp/visco_center_audit.py <<'PY'
import contextlib, io, runpy
import numpy as np
from scipy.special import expit
with contextlib.redirect_stdout(io.StringIO()):
    env = runpy.run_path('/tmp/visco_eval.py')
m, data, auc = (env[k] for k in ('m','data','auc'))

def metrics(y,p):
    return np.array([auc(y,p), np.mean((p-y)**2), np.mean(p[y==1]>=.5), np.mean(p[y==0]<.5)])

for i, case in enumerate(data):
    co = case['cohort']; tr = co['train']; taus = np.asarray(co['taus'])
    x = m._aligned(tr, taus)
    c,s,y = [np.asarray(tr[k]) for k in ('center_id','stratum','y')]
    raw = x.copy()
    for cc in np.unique(c):
        w = c == cc; panel = tr['controls'][w]
        raw[w] = raw[w]*np.nanmedian(np.nanstd(panel,axis=0)) + np.nanmedian(np.nanmean(panel,axis=0))
    pred = np.zeros((3, len(y)))
    for cc in np.unique(c):
        w = c != cc
        for j,xx in enumerate((raw,x)):
            mu, wt = m._cell_stats(xx[w],c[w],s[w]); a = np.column_stack((np.where(np.isnan(xx),mu,xx),np.eye(3)[s]))
            b = m._logistic(a[w],y[w],wt)
            pred[j,~w] = expit(a[~w]@b)
            if j == 1:
                # Fit center nuisances on training rows, then refit only stratum intercepts with slopes held fixed.
                d = np.eye(4)[c[w]][:,np.unique(c[w])[1:]]
                slopes = m._logistic(np.column_stack((a[w],d)),y[w],wt)[:3]
                offset = a[w,:3]@slopes
                intercept = []
                from scipy.optimize import brentq
                for ss in range(3):
                    k = s[w]==ss
                    intercept.append(brentq(lambda b: wt[k]@(expit(offset[k]+b)-y[w][k]),-50,50))
                pred[2,~w] = expit(a[~w,:3]@slopes+np.asarray(intercept)[s[~w]])
    print('cohort',i,'columns: AUROC/Brier/failure recall/nonfailure recall')
    for j, name in enumerate(('raw','panel aligned','panel aligned + nuisance slopes')):
        print(name,'pooled AUROC',round(auc(y,pred[j]),4))
        for cc in np.unique(c):
            w=c==cc
            print(' center',cc,metrics(y[w],pred[j,w]).round(4))
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/visco_center_audit.py && OPENBLAS_NUM_THREADS=1 python3 -I /tmp/verify_visco.py
```
- **Container Exit Code**: `0` (`4698 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cohort 0 columns: AUROC/Brier/failure recall/nonfailure recall
raw pooled AUROC 0.741
 center 0 [0.7384 0.217  0.6579 0.7941]
 center 1 [0.7257 0.2345 0.7879 0.5385]
 center 2 [0.8162 0.1685 0.52   0.9149]
 center 3 [0.75   0.2138 0.2857 0.8409]
panel aligned pooled AUROC 0.9225
 center 0 [0.9404 0.0997 0.8158 0.9118]
 center 1 [0.9378 0.1088 0.9091 0.7949]
 center 2 [0.9123 0.1122 0.72   0.9149]
 center 3 [0.9083 0.1185 0.7143 0.9318]
panel aligned + nuisance slopes pooled AUROC 0.9231
 center 0 [0.9427 0.0963 0.7895 0.9118]
 center 1 [0.9371 0.1095 0.9091 0.7949]
 center 2 [0.9132 0.1131 0.72   0.9149]
 center 3 [0.9099 0.1198 0.7143 0.9318]
cohort 1 columns: AUROC/Brier/failure recall/nonfailure recall
raw pooled AUROC 0.7672
 center 0 [0.7309 0.2104 0.8293 0.5806]
 center 1 [0.7663 0.1962 0.8444 0.5926]
 center 2 [0.7873 0.183  0.3929 0.9091]
 center 3 [0.7508 0.2078 0.5    0.7826]
panel aligned pooled AUROC 0.9137
 center 0 [0.9072 0.1263 0.8537 0.8065]
 center 1 [0.972  0.0742 0.9333 0.9259]
 center 2 [0.8929 0.1287 0.6429 0.9091]
 center 3 [0.8687 0.1455 0.6538 0.8261]
panel aligned + nuisance slopes pooled AUROC 0.9122
 center 0 [0.9072 0.1261 0.8537 0.8065]
 center 1 [0.9695 0.0755 0.9333 0.8889]
 center 2 [0.8945 0.1297 0.6429 0.9091]
 center 3 [0.8629 0.1486 0.6538 0.8261]
cohort 0 weight error [ 0.0227  0.1548 -0.0074]
query center 4 MAE/AUROC/Brier [0.07507 0.95751 0.08694]
query center 5 MAE/AUROC/Brier [0.06929 0.92738 0.09727]
cohort 1 weight error [-0.4658  0.5745 -0.3707]
query center 4 MAE/AUROC/Brier [0.08606 0.883   0.13727]
query center 5 MAE/AUROC/Brier [0.08811 0.92953 0.09302]
known coefficient replay: fits 720 worst absolute error 5.268008251846368e-14

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b736cb34e04f7ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b736cb3 (directory).
```

### Turn 8 — `submit_completed_task` (`15623 ms` LLM)
- **Rationale**: Panel alignment improved balanced accuracy over raw moduli in every held-out training center. The unchanged artifact passed isolated-process checks for the output contract and public thresholds; all 720 glitch-injected replays recovered the known coefficients within 5.3e-14.
- **Alternatives Weighed**: `Add training-center nuisance intercepts when estimating the slopes` (rejected: This lowered the weakest held-out-center AUROC from 0.8687 to 0.8629 and worsened its Brier score from 0.1455 to 0.1486.)
- **MECHANICAL ROOT-PULLBACK GATE INTERCEPTION**: HEGEL-M6.8.1-B' DEEP CAUSAL & TOLERANCE-MARGIN AUDIT GATE (`20` turns remaining; artifact status={'/holdout_task/viscoelastic_panel_calibration.py': 'FILE:4189'}): Do NOT lock prematurely with `20` unused turns! Run a focused `/tmp` audit on REAL measurements: (a) check `RUNG_2` algebraic completeness (rational quotients `P(x)/(1+a*Q(x)^2)`, pairwise differences `(x_i +- c*x_j)`, exact Fokker-Planck/HJB drift, BOTH initial relay modes `{0, 1}`); (b) check Simpson's-paradox nuisance confounding (if any multi-batch cohort has within-batch balanced accuracy `~0.50` / zero single-class recall, promote the batch-centered estimator EVEN IF confounded pooled CV drops!); (c) check specification tolerance margins (do NOT reject a calibration/control upgrade over a tiny sub-tolerance fluctuation when it fixes a primary bottleneck!), verify clean-process execution, and submit.
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. Every public query center met all three targets: worst MAE 0.0881, lowest AUROC 0.8830, and worst Brier score 0.1373. The lowest training leave-one-center-out AUROC was 0.8687.`

### Turn 9 — `execute_in_remote_docker` (`31285 ms` LLM)
- **Rationale**: The weakest query center clears the AUROC target by only 0.0030. Isolate observed-channel alignment error, mandated mean imputation, and risk-fit error using public ground truth only as a diagnostic. A materially better oracle on the same estimated inputs would justify investigating the risk fit inside training folds; a gap driven by missing channels would not justify changing the specified imputation.
- **Alternatives Weighed**: `Repeat the raw-versus-aligned center audit` (rejected: The audit already showed improved balanced accuracy in every held-out center after alignment; repeating it would not identify the source of the narrow public AUROC margin.)

```bash
cat >/tmp/visco_margin_audit.py <<'PY'
import pickle
import runpy
import contextlib
import io
import numpy as np
from scipy.special import expit, logit
with contextlib.redirect_stdout(io.StringIO()):
    env = runpy.run_path('/tmp/visco_eval.py')
m, cases, auc = (env[k] for k in ('m', 'data', 'auc'))
for i, case in enumerate(cases):
    co, gt = case['cohort'], case['gt']
    q = co['query']
    r = m.calibrate_viscoelastic_cohort(co)
    x, pred = r['aligned_query_moduli'], r['query_probabilities']
    z, y, p = (gt[k] for k in ('Z_query', 'y_query', 'prob_query'))
    mask = q['missing_channel_mask']
    # Diagnostic counterfactuals only: oracle substitutions never enter the estimator.
    eta = logit(p)
    beta = gt['beta']
    aligned_only = expit(eta + np.where(mask, 0, x-z) @ beta)
    imputed_only = expit(eta + np.where(mask, x-z, 0) @ beta)
    oracle_on_estimated = expit(eta + (x-z) @ beta)
    for c in np.unique(q['center_id']):
        w = q['center_id'] == c
        scores = [(round(auc(y[w], v[w]), 5), round(np.mean((v[w]-y[w])**2), 5)) for v in (pred, oracle_on_estimated, aligned_only, imputed_only, p)]
        print(f'cohort {i}, center {c}, missing rows {mask[w].any(axis=1).sum()}: fitted / oracle on estimated inputs / observed alignment error only / imputation error only / full oracle (AUROC, Brier): {scores}')
PY
OPENBLAS_NUM_THREADS=1 python3 -I /tmp/visco_margin_audit.py
```
- **Container Exit Code**: `0` (`2289 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
cohort 0, center 4, missing rows 10: fitted / oracle on estimated inputs / observed alignment error only / imputation error only / full oracle (AUROC, Brier): [(0.95751, 0.08694), (0.958, 0.08434), (0.9585, 0.08514), (0.95899, 0.0849), (0.96047, 0.08513)]
cohort 0, center 5, missing rows 12: fitted / oracle on estimated inputs / observed alignment error only / imputation error only / full oracle (AUROC, Brier): [(0.92738, 0.09727), (0.92492, 0.10175), (0.95015, 0.08228), (0.92369, 0.1016), (0.94892, 0.08287)]
cohort 1, center 4, missing rows 18: fitted / oracle on estimated inputs / observed alignment error only / imputation error only / full oracle (AUROC, Brier): [(0.883, 0.13727), (0.876, 0.14829), (0.91, 0.12012), (0.8755, 0.14801), (0.909, 0.11951)]
cohort 1, center 5, missing rows 18: fitted / oracle on estimated inputs / observed alignment error only / imputation error only / full oracle (AUROC, Brier): [(0.92953, 0.09302), (0.92798, 0.08984), (0.94702, 0.07548), (0.92541, 0.09065), (0.94496, 0.07649)]

[HOST ARTIFACT IDENTITY]: candidate_bundle_sha256=5b736cb34e04f7ca (canonical path+NUL+sha256(bytes) bundle digest, NOT raw `sha256sum <file>`) | verification_receipt_bound=True | latest_snapshot_dir=/tmp/.hegel_candidate_archive/v01_5b736cb3 (directory).
```

### Turn 10 — `submit_completed_task` (`16089 ms` LLM)
- **Rationale**: The unchanged artifact passed isolated-process contract checks, all public query-center thresholds, and 720 glitch-injected coefficient replays. Panel alignment improved balanced accuracy in every held-out training center. The final audit supports retaining the required cell-balanced mean imputation.
- **Alternatives Weighed**: `Revise the risk fit to widen the weakest public AUROC margin` (rejected: On the same estimated inputs, the oracle risk model scored 0.8760 versus 0.8830 for the fitted model. The diagnostic identified missing-channel imputation, not observed-channel alignment or coefficient estimation, as the main source of the gap to the full oracle.)
- **Submitted Final Answer / Summary**: `Implemented and verified `/holdout_task/viscoelastic_panel_calibration.py`. Every public query center met all three targets: worst MAE 0.0881, lowest AUROC 0.8830, and worst Brier score 0.1373. The lowest training leave-one-center-out AUROC was 0.8687.`

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
