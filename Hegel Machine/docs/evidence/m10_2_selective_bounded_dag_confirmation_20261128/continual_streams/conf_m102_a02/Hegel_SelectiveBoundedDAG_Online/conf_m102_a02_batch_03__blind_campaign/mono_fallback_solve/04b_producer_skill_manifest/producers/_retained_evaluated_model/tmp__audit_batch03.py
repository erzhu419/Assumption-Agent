import importlib.util
import pickle
import numpy as np
from itertools import combinations

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
tr = s['train']
x, y, z = (tr[k] for k in ('x', 'y', 'traj'))
panel = s['panel_a']
coef = m._fit_calibration(panel)
q = m._invert(x, coef)
base = {'raw': (m._design(x, 15), m._design(x, 5)), 'inverse': (m._design(q, 15), m._design(q, 5))}
print('Observed schema:', sorted(s), sorted(tr), 'telemetry:', s['telemetry_vec'])
print('Time grid:', s['eval_times'], 'trajectory singular values:', np.linalg.svd(z, compute_uv=False))
# Pairwise linear sums/differences and squares are already in the polynomial spans.
q2 = base['inverse'][0]
for i, j in combinations(range(4), 2):
    for c in (-2., -1., 1., 2.):
        d = q[:, i] + c*q[:, j]
        assert np.max(abs(d*d - q2 @ m._ls(q2, d*d))) < 1e-10
# Challenge with quotients on single and paired directions; select using only inner folds.
dirs = [q[:, j] for j in range(4)] + [q[:, i] + c*q[:, j] for i,j in combinations(range(4),2) for c in (-1,1)]

def fit(a,b,design,target):
    return design[b] @ m._ls(design[a],target[a])

def score(a,b,design,target):
    return np.mean(abs(fit(a,b,design,target)-target[b]))

for name,target,F in [('scalar',y,q2),('state',z,base['inverse'][1])]:
    alternatives = [('unchanged',F), ('raw',base['raw'][0 if name=='scalar' else 1]), ('extra polynomial',m._design(q,35 if name=='scalar' else 15))]
    for j,d in enumerate(dirs):
        for h in (.05,.22,1.):
            w=1+h*d*d
            alternatives.append((f'quotient {j} {h}',F/w[:,None]))
            alternatives.append((f'augmented quotient {j} {h}',np.c_[F,d/w]))
    res=[]; picks={}
    for seed in range(-1,12):
        p=np.arange(len(y)) if seed==-1 else np.random.default_rng(seed).permutation(len(y))
        a,b=p[:52],p[52:]
        inner=np.array_split(np.random.default_rng(123).permutation(a),4)
        pick,G=min(alternatives,key=lambda pair: np.mean([score(np.setdiff1d(a,v),v,pair[1],target) for v in inner]))
        picks[pick]=picks.get(pick,0)+1
        res.append([score(a,b,H,target) for H in [base['raw'][0 if name=='scalar' else 1], F,G]])
    print(name,'raw/inverse/nested challenger; 52/20:',res[0], 'mean:',np.mean(res,0),'worst:',np.max(res,0),'selected:',picks)

# Test calibration stability by deleting one real reference row at a time.
full_q = m._invert(np.r_[x,s['test']['x']],coef)
shifts=[]; calibration_cv=[]
for j in range(len(panel['u'])):
    mask=np.arange(len(panel['u']))!=j
    fit_coef=m._fit_calibration({k:v[mask] for k,v in panel.items()})
    shifts.append(np.max(np.abs(m._invert(np.r_[x,s['test']['x']],fit_coef)-full_q)))
    calibration_cv.append(np.linalg.norm(m._design(panel['u'][j],35)@fit_coef-panel['v'][j]))
print('Panel jackknife worst absolute latent displacement:',max(shifts),'sorted reference prediction errors:',np.sort(calibration_cv)[-8:])

# Fit only on 52 rows, then test both held-out extrapolation tails of each latent coordinate.
worst=np.zeros(3)
for j in range(4):
    for side in (-1,1):
        p=np.argsort(side*q[:,j]); a,b=p[:52],p[52:]
        o=m.solve_batch(dict(s,train={k:tr[k][a] for k in ('x','y','traj')},test={'x':x[b]}))
        ey=np.mean(abs(o['y_hat']-y[b])); ez=np.mean(abs(o['state_pred']-z[b]),axis=0)
        worst=np.maximum(worst,[ey,ez.mean(),ez.max()])
print('Worst latent-coordinate tail MAEs, scalar/state/worst horizon:',worst.tolist())
assert np.all(worst < .038)
