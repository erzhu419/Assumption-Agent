import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import hyp2f1
from itertools import combinations_with_replacement

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
print('Study fields:', sorted(s), '; train fields:', sorted(tr))
print('Telemetry shape:', np.shape(s['telemetry_vec']), '; distinct evaluation times:', len(np.unique(s['eval_times'])))

# Use the same robust pilot for the affine and cyclic fits, without the omitted row.
u, v = (s['panel_a'][key] for key in ('u', 'v'))
a = m._affine(u)
ref_errors = []
for i in range(len(u)):
    eligible = np.flatnonzero(np.arange(len(u)) != i)
    rng = np.random.default_rng(8)
    best_score = np.inf
    for _ in range(400):
        sample = rng.choice(eligible, a.shape[1], replace=False)
        err = np.linalg.norm(a @ m._fit(a[sample], v[sample]) - v, axis=1)
        score = np.median(err[eligible])
        if score < best_score:
            best_score, best_err = score, err
    keep = eligible[best_err[eligible] < max(.001, 5*best_score)]
    for _ in range(3):
        pilot = m._fit(a[keep], v[keep])
        err = np.linalg.norm(a @ pilot - v, axis=1)
        keep = eligible[err[eligible] < max(.001, 5*np.median(err[eligible]))]
    forward = np.array([m._calibration_basis(u[i:i+1], j)[0] @ m._fit(m._calibration_basis(u[keep], j), v[keep,j]) for j in range(u.shape[1])])
    ref_errors.append([np.linalg.norm(a[i] @ pilot - v[i]), np.linalg.norm(forward - v[i])])
ref_errors = np.array(ref_errors)
print('A leave-one-row-out row error median / 90th pct (affine, cyclic):', np.quantile(ref_errors,[.5,.9],axis=0))
print('A held-out residuals > 1 (affine, cyclic):', (ref_errors > 1).sum(0))

z = m._calibrate(s['panel_a'], tr['x'])
b = s['panel_b']
bz = m._affine(z) @ m._fit(m._affine(b['u']), b['v'])
cu, cv = (s['panel_c'][key] for key in ('u', 'v1'))
forms = {
    'integral': (lambda q,p: q*hyp2f1(.5,np.exp(p[1]),1.5,-np.exp(p[0])*q*q), [0.,0.]),
    'rational': (lambda q,p: q*(1+p[1]*q*q)/(1+np.exp(p[0])*q*q), [0.,.2]),
    'integral+linear': (lambda q,p: q*hyp2f1(.5,np.exp(p[1]),1.5,-np.exp(p[0])*q*q)+p[2]*q, [0.,0.,0.]),
    'asinh+linear': (lambda q,p: np.arcsinh(np.exp(p[0])*q)+p[1]*q, [.4,0.]),
}
allc = np.arange(len(cu))
coords = {}
for name, (phi, start) in forms.items():
    def calibrate(rows, x):
        def design(u,p): return m._affine(phi(u,p))
        def residual(p):
            g = design(cu[rows], p)
            return (g @ m._fit(g, cv[rows]) - cv[rows]).ravel()
        p = least_squares(residual,start,bounds=(-4,4),max_nfev=200).x
        return design(x,p) @ m._fit(design(cu[rows],p),cv[rows])
    loo = np.vstack([calibrate(allc[allc != i],cu[i:i+1]) for i in allc])
    coords[name] = calibrate(allc,bz)
    print(f'C1 {name} leave-one-row-out RMSE {np.mean((loo-cv)**2)**.5:.6g}')

ids = np.arange(len(z))
folds = [ids[52:], ids[56:], np.argsort(np.linalg.norm(z,axis=1))[-16:]]
folds += list(np.array_split(np.random.default_rng(207).permutation(len(z)),4))
for name,q in coords.items():
    g = m._affine(q)
    pair = np.column_stack((g,q[:,:2]*q[:,2:]))
    quad = np.column_stack((g,*[q[:,i]*q[:,j] for i,j in combinations_with_replacement(range(q.shape[1]),2)]))
    # The full quadratic includes every squared linear sum/difference. Separately
    # test whether a paired sum or difference acts in a rational denominator.
    rows = []
    for te in folds:
        ti = np.setdiff1d(ids,te)
        y = tr['y'][ti]
        def predict_rational(sign):
            d = (q[:,:2]+sign*q[:,2:])**2
            def numerator(p): return m._fit(pair[ti]/(1+d[ti]@p)[:,None],y)
            def residual(p): return (pair[ti] @ numerator(p))/(1+d[ti]@p)-y
            p = least_squares(residual,[.01,.01],bounds=(0,10)).x
            return (pair[te] @ numerator(p))/(1+d[te]@p)
        yh = [d[te] @ m._fit(d[ti],y) for d in (pair,quad)]
        yh.extend(predict_rational(sign) for sign in (1,-1))
        ds = np.abs(m._state(g[ti],tr['traj'][ti],g[te],s['eval_times'])-tr['traj'][te])
        rows.append([*(np.mean(np.abs(yj-tr['y'][te])) for yj in yh),ds.mean(),ds.mean(0).max()])
    print(f'{name}: 52/20 and worst fold MAEs [paired, quadratic, rational-sum, rational-diff, traj, worst timestamp]',np.round(rows[0],6),np.round(np.max(rows,axis=0),6))
