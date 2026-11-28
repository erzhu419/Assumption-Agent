import importlib.util
import pickle
import numpy as np

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('batch_solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
cal = s['panel_a']
u, v = cal['u'], cal['v']
A = m._design(u)
rng = np.random.default_rng(45)
fits = (m._ls(A[j], v[j]) for j in (rng.choice(len(u), 8, replace=False) for _ in range(120)))
pilot = min(fits, key=lambda q: np.median(np.linalg.norm(A @ q - v, axis=1)))
r = np.linalg.norm(A @ pilot - v, axis=1)
inliers = np.flatnonzero(r <= 5*np.median(r))

def diagonal(p, x):
    qs = np.array([m._ls(m._design(p['u'][:, k:k+1]), p['v'][:, k]) for k in range(4)])
    return (x - qs[:, 0])/qs[:, 1]

def affine(p, x):
    q = m._ls(m._design(p['u']), p['v'])
    return np.linalg.solve(q[1:].T, (x - q[0]).T).T

original = m._calibrate
tr = s['train']
fit = {k: tr[k][:52] for k in ('x', 'y', 'traj')}
check = dict(s, train=fit, test={'x': tr['x'][52:]})
print('Calibration ablation: held-out reference inverse RMSE and 52/20 downstream MAEs; downstream fitting unchanged')
for name, fn in [('diagonal affine', diagonal), ('full affine', affine), ('candidate cyclic cubic', original)]:
    residuals = []
    for i in inliers:
        keep = np.arange(len(u)) != i
        panel = {'u': u[keep], 'v': v[keep]}
        if fn is not original:
            mask = np.isin(np.flatnonzero(keep), inliers)
            panel = {k: q[mask] for k, q in panel.items()}
        residuals.append(fn(panel, v[i:i+1])[0] - u[i])
    panel = cal if fn is original else {k: q[inliers] for k, q in cal.items()}
    m._calibrate = fn
    out = m.solve_batch(dict(check, panel_a=panel))
    rmse = np.sqrt(np.mean(np.square(residuals)))
    ym = np.mean(np.abs(out['y_hat'] - tr['y'][52:]))
    tm = np.mean(np.abs(out['state_pred'] - tr['traj'][52:]))
    print(f'{name}: inverse_rmse={rmse:.8f}, response_mae={ym:.8f}, trajectory_mae={tm:.8f}')
    if fn is original:
        assert rmse < 0.001 and max(ym, tm) < 0.038
