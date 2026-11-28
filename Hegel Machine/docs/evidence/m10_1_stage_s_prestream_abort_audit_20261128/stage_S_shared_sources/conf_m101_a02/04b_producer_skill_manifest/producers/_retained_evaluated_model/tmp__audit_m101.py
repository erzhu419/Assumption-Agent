import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

path = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', path + 'batch_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open(path + 'public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
x, y, traj = (tr[k] for k in ('x', 'y', 'traj'))
z = solver._calibrated(s, x)
F = solver._readout(z)
pair_difference = z[:, 0]*z[:, 2] - z[:, 1]*z[:, 3]
full = np.column_stack([np.ones(len(x)), z, *[z[:, a]*z[:, b] for a in range(4) for b in range(a, 4)]])

def ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]

def predict(key, i, j):
    if key == 'rational':
        q = z @ ls(z[i], y[i])
        def design(idx, a):
            return F[idx] / (1 + a*q[idx, None]**2)
        def loss(a):
            d = design(i, a)
            return np.mean((d @ ls(d, y[i]) - y[i])**2)
        a = minimize_scalar(loss, bounds=(0., 0.5), method='bounded').x
        d = design(i, a)
        return design(j, a) @ ls(d, y[i])
    d = designs[key]
    return d[j] @ ls(d[i], y[i])

designs = {'affine': np.column_stack([np.ones(len(x)), z]), 'deployed': F, 'untie_products': np.column_stack([F, pair_difference]), 'full_quadratic': full}
keys = (*designs, 'rational')
idx = np.arange(len(x))
edge = np.argsort(np.linalg.norm((x-x.mean(0))/x.std(0), axis=1))
folds = [('ordered', idx[:56], idx[56:]), ('edge', edge[:-16], edge[-16:])]
for seed in (83, 211):
    for k, j in enumerate(np.array_split(np.random.default_rng(seed).permutation(idx), 6)):
        folds.append((f'{seed}:{k}', np.setdiff1d(idx, j), j))
err = {k: [] for k in (*keys, 'nested_selection')}
terr = {k: [] for k in ('affine', 'rank2', 'deployed')}
chosen = []
for n, (name, i, j) in enumerate(folds):
    inner = [(np.setdiff1d(i, v), v) for v in np.array_split(np.random.default_rng(550+n).permutation(i), 4)]
    key = min(keys, key=lambda k: np.mean([np.mean(np.abs(predict(k, a, b)-y[b])) for a, b in inner]))
    chosen.append(key)
    for k in keys:
        err[k].append(np.abs(predict(k, i, j)-y[j]))
    err['nested_selection'].append(err[key][-1])
    out = solver.solve_batch({**s, 'train': {k: tr[k][i] for k in ('x', 'y', 'traj')}, 'test': {'x': x[j]}})
    np.testing.assert_allclose(np.abs(out['y_hat']-y[j]), err['deployed'][-1], atol=1e-12, rtol=0)
    w = ls(designs['affine'][i], traj[i])
    direct = designs['affine'][j] @ w
    _, _, vh = np.linalg.svd(designs['affine'][i] @ w, full_matrices=False)
    for k, p in zip(terr, (direct, direct @ vh[:2].T @ vh[:2], out['state_pred'])):
        e = np.abs(p-traj[j]); terr[k].append([e.mean(), e[:, -1].mean(), e.mean(0).max()])
for k, es in err.items():
    avg = np.array([e.mean() for e in es]); worst = avg.argmax()
    print('scalar', k, 'pooled/ordered/edge/worst', np.round([np.concatenate(es).mean(), avg[0], avg[1], avg[worst]], 7), 'worst slice', folds[worst][0])
print('inner choices', {k: chosen.count(k) for k in keys})
for k, es in terr.items():
    es = np.array(es); print('trajectory', k, 'mean and worst [all, terminal, worst horizon]', np.round(es.mean(0), 7), np.round(es.max(0), 7))
print('available study keys', sorted(s), 'train keys', sorted(tr))
assert np.concatenate(err['deployed']).mean() < 0.01 * 2
assert np.mean(np.concatenate(err['affine'])) > 5*np.mean(np.concatenate(err['deployed']))
