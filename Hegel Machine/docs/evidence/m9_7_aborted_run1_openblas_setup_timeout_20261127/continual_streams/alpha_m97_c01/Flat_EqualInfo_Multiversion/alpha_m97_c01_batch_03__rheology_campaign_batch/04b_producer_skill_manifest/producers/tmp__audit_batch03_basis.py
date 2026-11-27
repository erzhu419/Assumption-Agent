import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c01_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c01_batch_03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
s = d['train']
lab = s['lab_id']
n = len(lab)
dt, t = d['past_dt'], d['eval_times']
y = s['strain_curves_obs'] - s['strain_0'][:, None]
r = s['instant_rate_obs']
state = tuple(v[:n] for v in m._prepare(d))
original = m._design

def basis(kind):
    def design(tau, inputs, dt, t):
        a, b = original(tau, inputs, dt, t)
        if kind == 'no cross-gap':
            return a[..., :-1], b[..., :-1]
        if kind == 'no signed slip':
            cols = [0, 1, 2, 3, 4, 5, 6, 8]
            return a[..., cols], b[..., cols]
        if kind == 'linear terminal flow':
            a[..., 2:8] *= (1 + .85*t)[None, :, None]
        return a, b
    return design

for kind in ('full', 'no cross-gap', 'no signed slip', 'linear terminal flow'):
    m._design = basis(kind)
    rows = []
    bias = []
    for c in np.unique(lab):
        train = lab != c
        test = ~train
        tau, w = m._fit(tuple(v[train] for v in state), dt, t, y[train], r[train])
        a, b = m._design(tau, tuple(v[test] for v in state), dt, t)
        e, f = a @ w - y[test], b @ w - r[test]
        edge = np.array_split(np.arange(len(t)), 2)
        rows.append([np.median(np.abs(e[:, j])) for j in edge] + [np.median(np.abs(f))])
        hold = state[1][test]
        for sign in (hold < 0, hold >= 0):
            if np.any(sign):
                bias.append(np.abs([np.median(e[sign]), np.median(f[sign])]))
        if kind == 'full':
            hits = (np.abs(e) > .2).sum(axis=1)
            print('lab', int(c), 'gross curve residuals by specimen', np.bincount(hits, minlength=len(t)+1).tolist())
    print(kind, 'held-lab median absolute errors (early strain, late strain, initial rate)', np.round(rows, 6).tolist(), 'worst', np.round(np.max(rows, axis=0), 6).tolist(), 'max lab-by-hold-sign absolute median biases', np.round(np.max(bias, axis=0), 6).tolist())
