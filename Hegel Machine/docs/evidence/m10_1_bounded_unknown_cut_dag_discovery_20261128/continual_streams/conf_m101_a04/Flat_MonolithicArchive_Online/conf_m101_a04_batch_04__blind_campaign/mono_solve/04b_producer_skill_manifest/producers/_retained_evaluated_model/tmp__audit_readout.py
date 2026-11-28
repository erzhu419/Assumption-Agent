import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
x = m._calibrate(s['panel_a'], tr['x'])
c, d = m._operators(s['panel_c'], x)
f, g, h = m._features(s, tr['x'])
p = s['panel_b']
b = m._bias(x) @ m._fit(m._bias(p['u']), p['v'])
cd = (c[:, :, None] * d[:, None, :]).reshape(len(x), -1)
i = np.arange(len(x))
splits = [i[52:], np.argsort(np.linalg.norm(x, axis=1))[-16:]]
for j in range(4):
    order = np.argsort(x[:, j]); splits.extend((order[:16], order[-16:]))
metrics = {k: [] for k in ('current', 'affine_auxiliary', 'full_cross', 'rational')}
strengths = []
for val in splits:
    fit = np.setdiff1d(i, val)
    rates, _ = m._fit_trajectory(g[fit], h[fit], s['eval_times'], tr['traj'][fit])
    design = m._temporal_design(g, h, s['eval_times'], rates)
    basis = m._temporal_design(np.ones((len(x), 1)), np.ones((len(x), 1)), s['eval_times'], rates)
    z = tr['traj']; y = tr['y']
    def errors(a, t, gain):
        a = a * gain[:, None]; t = t * gain[:, None, None]
        yp = a @ m._fit(a[fit], y[fit]); zp = t @ m._fit(t[fit].reshape(-1, t.shape[-1]), z[fit].ravel())
        return (yp-y), (zp-z)
    cross_time = np.concatenate((design, basis[:, :, :1]*cd[:, None, :], basis[:, :, 1:]*cd[:, None, :]), axis=-1)
    aux_time = np.concatenate((design, basis[:, :, :1]*b[:, None, :], basis[:, :, 1:]*b[:, None, :]), axis=-1)
    q = np.mean(x**2, axis=1)
    gain = lambda a: 1 / (1 + a*q)
    def objective(a):
        ey, ez = errors(f, design, gain(a))
        return np.mean(ey[fit]**2) + np.mean(ez[fit]**2)
    opt = minimize_scalar(objective, bounds=(0, 1), method='bounded', options={'xatol': 1e-8})
    a = opt.x if opt.fun < objective(0) else 0.
    strengths.append(a)
    for name, fa, ta, gainval in [('current',f,design,gain(0)),('affine_auxiliary',np.c_[f,b],aux_time,gain(0)),('full_cross',np.c_[f,cd],cross_time,gain(0)),('rational',f,design,gain(a))]:
        ey, ez = errors(fa,ta,gainval)
        metrics[name].append([np.abs(ey[val]).mean(), np.abs(ez[val]).mean()])
for name, scores in metrics.items():
    print(name, 'public', np.round(scores[0],6), 'worst',np.round(np.max(scores,axis=0),6))
print('rational strength range', min(strengths), max(strengths))
