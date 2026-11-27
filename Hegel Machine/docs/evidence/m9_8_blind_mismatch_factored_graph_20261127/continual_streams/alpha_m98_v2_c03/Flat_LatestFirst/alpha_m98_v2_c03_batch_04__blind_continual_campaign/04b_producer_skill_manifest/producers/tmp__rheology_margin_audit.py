import importlib.util
import pickle
import numpy as np
from scipy.optimize import root

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
# Report the previously identified gross redistribution rows separately.
gross = np.isin(np.arange(len(x)), [13, 14, 15])
order = np.argsort(x.sum(axis=1))

def basis(v, mode):
    p = m._bridge_basis(v)
    if mode == 'full-quadratic':
        p = np.concatenate([p, v**2, (v[..., 0]*v[..., 2])[..., None], (v[..., 1]*v[..., 3])[..., None]], axis=-1)
    return p

for mode in ('cyclic-cubic', 'full-quadratic', 'rational'):
    errors, channel, denominators, gross_error = [], [], [], []
    for test in np.array_split(order, 4):
        tr = np.ones(len(x), dtype=bool)
        tr[test] = False
        q0 = m._calibration({'true_channel_loads': x[tr], 'bridge_raw': y[tr]})
        tr &= np.max(abs(m._bridge_basis(x) @ q0 - y), axis=1) < .002
        p = basis(x[tr], mode)
        if mode == 'rational':
            # y_j = P_j(x)/(1 + a_j*x_j**2), with a_j fit on this fold only.
            fits = [np.linalg.lstsq(np.c_[p, -y[tr, j]*x[tr, j]**2], y[tr, j], rcond=None)[0] for j in range(4)]
            q, a = np.array(fits).T[:-1], np.array(fits).T[-1]
        else:
            q, a = np.linalg.lstsq(p, y[tr], rcond=None)[0], np.zeros(4)
        def forward(v):
            return (basis(v, mode) @ q)/(1 + a*v**2)
        clean = test[~gross[test]]
        err = forward(x[test])-y[test]
        channel.append(np.max(abs(err[~gross[test]])))
        gross_error.extend(abs(err[gross[test]].sum(1)))
        denominators.append(np.min(1 + a*x[test]**2))
        sums = []
        for i in clean:
            guess = (y[i]-q0[0]) @ np.linalg.inv(q0[1:5])
            r = root(lambda v: forward(v)-y[i], guess, tol=1e-10)
            if np.max(abs(forward(r.x)-y[i])) > 1e-8:
                sums.append(np.inf)
            else:
                sums.append(abs(r.x.sum()-x[i].sum()))
        errors.append(np.max(sums))
    print(mode, 'worst inverse absolute stress by load quartile', np.round(errors, 7), 'worst channel by quartile', np.round(channel, 7), 'minimum denominator', min(denominators), 'gross-row max summed forward error', max(gross_error))

tr, _ = m._prepare(d)
obs = d['train']
y = np.column_stack((obs['strain_curves_obs'] - obs['strain_0'][:, None], obs['instant_rate_obs']))
for lab in np.unique(obs['lab_id']):
    keep = obs['lab_id'] != lab
    taus, theta = m._fit(tuple(v[keep] for v in tr), d['past_dt'], d['eval_times'], y[keep])
    residual = m._design(tuple(v[~keep] for v in tr), d['past_dt'], d['eval_times'], taus) @ theta - y[~keep]
    for st in np.unique(obs['stratum'][~keep]):
        r = residual[obs['stratum'][~keep] == st]
        print('lab/stratum', lab, st, 'n', len(r), 'worst horizon abs p90', np.max(np.quantile(abs(r[:, :-1]), .9, axis=0)), 'rate abs p90', np.quantile(abs(r[:, -1]), .9), 'signed curve/rate medians', np.median(r[:, :-1]), np.median(r[:, -1]))
