import pickle
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _design, _features, _loads, _predict_chamber
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']
out = solve_batch(d)
q = calibrate_upstream(dict(d, test=item['campaign_diagnostics']))['test']
labs = q['lab_id']
x = _features(q, q)[len(labs):]
past, hold = _loads(q)
dt, t = d['past_dt'], d['eval_times']
tau, theta = out['estimated_taus'], out['hereditary_weights']
y0, y, yr = q['strain_0'], q['strain_curves_obs'], q['instant_rate_obs']
curve, rates = _design(tau, x, past, hold, dt, t)
mech = y0[:, None] + curve[..., :2] @ theta[:2]
drift = theta[2] + x @ theta[3:]
state = np.zeros((len(labs), 2))
for j, step in enumerate(dt):
    state = np.exp(-step / tau) * state - np.expm1(-step / tau) * past[:, j, None]
basis = -np.expm1(-t[:, None] / tau)
np.testing.assert_allclose(curve[..., :2], (hold[:, None] - state)[:, None, :] * basis)
pkg_y, pkg_r = _predict_chamber(tau, theta, x, past, hold, dt, t, y0)
obs = np.column_stack((y, yr))
packaged = np.column_stack((pkg_y, pkg_r))
unique = np.unique(labs)

def predict(kind, p):
    a = p[0]
    f = -a * np.expm1(-t/a)
    der = 1.
    if kind == 'rational':
        f = (t + p[1]*t*t) / (1 + (t/a)**2)
    elif kind == 'rational_linear':
        f = t / (1+t/a)
    pred = mech + drift[:, None] * f
    rate = rates @ theta + (der-1) * drift
    if kind == 'free_memory':
        change = ((p[1:3]-1)*hold[:, None] - (p[3:5]-1)*state)*theta[:2]
        pred += change @ basis.T
        rate += change @ (1/tau)
    return np.column_stack((pred, rate))

models = {'exp': ([1.], [.05], [25.]), 'within_lab_exp': ([1.], [.05], [25.]), 'rational': ([1., .5], [.05, 0.], [25., 10.]), 'rational_linear': ([1.], [.05], [25.]), 'free_memory': ([1.,1.,1.,1.,1.], [.05,0.,0.,0.,0.], [25.,2.,2.,2.,2.])}

def metrics(err):
    return np.array([np.sqrt(np.mean(err[:, :-1]**2, axis=0)).max(), max(np.sqrt(np.mean(err[labs == l, :-1]**2)) for l in unique), max(np.sqrt(np.mean(err[labs == l, -1]**2)) for l in unique)])

print('Worst-horizon strain RMSE, worst-lab strain RMSE, worst-lab rate RMSE')
print('exact packaged', metrics(packaged - obs))
for kind, (start, lower, upper) in models.items():
    e = np.empty_like(obs)
    fits = []
    for held in unique:
        keep = labs != held
        def res(p):
            r = (predict(kind, p) - obs)[keep]
            if kind == 'within_lab_exp':
                for l in unique[unique != held]:
                    m = labs[keep] == l
                    r[m] -= r[m].mean(axis=0)
            return r.ravel()
        p = least_squares(res, start, bounds=(lower, upper), loss='soft_l1', f_scale=.006).x
        fits.append(p)
        e[~keep] = (predict(kind, p) - obs)[~keep]
    score = metrics(e)
    print(kind, score, 'fold parameters min/max', np.min(fits, axis=0), np.max(fits, axis=0))
    if kind == 'exp':
        assert max(score) < .01
    if kind == 'within_lab_exp':
        print('within-lab drift spread', [round(float(np.std(drift[labs == l])), 5) for l in unique])
assert max(metrics(packaged - obs)) < .01
