import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
train, test = d['train'], d['test']
labs = train['lab_id']; strata = train['stratum']; n = len(labs)
x, observed = s._features(train, test)
complete = observed[:n].all(axis=1)
laws = s._calibrate(d['calibration_load_cells'])
dt, times = d['past_dt'], d['eval_times']
cal = s._Design(train, laws, dt, times, x[:n])
raw = s._Design(train, np.tile([1., 0.], (4, 1)), dt, times, x[:n])
public = s.solve_alpha_ep04(d)
tau = public['estimated_taus']
assert np.all(np.isfinite(tau)) and 0 < tau[0] < tau[1]
for name, design in [('train', cal), ('test', s._Design(test, laws, dt, times, x[n:]))]:
    state = np.zeros((len(design.hold), 2))
    for step, h in enumerate(dt):
        q = np.exp(-h/tau)
        state = q*state + (1-q)*design.past[:, step, None]
    gap = design.hold[:, None] - state
    R = design.matrix(np.log(tau))
    np.testing.assert_allclose(R[:, :-1, :2], gap[:, None, :]*(1-np.exp(-times[:, None]/tau)), atol=1e-12)
    np.testing.assert_allclose(R[:, -1, :2], gap/tau, atol=1e-12)
    peak = max(np.abs(design.past).max(), np.abs(design.hold).max())
    print(name, 'physical peak', round(peak, 6), 'yield margin', round(d['yield_stress']-peak, 6))
    assert peak < d['yield_stress']
y = np.column_stack([train['strain_curves_obs']-train['strain_0'][:, None], train['instant_rate_obs']])
pred = {k: np.empty_like(y) for k in ['calibrated', 'command-as-physical', 'cubic-drift']}
t = np.r_[times, 1.]
# Pairwise linear sums/differences already lie in the original drift span.
# Distinct three-feature products do not, so test their transfer separately.
xtr = x[:n]
cubic = np.column_stack([np.prod(xtr[:, [i,j,k]], axis=1) for i,j,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]])
for lab in np.unique(labs):
    held = labs == lab
    fit = ~held & complete
    for key, z in [('calibrated', cal), ('command-as-physical', raw)]:
        ft, w = s._fit(z, y, fit)
        A = z.matrix(np.log(ft))
        p = A[held] @ w
        p += np.where(~complete[held], y[held, -1]-p[:, -1], 0.)[:, None]*t
        pred[key][held] = p
        if key != 'calibrated':
            continue
        B = np.concatenate([A, cubic[:, None, :]*t[None, :, None]], axis=2)
        scale = s._scale(A[fit] @ w-y[fit])
        M = (B[fit]/scale[:, None]).reshape(-1, B.shape[-1])
        target = (y[fit]/scale).ravel()
        q = least_squares(lambda v: M@v-target, np.r_[w,np.zeros(4)], jac=lambda v: M.copy(), loss='cauchy', f_scale=2.)
        p = B[held] @ q.x
        p += np.where(~complete[held], y[held, -1]-p[:, -1], 0.)[:, None]*t
        pred['cubic-drift'][held] = p
for key, p in pred.items():
    e = np.abs(p-y)
    for mask in (complete, ~complete):
        cells = [mask & (labs == l) & (strata == st) for l in np.unique(labs) for st in np.unique(strata)]
        worst_curve = max(e[m, :-1].mean() for m in cells if m.sum() >= 3)
        print(key, 'n', int(mask.sum()), 'curve MAE', round(e[mask, :-1].mean(), 6), 'worst-cell', round(worst_curve, 6), 'median late', round(np.median(e[mask,-2]),6))
    print(key, 'complete worst-lab rate MAE', round(max(e[complete & (labs == l), -1].mean() for l in np.unique(labs)), 6))
