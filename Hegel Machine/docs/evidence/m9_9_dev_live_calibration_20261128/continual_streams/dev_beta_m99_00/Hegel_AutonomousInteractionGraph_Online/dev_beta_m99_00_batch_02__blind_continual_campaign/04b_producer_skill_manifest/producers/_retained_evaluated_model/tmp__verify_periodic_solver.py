import pickle
import sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch, _PAY
from _verified_upstream import calibrate_upstream
from downstream_solver import _forward, _calibrate, _box_simplex, solve_downstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
raw = item['study']
s = calibrate_upstream(raw, _PAY)
S, U = _forward(s)
nx, ny = s['nx'], s['ny']

def apply(u, shift):
    result = (s['V_reaction_2d'] + shift) * u
    for axis, (name, n) in enumerate(zip('xy', (nx, ny))):
        h = s['L' + name] / (n-1)
        k, v = s['kappa_' + name], s['vel_' + name]
        a, b = np.moveaxis(u, axis, 0), np.moveaxis(result, axis, 0)
        for i in range(n):
            j, l = (i-1) % n, (i+1) % n
            b[i] += (k[i]+k[j])/(2*h*h)*(a[i]-a[j]) + (k[i]+k[l])/(2*h*h)*(a[i]-a[l])
            b[i] += v[i]/h*((a[i]-a[j]) if v[i] >= 0 else (a[l]-a[i]))
    return result

pulses = []
for pulse in item['campaign_diagnostics']:
    y = _calibrate(calibrate_upstream({**raw, 'y_obs_raw': pulse['y_pulse_raw']}, _PAY))
    error = (S @ pulse['w_pulse'] - y).reshape(len(s['mu_shifts']), -1, 4)
    pulses.append(np.sqrt(np.mean(error**2, axis=1)))
worst = float(np.max(pulses))
assert worst < 2e-5, pulses

out = solve_batch(raw)
assert np.array_equal(out['weights'], solve_downstream(s)['weights'])
w = out['weights']; cap, budget, ridge = (s[k] for k in ('u_box', 'total_budget', 'source_ridge'))
field_err = np.max(np.abs(apply(out['terminal_field'], s['mu_shifts'][-1]) - np.tensordot(w, s['source_modes'], axes=1)))
assert field_err < 1e-9, field_err
Z = s['Z_nuisance']; P = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
A, y = P @ S, P @ _calibrate(s)
r = A @ w - y; g = A.T @ r + ridge*w
c = out['kkt_certificate']; nu = c['nu_sum']; low, up = map(np.asarray, (c['lam_lower'], c['lam_upper']))
obj = (r @ r + ridge * (w @ w))/2
assert np.array_equal(w, c['weights']) and np.all(low >= 0) and np.all(up >= 0)
assert abs(w.sum()-budget) < 1e-10 and np.all((0 <= w) & (w <= cap))
assert abs(c['primal_objective']-obj) < 1e-12
assert abs(c['dual_lower_bound'] - (obj - g @ w + nu*budget - cap*up.sum())) < 1e-12
kkt = np.max(np.abs(np.r_[g-nu-low+up, low*w, up*(cap-w), obj-c['dual_lower_bound']]))
assert kkt < 1e-10, kkt

rng = np.random.default_rng(42)
qp_worst = 0.
for i in range(200):
    M = rng.normal(size=(8, 8)); H = M.T @ M + 1e-3 * np.eye(8)
    c = rng.normal(size=8) * 10**(i%4-1)
    b = [0., 1., 3., 3.37, 7.99, 8.][i%6]
    v, nu = _box_simplex(H, c, 1., b)
    grad = H @ v - c - nu
    err = np.max(np.abs(np.r_[v.sum()-b, np.maximum(grad, 0)*v, np.minimum(grad, 0)*(1-v)]))
    qp_worst = max(qp_worst, err)
    assert np.all((v >= 0) & (v <= 1)) and err < 1e-9, (i, err)
print(f'worst pulse/shift/channel RMSE={worst:.3e}; terminal PDE max residual={field_err:.3e}; public KKT max error={kkt:.3e}; worst stress KKT error={qp_worst:.3e} across 200 budget/active-face cases')
