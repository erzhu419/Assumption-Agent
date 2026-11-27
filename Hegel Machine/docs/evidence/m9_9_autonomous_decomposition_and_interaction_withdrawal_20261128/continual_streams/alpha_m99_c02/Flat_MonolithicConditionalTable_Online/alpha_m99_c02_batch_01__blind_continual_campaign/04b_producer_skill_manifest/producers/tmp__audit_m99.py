import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['campaign_diagnostics']
t, dt = d['eval_times'], d['past_dt']
bridge = m._bridge(d['calibration_load_cells'])
out = m.solve_batch(d)
tau, theta = out['estimated_taus'], out['hereditary_weights']
assert .05 < tau[0] < tau[1] < 25
x = m._features(g, g)[0]
past, hold = m._loads(g, bridge)
lab = g['lab_id']
missing = np.isnan(g['coupon_features_raw']).any(axis=1)
drift = theta[2] + x @ theta[3:]
base_c, base_r = m._chamber(tau, theta, hold, x, t, route=np.zeros(3))
y = np.c_[g['strain_curves_obs'] - g['strain_0'][:, None] - base_c, g['instant_rate_obs'] - base_r]
T = np.vstack((np.c_[-np.expm1(-t[:, None] / tau), t], np.r_[1/tau, 1]))
def fit(F, keep):
    A = (F[:, None, :, None] * T[None, :, None, :]).reshape(len(lab), len(T), -1)
    a, b = A[keep].reshape(-1, A.shape[-1]), y[keep].ravel()
    start = np.linalg.lstsq(a, b, rcond=None)[0]
    p = least_squares(lambda v: a @ v - b, start, jac=lambda v: a, loss='soft_l1', f_scale=.004).x
    return A @ p

def score(e):
    return np.array([np.sqrt(np.mean(e[:, :-1]**2)), np.sqrt(np.mean(e[:, -1]**2))])

def grammar(kind, fit_rows):
    if kind == 'linear':
        return [drift[:, None]]
    if kind == 'rational':
        qs = [drift, hold-past[:, -1], *(x[:, j] + s*x[:, k] for j in range(4) for k in range(j+1, 4) for s in [-1, 1])]
        return [np.column_stack((drift, drift / (1 + a*(q/max(np.std(q[fit_rows]), 1e-10))**2))) for q in qs for a in [.1, 1.]]
    qs = [*(x[:, j] + s*x[:, k] for j in range(4) for k in range(j+1, 4) for s in [-1, 1]), *(hold-past[:, j] for j in range(past.shape[1]))]
    return [np.c_[drift, f] for q in qs for f in [q**2, q*drift, q*drift*hold]]

for kind in ('linear', 'rational', 'difference_interaction'):
    e = np.zeros_like(y)
    for outer in np.unique(lab):
        keep = lab != outer
        candidates = grammar(kind, keep)
        if len(candidates) > 1:
            scores = []
            for j in range(len(candidates)):
                inner_errors = []
                for inner in np.unique(lab[keep]):
                    it, iv = keep & (lab != inner), lab == inner
                    pred = fit(grammar(kind, it)[j], it)
                    inner_errors.append(score(pred[iv] - y[iv]))
                scores.append(np.max(inner_errors))
            A = candidates[np.argmin(scores)]
        else:
            A = candidates[0]
        e[~keep] = (fit(A, keep) - y)[~keep]
    print(kind, 'worst lab RMSE', np.max([score(e[lab == k]) for k in np.unique(lab)], axis=0), 'complete / imputed RMSE', score(e[~missing]), score(e[missing]))
print('isolated public solution shapes', {k: np.asarray(v).shape for k, v in out.items()})
