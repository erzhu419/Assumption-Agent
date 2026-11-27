import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from _verified_upstream import calibrate_upstream
from downstream_solver import _features, _fit, _design

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
raw = item['dataset']
d = calibrate_upstream(raw)
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
lab = tr['lab_id']
x, xt = _features(tr, te)
out = solve_batch(raw)
assert .05 < out['estimated_taus'][0] < out['estimated_taus'][1] < 25
for key, shape in {'estimated_taus': (2,), 'hereditary_weights': (7,), 'predicted_strain': (len(xt), len(t)), 'initial_rate_pred': (len(xt),)}.items():
    assert out[key].shape == shape and np.isfinite(out[key]).all(), key

def take(mask):
    return {k: v[mask] for k, v in tr.items()}

def residual(g, z, tau, theta):
    c, r = _design(g, z, dt, t, tau)
    return np.column_stack([g['strain_curves_obs'] - g['strain_0'][:, None] - c @ theta, g['instant_rate_obs'] - r @ theta])

aux = tr['aux_chamber_telemetry']
# Candidate correction bases are fixed before cross-validation. Fit only
# on other labs; evaluate each horizon and the instantaneous rate separately.
p, q = aux[:, :2], aux[:, 2:]
v = p - q
bases = [np.zeros((len(x), 0)), aux, np.c_[aux, v[:, 0]*v[:, 1], p[:, 0]*q[:, 0]*v[:, 1], v/(1+v*v)]]
rows = [[] for _ in bases]
for j in np.unique(lab):
    keep = lab != j
    fit, hold = take(keep), take(~keep)
    tau, theta = _fit(fit, x[keep], dt, t)
    e_fit = residual(fit, x[keep], tau, theta)
    e_hold = residual(hold, x[~keep], tau, theta)
    for b, scores in zip(bases, rows):
        # Median-scale clipping prevents residual outliers from driving the nuisance fit.
        scale = np.maximum(1.4826*np.median(abs(e_fit - np.median(e_fit, axis=0)), axis=0), 1e-8)
        bfit = b[keep] - b[keep].mean(axis=0)
        bh = b[~keep] - b[keep].mean(axis=0)
        w = np.linalg.lstsq(bfit, np.clip(e_fit, -3*scale, 3*scale), rcond=None)[0]
        err = e_hold - bh @ w
        scores.append([np.quantile(abs(err[:, :-1]), .9), np.max(np.quantile(abs(err[:, :-1]), .9, axis=0)), np.sqrt(np.mean(err[:, -1]**2))])
for name, scores in zip(('material only','linear telemetry','linear + pairwise/product/rational telemetry'), rows):
    print(name, 'worst lab [curve p90, worst horizon p90, rate RMSE]:', np.max(scores, axis=0))

# Finite differences over the actual coupons catch disagreement between
# the reported initial rate and the deployment curve at t=0.
h = 1e-5
a, b = out['estimated_taus'], out['hereditary_weights']
c, _ = _design(te, xt, dt, np.array([0., h, 2*h]), a)
y = c @ b
err = np.max(abs((-3*y[:,0]+4*y[:,1]-y[:,2])/(2*h) - out['initial_rate_pred']))
print('maximum initial-rate finite-difference error:', err)
assert err < 1e-7
