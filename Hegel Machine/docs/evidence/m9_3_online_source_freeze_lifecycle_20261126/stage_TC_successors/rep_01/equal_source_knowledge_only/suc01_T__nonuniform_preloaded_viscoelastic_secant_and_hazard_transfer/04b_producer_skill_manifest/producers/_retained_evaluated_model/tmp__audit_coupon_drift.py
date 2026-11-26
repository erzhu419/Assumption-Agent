import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'preloaded_hereditary_secant_hazard_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open(root + 'public_suc01_T_datasets.pkl', 'rb'))[0]['dataset']
a = d['train']
dt, t = d['past_dt'], d['eval_times']
lab, stratum = a['lab_id'], a['stratum']
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)
_, cell = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
fold = np.zeros(len(lab), dtype=int)
rng = np.random.default_rng(28)
for j in np.unique(cell):
    rows = rng.permutation(np.flatnonzero(cell == j))
    fold[rows] = np.arange(len(rows)) % 4

def late_residual(block, design, beta):
    late = slice(len(t)//2, None)
    return np.median((block['strain_curves_obs'][:, late] - block['strain_0'][:, None] - (design @ beta)[:, late]) / t[late], axis=1)

observed = np.empty(len(lab))
pred = np.empty((len(lab), 3))
for f in np.unique(fold):
    fit = fold != f
    tr, va = ({k: v[mask] for k, v in a.items()} for mask in (fit, ~fit))
    x, xv, w = s._features(tr, va)
    tau, b = s._fit(tr, dt, t, x, w)
    curve, rate = s._design(tr, dt, t, x, tau)
    curve_v, rate_v = s._design(va, dt, t, xv, tau)
    delta = late_residual(tr, curve, b)
    err = tr['instant_rate_obs'] - rate @ b - delta
    innovation = va['instant_rate_obs'] - rate_v @ b
    base = s._rate_update(tr, va, dt, t, x, tau, b, rate_v)
    observed[~fit] = late_residual(va, curve_v, b)
    pred[~fit, 0] = base
    for j in np.unique(lab):
        cal = complete[fit] & (lab[fit] == j)
        sel = lab[~fit] == j
        center = np.median(delta[cal])
        scale = lambda v: 1.4826*np.median(np.abs(v - np.median(v)))
        drift_sd, noise_sd = scale(delta[cal]), scale(err[cal])
        gain = drift_sd**2 / (drift_sd**2 + noise_sd**2)
        has_panel = complete[~fit][sel]
        pred[np.flatnonzero(~fit)[sel], 1] = np.where(has_panel, center + gain*(innovation[sel]-center), base[sel])
        pred[np.flatnonzero(~fit)[sel], 2] = np.where(has_panel, innovation[sel], base[sel])
        print('fold/lab', f, int(j), 'center/drift SD/rate noise SD', np.round([center, drift_sd, noise_sd], 5))
error = np.abs(pred - observed[:, None])
print('OOF late-slope MAE: current / lab-conditioned prior / rate only for complete coupons')
print('pooled', error.mean(0), 'worst cell', np.max([error[cell == j].mean(0) for j in np.unique(cell)], axis=0))
for j in np.unique(lab):
    m = (lab == j) & complete
    print('complete lab', int(j), 'MAE', error[m].mean(0), 'current signed residual median', np.median((observed-pred[:, 0])[m]))
