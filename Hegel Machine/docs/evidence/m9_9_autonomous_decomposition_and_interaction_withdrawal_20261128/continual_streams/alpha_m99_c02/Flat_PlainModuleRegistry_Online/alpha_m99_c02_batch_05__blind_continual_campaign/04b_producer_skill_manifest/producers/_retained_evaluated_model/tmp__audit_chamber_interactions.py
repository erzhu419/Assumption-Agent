import pickle
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _basis, _deployment_features
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']
fit = solve_batch(d)
cal = calibrate_upstream(dict(d, chamber_verification_trials=item['campaign_diagnostics']))
q = cal['chamber_verification_trials']
t = cal['eval_times']
tau, theta = fit['estimated_taus'], fit['hereditary_weights']
x = _deployment_features(q)
u = theta[2] + x @ theta[3:]
past, hold = q['past_channel_raw'].sum(-1), q['hold_channel_raw'].sum(-1)
curve_basis, rate_basis = _basis(tau, past, hold, x, cal['past_dt'], t)
relax = -np.expm1(-t[:, None] / tau)
amp = hold[:, None] * theta[:2]
base_c = amp @ relax.T
base_r = amp @ (1 / tau)
memory_c = curve_basis[:, :, :2] @ theta[:2] - base_c
memory_r = rate_basis[:, :2] @ theta[:2] - base_r
obs_c = q['strain_curves_obs'] - q['strain_0'][:, None]
obs_r = q['instant_rate_obs']
lab = q['lab_id']
missing = np.isnan(q['coupon_features_raw']).any(axis=1)
slices = [missing, ~missing] + [lab == g for g in np.unique(lab)]

def predict(kind, v):
    f, s = v[:2]
    kernel = f * tau[0] * relax[:, 0] - (1-f) * s * np.expm1(-t/s)
    drift = u
    mix = 0.
    if kind == 'memory':
        mix = v[2]
    elif kind == 'rational':
        drift = u / (1 + v[2] * hold**2)
    elif kind == 'pairwise':
        drift = u + v[2] * (hold - past[:, -1])
    elif kind == 'product':
        drift = u + v[2] * u * hold
    return base_c + mix * memory_c + drift[:, None] * kernel, base_r + mix * memory_r + drift

def errors(kind, v):
    c, r = predict(kind, v)
    return c-obs_c, r-obs_r

def summary(c, r):
    return np.array([max(np.quantile(np.abs(c[m, -3:]), .9) for m in slices), max(np.quantile(np.abs(r[m]), .9) for m in slices)])

for kind in ('reset', 'memory', 'rational', 'pairwise', 'product'):
    init, lo, hi = [.26, 7.6], [0, .05], [1, 50]
    if kind != 'reset':
        init.append(0.); lo.append(0. if kind == 'rational' else -1.); hi.append(1.)
    c = np.empty_like(obs_c); r = np.empty_like(obs_r)
    for group in np.unique(lab):
        m = lab != group
        def residual(v):
            ec, er = errors(kind, v)
            return np.r_[ec[m].ravel() / np.sqrt(t.size), er[m]]
        v = least_squares(residual, init, bounds=(lo, hi), loss='soft_l1', f_scale=.003).x
        ec, er = errors(kind, v)
        c[~m], r[~m] = ec[~m], er[~m]
    print(kind, 'leave-lab-out worst-slice p90 (late strain, rate):', summary(c, r).round(6))
print('Schema outputs finite:', {k: (v.shape, bool(np.isfinite(v).all())) for k,v in fit.items()})
