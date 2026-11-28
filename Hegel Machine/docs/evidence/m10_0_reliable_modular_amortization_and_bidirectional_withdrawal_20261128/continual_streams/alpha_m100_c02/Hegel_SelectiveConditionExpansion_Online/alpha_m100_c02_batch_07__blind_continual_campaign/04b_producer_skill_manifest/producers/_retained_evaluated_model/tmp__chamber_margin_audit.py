import json
import pickle
import sys
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, '/holdout_task')
import downstream_solver as s
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
with open('/holdout_task/_upstream_param_payload.json') as f:
    payload = json.load(f)
raw = item['dataset']
d = calibrate_upstream({**raw, 'test': item['campaign_diagnostics']}, payload)['test']
dt, t, _, tau, theta = s._intrinsic(calibrate_upstream(raw))
past, hold = s._stress(d)
x = s._features(d, d)[:len(hold)]
origin, y, r = (d[k] for k in ('strain_0', 'strain_curves_obs', 'instant_rate_obs'))
ref, ref_rate = s._chamber_response(past, hold, x, origin, dt, t, tau, theta, np.zeros(2))
Y = np.column_stack([y-ref, r-ref_rate])
K = np.vstack([-np.expm1(-t[:, None]/tau)*tau-t[:, None], [0., 0.]])
L = np.r_[t, 1.]
drift = x @ theta[2:]
core = drift[:, None, None]*K[None, :, :]
# The chamber verification coupons have no auxiliary telemetry. Use measured
# load summaries to test whether the drift correction depends on load regime.
tele = np.column_stack([hold, past[:, -1], past.mean(axis=1), past.std(axis=1)])

def fixed_covariates(sel, centered):
    if centered:
        tele_fit = tele[sel]
        labs_fit = d['lab_id'][sel]
        pooled = tele_fit.mean(0)
        center = np.vstack([tele_fit[labs_fit == lab].mean(0) if np.any(labs_fit == lab) else pooled for lab in d['lab_id']])
    else:
        center = tele[sel].mean(0)
    u = tele - center
    return u / np.maximum(u[sel].std(0), 1e-8)

def evaluate(name, sel):
    A = core
    if name in ('linear_load', 'lab_centered_load', 'nonlinear_load'):
        u = fixed_covariates(sel, name == 'lab_centered_load')
        v = u
        if name == 'nonlinear_load':
            pairs = list(combinations(range(u.shape[1]), 2))
            triples = list(combinations(range(u.shape[1]), 3))
            v = np.column_stack([u, *[u[:, i]*u[:, j] for i,j in pairs], *[u[:, i]*u[:, j]*u[:, k] for i,j,k in triples], *[(u[:, i]-u[:, j])/(1+(u[:, i]+u[:, j])**2) for i,j in pairs]])
        A = np.concatenate([A, (core[:, :, :, None]*v[:, None, None, :]).reshape(len(hold), len(t)+1, -1)], axis=2)
    if name == 'free_drift':
        A = np.concatenate([A, (x[:, None, :, None]*K[None, :, None, :]).reshape(len(hold), len(t)+1, -1), x[:, None, :] * L[None, :, None]], axis=2)
    B, target = A[sel].reshape(-1, A.shape[-1]), Y[sel].ravel()
    reg = np.r_[0., 0., np.full(A.shape[-1]-2, .03)]
    q0 = np.r_[s._CHAMBER_FRACTIONS, np.zeros(A.shape[-1]-2)]
    q = least_squares(lambda w: np.r_[B@w-target, reg*w], q0, loss='soft_l1', f_scale=.004).x
    return Y[~sel] - A[~sel]@q

print('diagnostic load-summary std:', tele.std(0))
for key in ('lab_id', 'stratum'):
    g = d[key]
    for name in ('constant', 'linear_load', 'lab_centered_load', 'nonlinear_load', 'free_drift'):
        residual = np.empty_like(Y)
        scores = []
        for label in np.unique(g):
            held = g == label
            e = evaluate(name, ~held)
            residual[held] = e
            scores.append([np.median(abs(e[:, :-1])), np.median(abs(e[:, -1]))])
        print(key, name, 'worst group [strain, rate]', np.max(scores, axis=0), 'worst time strain', np.max(np.median(abs(residual[:, :-1]), axis=0)))

from batch_solver import solve_batch
out = solve_batch(raw)
assert all(np.isfinite(v).all() for v in out.values())
for eps in (1e-5, 1e-6):
    curve, rate = s._chamber_response(past, hold, x, origin, dt, np.array([0., eps, 2*eps]), tau, theta, s._CHAMBER_FRACTIONS)
    np.testing.assert_allclose(curve[:,0], origin, atol=1e-12, rtol=0)
    error = np.max(abs((-3*curve[:,0]+4*curve[:,1]-curve[:,2])/(2*eps) - rate))
    print('zero-time derivative max error', eps, error)
    assert error < 1e-7
