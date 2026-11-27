import pickle
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _features, _fit, _design, _chamber_response
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
raw = item['dataset']
d = calibrate_upstream(raw)
g = calibrate_upstream(dict(raw, test=item['campaign_diagnostics']))['test']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
X, _ = _features(tr, te)
F, _ = _features(g, {k:v[:0] for k,v in g.items()})
labs = g['lab_id']
complete = np.isfinite(g['coupon_features_raw']).all(axis=1)
y, r = g['strain_curves_obs'], g['instant_rate_obs']
models = ['retained', 'no_history', 'linear_drift', 'pairwise', 'interaction', 'rational']
C = {m:np.empty_like(y) for m in models}
R = {m:np.empty_like(r) for m in models}
rmse = lambda x: float(np.sqrt(np.mean(x*x)))
phi = -np.expm1(-t)
for lab in np.unique(labs):
    train = tr['lab_id'] != lab
    tau, w = _fit({k:v[train] for k,v in tr.items()}, X[train], dt, t)
    c, rate = _chamber_response(g, F, dt, t, tau, w)
    c0, r0 = _chamber_response(dict(g, past_channel_raw=np.zeros_like(g['past_channel_raw'])), F, dt, t, tau, w)
    _, D = _design(g, F, dt, t, tau)
    b0, b1 = (D[:, :2] * w[:2]).T
    drift = D[:, 2:] @ w[2:]
    predictions = {'retained':(c,rate), 'no_history':(c0,r0), 'linear_drift':(c+drift[:,None]*(t-phi),rate)}
    # Perturb the drift-rate amplitude with a branch contrast, an interaction,
    # or rational suppression; apply the same amplitude to the initial rate.
    use = (labs != lab) & complete
    q = b0-b1
    q = q / np.sqrt(np.mean(q[use]**2))
    for name in ('pairwise', 'interaction', 'rational'):
        def delta(p):
            if name == 'pairwise': return p[0]*q
            if name == 'interaction': return p[0]*q*drift
            return drift/(1+p[0]*q*q)-drift
        def residual(p):
            a = delta(p)[use]
            return np.r_[((c[use]-y[use]+a[:,None]*phi)/.004).ravel(), (rate[use]-r[use]+a)/.002]
        fit = least_squares(residual, [0.], bounds=(0.,np.inf) if name == 'rational' else (-np.inf,np.inf), loss='soft_l1')
        a = delta(fit.x)
        predictions[name] = (c+a[:,None]*phi, rate+a)
    ix = labs == lab
    for name,(pc,pr) in predictions.items():
        C[name][ix], R[name][ix] = pc[ix],pr[ix]
for name in models:
    ec, er = C[name]-y, R[name]-r
    print(name, 'pooled', tuple(round(v,6) for v in (rmse(ec),rmse(er))), 'complete', tuple(round(v,6) for v in (rmse(ec[complete]),rmse(er[complete]))), 'worst lab', tuple(round(max(rmse(a[labs==lab]) for lab in np.unique(labs)),6) for a in (ec,er)), 'worst horizon', round(max(rmse(ec[:,j]) for j in range(len(t))),6))
    if name == 'retained':
        print('retained lab mean rate residual, complete vs all', [(int(lab), round(float(er[(labs==lab)&complete].mean()),6), round(float(er[labs==lab].mean()),6)) for lab in np.unique(labs)])
        print('retained missing-feature RMSE', rmse(ec[~complete]),rmse(er[~complete]))
assert all(rmse((R['retained']-r)[labs==lab]) < rmse((R['no_history']-r)[labs==lab]) for lab in np.unique(labs))
output = solve_batch(raw)
for k, shape in {'estimated_taus':(2,), 'hereditary_weights':(7,), 'predicted_strain':(len(te['lab_id']),len(t)), 'initial_rate_pred':(len(te['lab_id']),)}.items():
    assert output[k].shape == shape and np.isfinite(output[k]).all(), k
assert .05 < output['estimated_taus'][0] < output['estimated_taus'][1] < 25
print('Isolated entry point passed; no task files changed.')
