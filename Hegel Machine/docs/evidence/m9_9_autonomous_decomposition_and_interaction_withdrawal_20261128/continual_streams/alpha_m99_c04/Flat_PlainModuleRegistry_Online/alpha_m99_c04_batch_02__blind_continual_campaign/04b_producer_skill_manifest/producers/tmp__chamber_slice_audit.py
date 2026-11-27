import importlib.util
import pickle
from itertools import combinations
import numpy as np
from scipy.optimize import minimize_scalar

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']
q = item['campaign_diagnostics']
tr, te = d['train'], d['test']
t, dt = d['eval_times'], d['past_dt']
x, _ = m._features(tr, te)
s = m._stress_map(d['calibration_load_cells'])
tau, theta = m._fit(tr, x, s, dt, t)
# The diagnostics are a separate reference panel. Use complete coupons to
# distinguish an omitted chamber term from differences in imputation.
xq, _ = m._features(q, q)
complete = np.isfinite(q['coupon_features_raw']).all(axis=1)
lab = q['lab_id']
c, a = m._design(q, xq, s, dt, t, tau)
physical = q['strain_0'][:, None] + c[..., :2] @ theta[:2]
drift = theta[2] + xq @ theta[3:]
y = q['strain_curves_obs'] - physical
rate_error = q['instant_rate_obs'] - a @ theta
u = -np.expm1(-t)
baseline = drift[:, None] * u
h = s[0] + q['hold_channel_raw'] @ s[1:]
g0, g1 = (a[:, :2] * tau).T
basis = {}
for i,j in combinations(range(4), 2):
    f,g = xq[:, i], xq[:, j]
    for sign in (-1,1):
        z = f + sign*g
        basis[f'x{i}{sign:+}x{j}'] = z
        basis[f'rat_x{i}{sign:+}x{j}'] = z/(1+z*z)
    basis[f'x{i}*x{j}'] = f*g
for i,j,k in combinations(range(4),3):
    basis[f'x{i}*x{j}*x{k}'] = xq[:,i]*xq[:,j]*xq[:,k]
basis.update(gap_difference=g0-g1, gap_product=g0*g1, rational_gap=(g0-g1)/(1+(g0+g1)**2), load_gap_product=h*(g0-g1), load_gap_triple=h*g0*g1)
# u^2 permits a nonlinear chamber contribution without changing the
# initial-rate law. Select the term using only inner training labs.
terms = {k:v[:, None]*u**2 for k,v in basis.items()}
resid = y - baseline
rmse = lambda z: float(np.sqrt(np.mean(z*z)))
def coef(v, mask):
    return np.sum(v[mask]*resid[mask])/np.sum(v[mask]**2)
all_errors = [[], [], []]
for group in np.unique(lab[complete]):
    fit = complete & (lab != group)
    val = complete & (lab == group)
    fitlabs = np.unique(lab[fit])
    def inner(v):
        return rmse(np.concatenate([(resid-coef(v,fit & (lab!=g))*v)[fit & (lab==g)] for g in fitlabs]))
    name = min(terms, key=lambda k:inner(terms[k]))
    v = terms[name]
    def free(tt):
        return drift[:,None]*(-tt*np.expm1(-t/tt))
    opt = minimize_scalar(lambda tt: rmse((y-free(tt))[fit]), bounds=(.3,3), method='bounded')
    errs = [resid[val], (y-free(opt.x))[val], (resid-coef(v,fit)*v)[val]]
    for j,e in enumerate(errs): all_errors[j].append(e)
    print(f'lab={group} n={val.sum()} strain RMSE fixed/free/sparse={tuple(round(rmse(e),6) for e in errs)} rate RMSE={rmse(rate_error[val]):.6f} rate mean={rate_error[val].mean():+.6f} free_tau={opt.x:.4f} term={name}')
print('pooled complete strain RMSE fixed/free/sparse:', [round(rmse(np.concatenate(e)),6) for e in all_errors])
print('missing diagnostic n, strain and rate RMSE:', int((~complete).sum()), round(rmse(resid[~complete]),6), round(rmse(rate_error[~complete]),6))
