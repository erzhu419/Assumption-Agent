import sys
sys.path.insert(0, '/holdout_task')
import pickle
import numpy as np
import downstream_solver as m
from batch_solver import solve_batch
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
raw = item['dataset']
d = calibrate_upstream(raw)
out = solve_batch(raw)
for k, v in m.solve_downstream(d).items():
    np.testing.assert_array_equal(out[k], v)
a, b = d['train'], d['test']
diag = calibrate_upstream(dict(raw, test=item['campaign_diagnostics']))['test']
tau, theta = out['estimated_taus'], out['hereditary_weights']
assert .05 < tau[0] < tau[1] < 25
assert all(np.isfinite(v).all() for v in out.values())
dt, t = d['past_dt'], d['eval_times']
x = m._features(diag)
strain, rate = m._predict(diag, x, tau, theta, dt, t)
start, _ = m._predict(diag, x, tau, theta, dt, np.array([0., 1e-6]))
np.testing.assert_allclose(start[:, 0], diag['strain_0'], rtol=0, atol=1e-14)
np.testing.assert_allclose((start[:, 1]-start[:, 0])/1e-6, rate, rtol=2e-6, atol=1e-8)
e, er = strain-diag['strain_curves_obs'], rate-diag['instant_rate_obs']
k = np.r_[-np.expm1(-t), 1.]
E = np.c_[e, er]
orth = E - np.outer(E @ k / (k @ k), k)
labs = diag['lab_id']
complete = np.isfinite(diag['coupon_features_raw']).all(1)
def rms(v):
    return np.sqrt(np.mean(v*v))
print('diagnostic strain/rate/orthogonal RMSE by lab:', {int(l): [rms(e[labs==l]), rms(er[labs==l]), rms(orth[labs==l])] for l in np.unique(labs)})
print('complete/missing strain and rate RMSE:', [(rms(e[g]),rms(er[g])) for g in (complete,~complete)])
print('missing diagnostics rows [row, feature, local imputation, joint imputation]:')
x_joint = m._features(a, b, diag)[-len(x):]
for i, j in zip(*np.where(~np.isfinite(diag['coupon_features_raw']))):
    print(i, j, round(x[i,j],5), round(x_joint[i,j],5))
print('diagnostic strain early/late RMSE:',rms(e[:,:3]),rms(e[:,-3:]))
xt = m._features(a,b)[:len(a['lab_id'])]
cs, rs = m._design(tau, m._inputs(a,xt), dt, t)
obs = a['strain_curves_obs']-a['strain_0'][:,None]
print('train median absolute strain/rate residuals:',np.median(np.abs(cs@theta-obs)),np.median(np.abs(rs@theta-a['instant_rate_obs'])))
