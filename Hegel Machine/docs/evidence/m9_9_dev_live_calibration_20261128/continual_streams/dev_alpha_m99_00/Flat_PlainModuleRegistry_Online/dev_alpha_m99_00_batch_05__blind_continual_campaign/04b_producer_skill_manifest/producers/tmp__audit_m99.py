import sys
sys.path.insert(0, '/holdout_task')
import pickle
import numpy as np
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
from downstream_solver import _features, _predict_chamber
from batch_solver import solve_batch
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
raw = item['dataset']
d = calibrate_upstream(dict(raw, chamber_verification_trials=item['campaign_diagnostics']))
q = d['chamber_verification_trials']
t = np.asarray(d['eval_times'])
out = solve_batch(raw)
tau, th = out['estimated_taus'], out['hereditary_weights']
x = _features(q)
y, rate = q['strain_curves_obs'], q['instant_rate_obs']
yp, rp = _predict_chamber(tau, th, q, x, t)
hold = q['hold_channel_raw'].sum(-1)
past = q['past_channel_raw'].sum(-1)
state = np.zeros((len(hold), 2))
for stress, dt in zip(past.T, d['past_dt']):
    state += -np.expm1(-dt/tau) * (stress[:,None] - state)
K = -np.expm1(-t[:,None]/tau)
mem = q['strain_0'][:,None] + hold[:,None] * (K @ th[:2])
r = rate - hold * (th[:2] @ (1/tau))
E = y - mem
labs = q['lab_id']
rmse = lambda x: np.sqrt(np.mean(x*x))
g = 2*t/(np.sqrt(1+t)+1)
kernels = {
    'unit sqrt': (lambda p: g, []),
    'archived log': (lambda p: np.log1p(t), []),
    'free sqrt scale': (lambda p: 2*t/(np.sqrt(1+p[0]*t)+1), [1.]),
    'quadratic rational': (lambda p: t/(1+p[0]*t*t), [.01]),
    'saturating rational': (lambda p: t/(1+p[0]*t), [.1]),
}
for name,(fn,init) in kernels.items():
    errors = np.empty_like(E)
    for lab in np.unique(labs):
        m = labs == lab
        p = least_squares(lambda p: (r[~m,None]*fn(p)-E[~m]).ravel(),init,bounds=(0.,np.inf)).x if init else []
        errors[m] = r[m,None]*fn(p)-E[m]
    print(name, 'kernel-only LOLO strain RMSE', round(rmse(errors),6), 'worst lab/horizon', round(max(np.sqrt(np.mean(errors[labs==l]**2,axis=0)).max() for l in np.unique(labs)),6))
base = E - r[:,None]*g
# At fixed measured initial rate, retained branch memory must change both the
# branch displacement and the drift amplitude recovered from that initial rate.
branch = state[:,None,:]*th[:2]*(g[:,None]/tau-K)
contrast = q['hold_channel_raw'][:,0]-q['hold_channel_raw'][:,1]
pair = contrast[:,None]*(K[:,0]-tau[1]/tau[0]*K[:,1])
for name, terms in [('retained branches',branch),('channel contrast',pair[:,:,None])]:
    err = np.empty_like(base)
    for l in np.unique(labs):
        m=labs==l
        f=terms[~m].reshape(-1,terms.shape[-1])
        coef=np.linalg.lstsq(f,base[~m].ravel(),rcond=None)[0]
        err[m]=base[m]-terms[m]@coef
    print(name, 'kernel-only LOLO strain RMSE',round(rmse(err),6),'worst lab/horizon',round(max(np.sqrt(np.mean(err[labs==l]**2,axis=0)).max() for l in np.unique(labs)),6))
full = yp-y
print('full prediction pooled/worst lab-horizon strain RMSE',rmse(full),max(np.sqrt(np.mean(full[labs==l]**2,axis=0)).max() for l in np.unique(labs)))
print('full prediction pooled/worst lab initial-rate RMSE',rmse(rp-rate),max(rmse((rp-rate)[labs==l]) for l in np.unique(labs)))
miss=np.isnan(q['coupon_features_raw']).any(axis=1)
for label,m in [('missing',miss),('complete',~miss)]:
    if m.any(): print(label,int(m.sum()),'strain/rate RMSE',rmse(full[m]),rmse((rp-rate)[m]))
print('retained versus reset initial-rate RMSE',rmse(rp-state@(th[:2]/tau)-rate),rmse(rp-rate))
