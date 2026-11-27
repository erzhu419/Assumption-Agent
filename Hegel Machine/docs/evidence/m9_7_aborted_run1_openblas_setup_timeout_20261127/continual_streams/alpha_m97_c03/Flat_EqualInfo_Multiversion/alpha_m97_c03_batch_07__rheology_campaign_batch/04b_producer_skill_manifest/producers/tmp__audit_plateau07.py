import importlib.util
import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p+'alpha_m97_c03_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
r = pickle.load(open(p+'public_alpha_m97_c03_batch_07.pkl', 'rb'))[0]
d, g = r['dataset'], r['gt']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
x, xt = s._align(tr, te)
lines = s._calibrate(d['calibration_load_cells'])
i = s._inputs(tr, x, lines, dt, d['crit_stress'], True)
y = np.c_[tr['strain_curves_obs']-tr['strain_0'][:, None], tr['instant_rate_obs']]
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
flow = np.r_[np.log1p(t), 1.]
# Pair differences are already in the linear coupon span. Check whether adding
# fixed interaction or saturating corrections improves unseen-lab residuals.
pairs = list(combinations(range(4), 2))
triples = list(combinations(range(4), 3))
extras = {
    'product': np.column_stack([x[:,j]*x[:,k] for j,k in pairs]+[np.prod(x[:,idx],axis=1) for idx in triples]),
    'saturating_difference': np.column_stack([(x[:,j]-x[:,k])/(1+(x[:,j]-x[:,k])**2) for j,k in pairs]),
}
for j,k in pairs:
    for sign in (-1,1):
        np.testing.assert_allclose((x[:,j]+sign*x[:,k])[:,None]*flow, x[:,j,None]*flow+sign*x[:,k,None]*flow, atol=1e-14)
base = np.empty_like(y)
challengers = {k:np.empty_like(y) for k in extras}
for lab in range(4):
    held = tr['lab_id']==lab
    fit = ~held
    b,w = s._fit(tuple(a[fit] for a in i),dt,t,y[fit],True,complete[fit])
    nuisance = y-s._design(b,i,dt,t,True)@w
    base[held] = nuisance[held]
    m = fit & complete
    valid = np.isfinite(nuisance[m])
    for name, f in extras.items():
        basis = f[:,None,:]*flow[None,:,None]
        a, target = basis[m][valid], nuisance[m][valid]
        scale = 1.4826*np.median(np.abs(target-np.median(target)))
        v = least_squares(lambda v:a@v-target,np.zeros(a.shape[-1]),loss='cauchy',f_scale=max(2.385*scale,1e-6)).x
        challengers[name][held] = nuisance[held]-basis[held]@v

def scores(e,m):
    return np.around([np.nanmedian(abs(e[m,:-1])),np.nanmedian(abs(e[m,-1]))],6)
for name, e in [('baseline',base),*challengers.items()]:
    print(name,'lab [curve/rate MedAE]',[scores(e,complete&(tr['lab_id']==k)).tolist() for k in range(4)],'damage slices',[scores(e,complete&((i[-1]>0)==k)).tolist() for k in (0,1)])
for lab in range(4):
    m = complete & (tr['lab_id']==lab)
    print('held lab',lab,'n',m.sum(),'signed mid/final/rate median',np.round(np.nanmedian(base[m][:,[len(t)//2,-2,-1]],axis=0),6))
out = s.solve_alpha_m97_c03_batch_07(d)
curve = out['predicted_strain']-g['te_clean']
rate = out['initial_rate_pred']-g['te_true_rate0']
full = np.isfinite(te['coupon_features_raw']).all(axis=1)
for m,name in [(full,'complete'),(~full,'imputed')]:
    # If the imputation gap explains the residual, the curve/rate error ratio
    # follows log1p(t) with no remaining history-dependent term.
    dev=curve[m]-rate[m,None]*np.log1p(t)
    print(name,'n',m.sum(),'curve/rate RMSE',np.sqrt(np.mean(curve[m]**2)),np.sqrt(np.mean(rate[m]**2)),'RMSE after drift component removed',np.sqrt(np.mean(dev**2)))
print('relative tau errors',np.abs(out['estimated_taus']/g['taus']-1))
