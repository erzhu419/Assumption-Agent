import importlib.util
import pickle
from itertools import combinations, combinations_with_replacement
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(p+'public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
ds, d = item['dataset'], item['campaign_diagnostics']
o = m.solve_batch(ds)
tau, th = o['estimated_taus'], o['hereditary_weights']
print('material taus and weights:', tau, th)
cal = ds['calibration_load_cells']
raw, true = cal['bridge_raw'], cal['true_channel_loads']
A = np.c_[np.ones(len(raw)), raw]
v = true.sum(1)
order = np.argsort(v)
for label, held in [('low-load tail', order[:len(v)//5]), ('high-load tail', order[-(len(v)//5):])]:
    keep = np.ones(len(v), bool); keep[held] = False
    e = A[held] @ np.linalg.lstsq(A[keep], v[keep], rcond=None)[0] - v[held]
    print(label, 'axial RMS', np.sqrt(np.mean(e**2)))

br = m._calibration(cal)
dt, t = ds['past_dt'], ds['eval_times']
x = m._features(d,d)[:len(d['lab_id'])]
coupons = m._coupon_inputs(d,x,br)
past, hold, X = coupons
C,R = m._design(tau,coupons,dt,t)
curve, rate = m._chamber(tau,th,coupons,dt,t)
e = d['strain_curves_obs'] - d['strain_0'][:,None] - curve
er = d['instant_rate_obs'] - rate
G = -np.expm1(-t)
y = (e@G + er)/(G@G+1)
missing = np.isnan(d['coupon_features_raw']).any(1)
lab = d['lab_id']
axes = np.column_stack((X@th[2:],x,hold,hold[:,None]-R[:,:2]*tau))
cols = [*axes.T]
cols += [axes[:,i]+s*axes[:,j] for i,j in combinations(range(axes.shape[1]),2) for s in (-1,1)]
cols += [axes[:,i]*axes[:,j] for i,j in combinations_with_replacement(range(axes.shape[1]),2)]
cols += [axes[:,0]*axes[:,i]*axes[:,j] for i,j in combinations_with_replacement(range(1,axes.shape[1]),2)]
cols += [axes[:,0]/(1+a*q*q) for a in (.1,1.,10.) for q in axes[:,1:].T]
# Fit one correction at a time; restrict selection to inner leave-lab-out folds.
W = np.column_stack(cols)
def fit(a,b):
    V = W[a]
    denom = np.sum(V*V,axis=0)
    return W[b]*np.divide(V.T@y[a],denom,out=np.zeros_like(denom),where=denom>0)
pred = np.zeros(len(y))
for group in np.unique(lab):
    outer = lab != group
    folds = [lab == g for g in np.unique(lab[outer])]
    score = np.vstack([np.mean((fit(outer & ~val,val)-y[val,None])**2,axis=0) for val in folds]).mean(0)
    zero = np.mean([np.mean(y[val]**2) for val in folds])
    j = int(np.argmin(score))
    if score[j] < zero:
        pred[~outer] = fit(outer,~outer)[:,j]
    print('outer lab',group,'selected correction', j if score[j]<zero else 'none')

def rms(a): return np.sqrt(np.mean(a*a))
for label, group in [('pooled',np.ones(len(y),bool)),('complete',~missing),('missing',missing),*[(f'lab {g}',lab==g) for g in np.unique(lab)]]:
    print(label, 'n', group.sum(), 'fixed curve/rate RMS', rms(e[group]), rms(er[group]), 'nested challenger', rms((e-pred[:,None]*G)[group]), rms((er-pred)[group]))
