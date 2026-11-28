import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, v = item['dataset'], item['campaign_diagnostics']
o = m.solve_batch(d)
tau, theta, t = o['estimated_taus'], o['hereditary_weights'], d['eval_times']
cal = m._calibration(d['calibration_load_cells'])
x = m._features(v, v)[:len(v['lab_id'])]
past, hold, cov = m._inputs(v, x, cal)
g = cov @ theta[2:]
state = m._design(tau, (past, np.zeros_like(hold), cov), d['past_dt'], t)[1][:,:2] * -tau
A = np.vstack((-np.expm1(-t[:,None]/tau), 1/tau))
L = np.r_[t, 1.]
y = np.c_[v['strain_curves_obs']-v['strain_0'][:,None], v['instant_rate_obs']]
base = (hold[:,None]*theta[:2]) @ A.T
response = (theta[:2]*tau) @ A.T
lab = v['lab_id']
complete = ~np.isnan(v['coupon_features_raw']).any(axis=1)

def predict(p, mode):
    drive = g
    if mode == 'saturation':
        drive = g/(1+p[2]*g*g)
    out = base + (p[0]*drive)[:,None]*response + (p[1]*g)[:,None]*L
    if mode == 'pulse difference':
        out += (p[2]*(state[:,0]-state[:,1]))[:,None]*response
    return out

preds = {'artifact': np.column_stack(m._chamber(tau, theta, hold, x, t))}
for mode in ('reset', 'saturation', 'pulse difference'):
    pred = np.empty_like(y)
    for j in np.unique(lab):
        keep = (lab != j) & complete
        p0 = [.7,.065] + ([] if mode == 'reset' else [0.])
        bounds = ([0,0,0],[np.inf]*3) if mode == 'saturation' else (-np.inf,np.inf)
        p = least_squares(lambda z: (predict(z,mode)[keep]-y[keep]).ravel(),p0,bounds=bounds,loss='soft_l1',f_scale=.009).x
        pred[lab==j] = predict(p,mode)[lab==j]
    preds[mode] = pred
for name, pred in preds.items():
    for label, mask in [('complete',complete),('missing',~complete)]:
        sub = [(pred-y)[mask & (lab==j)] for j in np.unique(lab[mask])]
        worst = np.max([[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))] for e in sub],axis=0)
        bias = max(abs(np.mean(e,axis=0)).max() for e in sub)
        print(name,label,'worst_lab_rmse_strain_rate',worst,'max_lab_time_bias',round(bias,6))

c = d['calibration_load_cells']
raw, loads = c['bridge_raw'], c['true_channel_loads'].sum(axis=1)
# Hold the highest and lowest applied loads out of the bridge fit.
order = np.argsort(loads)
k = max(1,len(order)//10)
center = order[k:-k]
beta = m._calibration({key:c[key][center] for key in ('bridge_raw','true_channel_loads')})
for label, ix in [('low',order[:k]),('high',order[-k:])]:
    e = raw[ix] @ beta[:4] + beta[4] - loads[ix]
    print('bridge',label,'RMSE',np.sqrt(np.mean(e**2)),'max',abs(e).max())
