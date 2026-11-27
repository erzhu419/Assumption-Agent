import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'alpha_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
x = m._align(tr, te)
cal = m._calibration(d['calibration_load_cells'])
ti = m._inputs(d, tr, x[:n], cal)
vi = m._inputs(d, te, x[n:], cal)
y = np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None], tr['instant_rate_obs']]
complete = np.isfinite(tr['coupon_features_raw']).all(1)
def subset(inp, keep):
    return tuple(v[keep] if j in (0,1,2,5,6) else v for j,v in enumerate(inp))

out = m.solve_alpha_m97_c03_batch_01(d)
tau, w, rw = (out[k] for k in ('estimated_taus','hereditary_weights','regime_mode_weights'))
stress, hold, feat, dt, times, active, _, campaign = vi
memory = np.zeros((len(hold),2))
damage = np.zeros(len(hold))
for step, step_dt in enumerate(dt):
    a = np.exp(-step_dt/tau)
    memory = a*memory + (1-a)*stress[:,step,None]
    damage += np.maximum(0, abs(stress[:,step])-d['crit_stress'])**2*step_dt
activation = np.tanh(damage)*(damage>0)*campaign
weights = (hold[:,None]-memory)*w[:2]
weights[:,0] += rw[0]*activation*hold
terminal_weight = w[2]+feat@w[3:]+rw[1]*activation*memory[:,0]
clock = np.log1p(times) if campaign else times
curve = te['strain_0'][:,None] + weights@(-np.expm1(-times[None,:]/tau[:,None])) + terminal_weight[:,None]*clock
rate = weights@(1/tau) + terminal_weight
for label, keep in [('active',active),('inactive',~active)]:
    if keep.any():
        e = np.max(abs(curve[keep]-out['predicted_strain'][keep])); r = np.max(abs(rate[keep]-out['initial_rate_pred'][keep]))
        print('independent recurrence and rate',label,keep.sum(),e,r)
        assert max(e,r) < 1e-10

# Keep the mandated panels and calibrations available, but hold out all response labels from each lab.
e = np.empty_like(y)
for lab in np.unique(tr['lab_id']):
    test = tr['lab_id']==lab
    fit = ~test & complete
    q = m._fit(subset(ti,fit), y[fit])
    e[test] = m._design(q[:2],*subset(ti,test))[0]@q[2:]-y[test]
for lab, stratum in np.unique(np.c_[tr['lab_id'],tr['stratum']],axis=0):
    k = complete & (tr['lab_id']==lab) & (tr['stratum']==stratum)
    if k.any():
        print('OOF complete cell',lab,stratum,k.sum(),'signed curve-end/rate medians',np.round(np.median(e[k][:,[-2,-1]],axis=0),5),'absolute curve/rate medians',round(np.median(abs(e[k,:-1])),5),round(np.median(abs(e[k,-1])),5))
curve_e = e[complete,:-1]
print('OOF complete abs curve error quantiles by time',np.round(np.quantile(abs(curve_e),[.5,.9,.95,.99],axis=0),4))
print('OOF complete gross residual counts per coupon',tuple(v.tolist() for v in np.unique((abs(curve_e)>.25).sum(1),return_counts=True)))
print('OOF complete signed median by horizon and rate',np.round(np.median(e[complete],axis=0),5))
