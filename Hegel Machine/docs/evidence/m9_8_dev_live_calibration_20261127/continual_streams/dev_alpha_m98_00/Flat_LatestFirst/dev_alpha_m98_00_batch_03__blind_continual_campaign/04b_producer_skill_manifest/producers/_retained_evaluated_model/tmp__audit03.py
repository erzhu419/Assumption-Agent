import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']; a = item['campaign_diagnostics']; tr = d['train']
dt, t = d['past_dt'], d['eval_times']
ct, _ = m._prepare(d)
ytr = tr['strain_curves_obs'] - tr['strain_0'][:,None]
rtr = tr['instant_rate_obs']
tau, beta = m._fit(dt, t, ct, ytr, rtr)
for lab in np.unique(tr['lab_id']):
    mask = tr['lab_id'] != lab
    tt, bb = m._fit(dt, t, tuple(x[mask] for x in ct), ytr[mask], rtr[mask])
    dc, dr = m._design(tt,dt,t,tuple(x[~mask] for x in ct))
    print('benchtop held-out lab', lab, 'tau max relative change / strain median abs / rate median abs', np.max(np.abs(tt/tau-1)), np.median(np.abs(dc@bb-ytr[~mask])), np.median(np.abs(dr@bb-rtr[~mask])))

keys = ('lab_id','stratum','ref_coupons','coupon_features_raw')
x = m._aligned_features(a,{k:a[k][:0] for k in keys})
cal = m._stress_calibration(d['calibration_load_cells'])
past = a['past_channel_raw']@cal[:-1]+cal[-1]
hold = a['hold_channel_raw']@cal[:-1]+cal[-1]
C,R = m._design(tau,dt,t,(past,hold,x))
v = x@beta[2:]; g = -np.expm1(-t[:,None]/tau)
z = np.zeros((len(v),2))
for step, stress in zip(dt,past.T):
    z += -np.expm1(-step/tau)*(stress[:,None]-z)
y = np.c_[a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']]
base = np.c_[C@beta,R@beta]
Z = np.zeros((1,2))
forcing = np.vstack((g-t[:,None]/tau,Z))
W = v[:,None,None]*forcing[None,:,:]
# The two state columns span their pairwise sums and differences exactly.
M = z[:,None,:]*np.vstack((g,1/tau))[None,:,:]
S = v[:,None,None]*np.r_[t,1.][None,:,None]
models = {'conserved forcing': [W], 'pairwise memory': [np.concatenate((W,M),axis=2)], 'independent initial slope': [np.concatenate((W,S),axis=2)], 'rational saturation': [W/(1+k*v[:,None,None]**2) for k in (0.,0.03,0.1,0.3,1.,3.)]}
labs = a['lab_id']; groups = np.unique(labs); target = y-base

def fit(A,mask):
    return np.linalg.lstsq(A[mask].reshape(-1,A.shape[2]),target[mask].ravel(),rcond=None)[0]

for name, designs in models.items():
    stats=[]; choices=[]
    for lab in groups:
        keep = labs!=lab
        if len(designs)>1:
            def inner(A):
                return max(np.sqrt(np.mean((A[labs==j]@fit(A,keep&(labs!=j))-target[labs==j])**2)) for j in groups if j!=lab)
            choice=min(range(len(designs)),key=lambda i:inner(designs[i]))
        else:
            choice=0
        A=designs[choice]; c=fit(A,keep); e=A[~keep]@c-target[~keep]
        stats.append([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))]);choices.append(choice)
    print(name,'outer lab RMSE (strain, initial rate)',np.round(stats,6).tolist(),'worst',np.round(np.max(stats,axis=0),6).tolist(),'selected index',choices)
print('conserved initial-rate mean residual by lab', [round(float(np.mean((base-y)[labs==j,-1])),6) for j in groups])
out=m.solve_batch(d)
assert all(np.isfinite(arr).all() for arr in out.values())
assert .05 < out['estimated_taus'][0] < out['estimated_taus'][1] < 25
