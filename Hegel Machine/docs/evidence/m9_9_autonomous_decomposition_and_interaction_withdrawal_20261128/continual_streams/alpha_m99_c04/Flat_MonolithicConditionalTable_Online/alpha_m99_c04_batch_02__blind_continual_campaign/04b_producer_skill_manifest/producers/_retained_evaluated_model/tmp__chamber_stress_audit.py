import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
i = pickle.load(open(root+'public_batch.pkl','rb'))[0]
d, g = i['dataset'], i['campaign_diagnostics']
tr, te, dt, t = d['train'], d['test'], d['past_dt'], d['eval_times']
xt, _ = m._features(tr, te)
s = m._stress_map(d['calibration_load_cells'])
tau, th = m._fit(tr, xt, s, dt, t)
x, _ = m._features(g, g)
complete = np.isfinite(g['coupon_features_raw']).all(1)
lab = g['lab_id']
past = s[0]+g['past_channel_raw']@s[1:]
hold = s[0]+g['hold_channel_raw']@s[1:]
z = np.zeros((len(lab),2))
for j, dtj in enumerate(dt):
    z = z*np.exp(-dtj/tau)+past[:,j,None]*(-np.expm1(-dtj/tau))
gap = hold[:,None]-z
B = np.vstack([np.c_[-np.expm1(-t[:,None]/tau), t], np.r_[1/tau,1.]])
c,r = m._design(g,x,s,dt,t,tau)
assert np.allclose(r[:,:2]*tau, gap, rtol=0, atol=2e-14)
y = np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]
prior = (gap*th[:2])@B[:,:2].T
v = y-prior
scale = np.r_[np.full(len(t),.004),.006]
rd = th[2]+x@th[3:]
base = rd[:,None,None]*B[None,:,:]
# All candidates use the same outer splits and independently refit their coefficients.
for name in ('base','gap_sum_and_difference','drift_x_gap_difference','rational_drift','rational_gap_difference'):
    err = np.full_like(y,np.nan)
    for l in np.unique(lab):
        fit = complete & (lab!=l)
        check = complete & (lab==l)
        q = gap[:,0]-gap[:,1]
        if name == 'rational_drift':
            q = rd
        if name == 'gap_sum_and_difference':
            F = np.concatenate([base, (gap.sum(1)[:,None,None]*B[None,:,:2]), (q[:,None,None]*B[None,:,:2])],axis=2)
        elif name == 'drift_x_gap_difference':
            F = np.concatenate([base,q[:,None,None]*base],axis=2)
        else:
            F = base
        rational = name.startswith('rational_')
        def predict(p):
            return (F@p[:-1])/(1+p[-1]*q[:,None]**2) if rational else F@p
        p = np.r_[.33,2.49,-.26,np.zeros(F.shape[-1]-3)]
        kwargs = {}
        if rational:
            p = np.r_[p,0.]
            kwargs['bounds'] = (np.r_[np.full(F.shape[-1],-np.inf),0.],np.inf)
        o = least_squares(lambda p: ((predict(p)-v)[fit]/scale).ravel(),p,loss='soft_l1',f_scale=1.5,**kwargs)
        err[check] = (predict(o.x)-v)[check]
    scores = np.array([np.sqrt(np.mean(err[complete & (lab==l)]**2,axis=0)) for l in np.unique(lab)])
    print(name, 'worst-lab early/late/rate RMSE', np.round([np.max(np.sqrt(np.mean(scores[:,:3]**2,1))),np.max(np.sqrt(np.mean(scores[:,-4:-1]**2,1))),np.max(scores[:,-1])],6), 'pooled strain/rate RMSE',np.round([np.sqrt(np.mean(err[complete,:-1]**2)),np.sqrt(np.mean(err[complete,-1]**2))],6))
    if name=='base':
        print('held-out signed lab mean deviations (curve, rate)', [(int(l), np.round([err[complete & (lab==l),:-1].mean(),err[complete & (lab==l),-1].mean()],6).tolist()) for l in np.unique(lab)])
