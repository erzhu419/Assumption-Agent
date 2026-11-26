import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
p = '/holdout_task/alpha_w02_batch_07_solver.py'
s = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_alpha_w02_batch_07.pkl','rb'))[0]['dataset']
tr,te = d['train'],d['test']; lab = tr['lab_id']; n = len(lab)
dt,t,sy = (d[k] for k in ('past_dt','eval_times','yield_stress'))
f = m._features(tr,te)[:n]; cal = m._calibrate(d['calibration_load_cells'],sy)
x = m._inputs(tr,f,cal,dt,sy); sx = lambda mask: tuple(v[mask] for v in x)
y = np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
full = np.isfinite(tr['coupon_features_raw']).all(axis=1)
hold = x[1]/sy; past = x[0]; z = f[:,1:]
# Pairwise sums and differences alone are already in the linear feature span.
# Add their squares through the equivalent pair products; also test stress-dependent saturation.
pairs = np.column_stack([z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i+1,z.shape[1])])
rational = np.column_stack([hold/(1+hold**2), z* (hold**2/(1+hold**2))[:,None], (past[:,-1]/sy)/(1+hold**2)])
blocks = {'pairs':pairs, 'saturation':rational, 'offset':np.ones((n,1))}

def residual(fit,val):
    tau,b = m._fit(sx(fit),dt,t,y[fit]); a = m._design(sx(val),dt,t,tau)
    return y[val]-a@b,tau

def slope(r):
    return np.nanmedian(r[:,:-1]/t,axis=1)

def scores(r):
    return tuple(round(float(v),6) for v in (np.nanmedian(abs(slope(r))),np.nanmedian(abs(r[:,-1]))))

def correction(a,b,target):
    good=np.isfinite(target); a=a[good]; target=target[good]
    norm=np.maximum(np.sqrt(np.mean(a*a,axis=0)),1e-9); a=a/norm; b=b/norm
    scale=max(m._mad(target),1e-6)
    coef=least_squares(lambda c: a@c-target,np.zeros(a.shape[1]),loss='soft_l1',f_scale=scale).x
    return b@coef

for j in np.unique(lab):
    train=full&(lab!=j); val=full&(lab==j); r,tau=residual(train,val)
    inner=np.full_like(y,np.nan)
    for k in np.unique(lab[train]):
        a=train&(lab==k); inner[a]=residual(train&~a,a)[0]
    h=np.r_[t,1.]
    print('lab',j,'n',int(val.sum()),'median abs slope/rate',scores(r),'tau',np.round(tau,5),flush=True)
    for label,b in blocks.items():
        delta=correction(b[train],b[val],slope(inner[train]))
        print(' ',label,scores(r-delta[:,None]*h),flush=True)
    st=tr['stratum'][val]
    print('  within stratum signed slope medians',[(int(k),int((st==k).sum()),round(float(np.nanmedian(slope(r)[st==k])),6)) for k in np.unique(st)])
    # Verify the analytic t=0+ column against a forward secant on actual histories.
    a=m._design(sx(val),dt,np.array([1e-6]),tau)
    print('  onset operator max secant error',np.max(abs(a[:,0,:]/1e-6-a[:,1,:])))
