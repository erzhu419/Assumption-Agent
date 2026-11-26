import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
from pathlib import Path
root=Path('/holdout_task'); spec=importlib.util.spec_from_file_location('solver',root/'alpha_w04_batch_01_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with (root/'public_alpha_w04_batch_01.pkl').open('rb') as f: a=pickle.load(f)[0]
d,g=a['dataset'],a['gt']; tr,te=d['train'],d['test']; T=d['eval_times']; dt=d['past_dt']; affine=s._calibrate(d['calibration_load_cells']); fit0=s._fit

def profile(dt,T,h,p,x,z,y,r):
    tau,w=fit0(dt,T,h,p,x,z,y,r)
    # Pairwise differences of time-normalized strain cancel linear drift;
    # robustly profile coupon drift in the original observation coordinates instead.
    D=np.r_[T,1.]; target=np.column_stack((y,r)); q0=np.r_[tau,w[:2]]
    b,o=s._design(tau,dt,T,h,p,x,z)
    resid=np.column_stack((y-b@w,r-o@w))
    scale=np.maximum(1e-6,1.4826*np.median(np.abs(resid-np.median(resid,axis=0)),axis=0)); scale*=2.385
    A=D/scale
    def fun(q, profiled=True):
        b,o=s._design(q[:2],dt,T,h,p,x,z)
        u=(target-np.column_stack((b[:,:,:2]@q[2:],o[:,:2]@q[2:])))/scale
        off=np.median(u/A,axis=1)
        for _ in range(8):
            wt=1/(1+(u-off[:,None]*A)**2)
            off=((wt*u)@A)/(wt@(A*A))
        return (u-off[:,None]*A).ravel() if profiled else off
    q=least_squares(fun,q0,bounds=([.2,.2,-np.inf,-np.inf],[8.,8.,np.inf,np.inf]),loss='cauchy',max_nfev=80).x
    drift=fun(q,False)
    X=np.column_stack((np.ones(len(x)),x)); error=drift-X@w[2:]; spread=max(.003,1.4826*np.median(abs(error-np.median(error))))
    v=least_squares(lambda b:(drift-X@b)/spread,w[2:],loss='soft_l1').x
    if q[0]>q[1]: q=q[[1,0,3,2]]
    return q[:2],np.r_[q[2:],v]

def sub(u,m): return {k:v[m] for k,v in u.items()}
def run(fit,train,test):
    s._fit=fit
    try: return s._fit_split(d,train,test)
    finally: s._fit=fit0

def rm(e): return np.sqrt(np.mean(e*e))
for lab in range(4):
    m=tr['lab_id']==lab; t=sub(tr,m); missing=np.isnan(t['coupon_features_raw']).any(axis=1)
    for name, f in [('base',fit0),('profile',profile)]:
        tau,y,p,r=run(f,sub(tr,~m),t); e=y-t['strain_curves_obs']; print('LOLO',lab,name,np.round(tau,5),'complete/missing terminal MAE',np.round([np.mean(abs(e[q,-1])) for q in (~missing,missing)],5),'complete onset RMSE',round(rm((r-t['instant_rate_obs'])[~missing]),5))
for name,f in [('base',fit0),('profile',profile)]:
    tau,y,p,r=run(f,tr,te); h=s._crossing(T,te['strain_0'],y,te['critical_strain']); sec=(y[:,-1]-te['strain_0'])/np.maximum(abs(p),.5); miss=np.isnan(te['coupon_features_raw']).any(axis=1); lab=te['lab_id']; print(name,'tau',np.round(tau,6),'max relative tau error',round(max(abs(tau/g['taus']-1)),6),'overall/missing/worst lab RMSE',np.round([[rm(e),rm(e[miss]),max(rm(e[lab==c]) for c in range(4))] for e in (y-g['te_clean'],sec-g['te_secant'],h-g['te_cross'])],6).tolist())
