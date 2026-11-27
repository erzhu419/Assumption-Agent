import sys
sys.path.insert(0, '/holdout_task')
import pickle, numpy as np
from _verified_upstream import calibrate_upstream
from downstream_solver import _design, _predict
from batch_solver import solve_batch
with open('/holdout_task/public_batch.pkl','rb') as f: item = pickle.load(f)[0]
raw = item['dataset']; dep = raw['test']; dg = item['campaign_diagnostics']
out = solve_batch(raw); tau, beta = out['estimated_taus'], out['hereditary_weights']
cal = calibrate_upstream({**raw,'test':dg}); d = cal['test']; dt,t = cal['past_dt'],cal['eval_times']; labs = d['lab_id']
def align(s):
    x = s['coupon_features_raw'].copy()
    for l in np.unique(s['lab_id']):
        ref = d['ref_coupons'][labs==l]; m = s['lab_id']==l
        x[m] = (x[m]-np.median(ref.mean(0)))/np.median(ref.std(0))
    return x
pool=align(dep); _,idx,n=np.unique(np.c_[dep['lab_id'],dep['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[idx]
x=align(d); x=np.where(np.isfinite(x),x,np.nansum(pool*w[:,None],0)/(np.isfinite(pool).T@w))
p,h=d['past_channel_raw'].sum(-1),d['hold_channel_raw'].sum(-1)
a,c=_design(tau,x,p,h,dt,t); z=np.zeros((len(h),2))
for j in range(len(dt)):
    z=np.exp(-dt[j]/tau)*z-np.expm1(-dt[j]/tau)*p[:,j,None]
np.testing.assert_allclose(c[:,:2]*tau,h[:,None]-z,rtol=0,atol=2e-14)
base=np.c_[a[:,:,:2]@beta[:2],c[:,:2]@beta[:2]]; y=np.c_[d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs']]
drift=beta[2]+x@beta[3:]; f=np.r_[np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1][None,:]]
q,r=_predict(tau,beta,x,p,h,dt,t); fixed=np.c_[q,r]; hh=(h-h.mean())/h.std(); zz=(z[:,0]-z[:,1]); zz=(zz-zz.mean())/zz.std(); dd=drift/drift.std()
cv={}; wt=np.r_[np.ones(len(t)),2.]; delta=y-base
for name,g in [('free modal',drift[:,None]),('offset',np.c_[drift,np.ones(len(h))]),('load interaction',np.c_[drift,drift*hh,drift*zz]),('ratio law',np.c_[drift,drift/(1+dd**2)]),('three-way interaction',np.c_[drift,drift*hh,drift*zz,drift*hh*zz])]:
    M=(g[:,None,:,None]*f[None,:,None,:]).reshape(len(h),len(wt),-1); pred=base.copy()
    for l in np.unique(labs):
        m=labs!=l; K=(M[m]*wt[None,:,None]).reshape(-1,M.shape[-1]); coef=np.linalg.lstsq(K,(delta[m]*wt).ravel(),rcond=None)[0]; pred[~m]+=M[~m]@coef
    cv[name]=pred
for name,pred in [('benchtop',np.c_[a@beta,c@beta]),('artifact',fixed),*cv.items()]:
    e=pred-y; rms=lambda v: np.sqrt(np.mean(v*v))
    print(name, 'worst lab strain/rate', max(rms(e[labs==l,:-1]) for l in np.unique(labs)),max(rms(e[labs==l,-1]) for l in np.unique(labs)), 'last horizon',rms(e[:,-2]), 'within-lab strain RMSE', rms(np.concatenate([e[labs==l,:-1]-e[labs==l,:-1].mean(0) for l in np.unique(labs)])))
