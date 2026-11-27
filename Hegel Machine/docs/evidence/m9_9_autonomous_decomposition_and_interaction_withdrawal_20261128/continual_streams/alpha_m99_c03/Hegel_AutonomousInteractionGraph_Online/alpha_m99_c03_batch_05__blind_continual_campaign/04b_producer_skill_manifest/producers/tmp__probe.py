import sys,pickle,importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
sp=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c03_batch_01_solver.py'); p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=calibrate_upstream(dict(a['dataset'],chamber_verification_trials=a['campaign_diagnostics'])); tr,te,q=(d[k] for k in ('train','test','chamber_verification_trials')); dt,t=d['past_dt'],d['eval_times']; x=p._features(tr,te); nq=len(q['lab_id']); xq=p._features(tr,q)[len(tr['lab_id']):]; bridge=np.vstack((np.zeros(4),np.eye(4))); train=p._coupon_inputs(tr,x[:len(tr['lab_id'])],bridge); diag=p._coupon_inputs(q,xq,bridge)
ytr=tr['strain_curves_obs']-tr['strain_0'][:,None]; rtr=tr['instant_rate_obs']; tau,th=p._fit(train,dt,t,ytr,rtr); print('material',tau,th)
y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; past,h,F=diag; dr=F@th[2:]; A,B=p._design(tau,diag,dt,t); gap=B[:,:2]*tau; z=h[:,None]-gap; E=-np.expm1(-t[:,None]/tau)
def err(a,b):
 e=np.c_[a-y,b-r]; print('curve/rate rmse',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)), 'median', np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1])))
print('ranges',h.min(),h.max(),y.min(),y.max(),r.min(),r.max(),dr.min(),dr.max()); print('baseline');err(A@th,B@th);print('old law');err(h[:,None]*(E@th[:2])+dr[:,None]*(0.5*E@tau),h*np.sum(th[:2]/tau)+dr)
print('diag sample rows: hold, memories, drift, first, last, rate');print(np.round(np.c_[h,z,dr,y[:,0],y[:,-1],r],5));print('singular values',np.linalg.svd(np.c_[y,r],compute_uv=False))
obs=np.c_[y,r]; tt=np.r_[t,0.];
M=np.eye(len(h))-np.outer(h,h)/(h@h); u,s,v=np.linalg.svd(M@obs,full_matrices=False); kernel=v[0]/v[0,-1]; latent=(M@obs@kernel)/(kernel@kernel); hcoef=h@obs/(h@h)
print('hold-projected singular values',np.round(s,6)); print('latent normalized kernel',np.round(kernel,6)); print('hold slopes',np.round(hcoef,6)); print('old kernel',np.round(np.r_[.5*E@tau,1],6)); print('linear kernel',np.r_[t,1]); print('hold kernel differing from intrinsic',np.round(hcoef-np.r_[E@th[:2],np.sum(th[:2]/tau)],6));
def combine(*bs): return {k:np.concatenate([b[k] for b in bs]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
for name,xa in [('tr,q',xq),('q,q',p._features(q,q)[nq:]),('tr,te,q',p._features(combine(tr,te),q)[-nq:]),('te,q',p._features(te,q)[-nq:])]:
 rr=np.c_[np.ones(nq),xa]@th[2:]; xx=np.c_[h,rr]; coef=np.linalg.lstsq(xx,obs,rcond=None)[0]; er=xx@coef-obs; print(name,'rmse',np.sqrt(np.mean(er[:,:-1]**2)),np.sqrt(np.mean(er[:,-1]**2)),'max',np.max(np.abs(er)), 'rate coef',coef[:,-1]); print('curves hold',np.round(coef[0,:-1],5),'drift',np.round(coef[1,:-1],5));
 xx=np.c_[h,np.ones(nq),xa]; coef=np.linalg.lstsq(xx,obs,rcond=None)[0]; print('free spatial all',np.sqrt(np.mean((xx@coef-obs)**2)),np.round(coef[:,-1],5))
print('branch relaxation design',np.round(np.c_[t,E],5))
from scipy.optimize import least_squares, minimize_scalar
xqq=p._features(q,q)[nq:]; dq=np.c_[np.ones(nq),xqq]@th[2:];
base=h[:,None]*np.r_[E@th[:2],np.sum(th[:2]/tau)]
J=np.vstack((E*tau,np.ones((1,2)))); L=np.r_[t,1];
def K(s):return np.r_[-s*np.expm1(-t/s),1.]
for name,D in [('two material branches',J),('linear plus slow',np.c_[L,J[:,1]]),('linear plus both',np.c_[L,J])]:
 c=np.linalg.lstsq(D,kernel,rcond=None)[0];print(name,c,'kernel RMSE',np.sqrt(np.mean((D@c-kernel)**2)))
f=minimize_scalar(lambda s: np.linalg.norm(K(s)-kernel),bounds=(.1,25),method='bounded'); print('single free time',f.x,'kernel RMSE',f.fun/np.sqrt(len(kernel)))
for name, fn, p0 in [('free time plus fast',lambda v: v[0]*K(tau[0])+(1-v[0])*K(v[1]),[.1,5]),('free time plus slow',lambda v: v[0]*K(tau[1])+(1-v[0])*K(v[1]),[.5,3]),('two free times',lambda v:v[0]*K(v[1])+(1-v[0])*K(v[2]),[.1,.7,5]),('free time plus linear',lambda v:v[0]*L+(1-v[0])*K(v[1]),[.1,4])]:
 g=least_squares(lambda v:fn(v)-kernel,p0,bounds=([0]+[.05]*(len(p0)-1),[1]+[80]*(len(p0)-1))); print(name,g.x,'kernel RMSE',np.sqrt(np.mean(g.fun**2)))
a0=(obs-base)@kernel/(kernel@kernel); hh=obs-a0[:,None]*kernel; print('constraint hold rms',np.sqrt(np.mean((hh-base)**2)), 'max',np.max(np.abs(hh-base)));print('amplitude by coupon (lab,stratum,number missing,latent,diagnostic drift,difference):');print(np.round(np.c_[q['lab_id'],q['stratum'],np.isnan(q['coupon_features_raw']).sum(1),a0,dq,a0-dq],6)); print('qb rmse clean subset',np.sqrt(np.mean((a0[np.isfinite(q['coupon_features_raw']).all(1)]-dq[np.isfinite(q['coupon_features_raw']).all(1)])**2)))
for name,D in [('two',J),('slow',J[:,1:]),('linear+both',np.c_[L,J])]:
 def obj(c):return (base+dq[:,None]*(D@c)-obs).ravel()
 g=least_squares(obj,np.ones(D.shape[1])/D.shape[1]);print('anchored',name,g.x,np.sqrt(np.mean(g.fun**2)))
print('test/train counts missing',np.isnan(tr['coupon_features_raw']).sum(0),np.isnan(te['coupon_features_raw']).sum(0)); print('train residual MAD',np.median(np.abs((p._design(tau,train,dt,t)[0]@th-ytr))),np.median(np.abs((p._design(tau,train,dt,t)[1]@th-rtr))))
clean=np.isfinite(q['coupon_features_raw']).all(1)
D=np.c_[L,J]; Z=np.stack([D[:,0]-D[:,2],D[:,1]-D[:,2]],1)
for label, mask in [('all',np.ones(nq,bool)),('complete',clean)]:
 a=dq[mask,None,None]*Z; b=obs[mask]-base[mask]-dq[mask,None]*D[:,2]; w=np.linalg.lstsq(a.reshape(-1,2),b.ravel(),rcond=None)[0]; print('simplex',label,np.r_[w,1-w.sum()]);
 for ww in [np.r_[w,1-w.sum()],[.25,.25,.5],[.25,.23,.52],[.25,.225,.525]]:
  er=base+dq[:,None]*(D@ww)-obs;print(ww,'complete RMSE',np.sqrt(np.mean(er[clean,:-1]**2)),np.sqrt(np.mean(er[clean,-1]**2)), 'missing RMSE',np.sqrt(np.mean(er[~clean,:-1]**2)),np.sqrt(np.mean(er[~clean,-1]**2)))
raw=q['coupon_features_raw']; xraw=xqq.copy(); xraw[~np.isfinite(raw)]=np.nan
keys=np.column_stack((q['lab_id'],q['stratum'])); _,ix,c=np.unique(keys,axis=0,return_inverse=True,return_counts=True); present=np.isfinite(xraw)
for name,weights in [('balanced',1/c[ix]),('plain',np.ones(nq))]:
 W=weights[:,None]; m=np.nansum(W*xraw,0)/np.sum(W*present,0); drift=np.c_[np.ones(nq),np.where(present,xraw,m)]@th[2:]; print(name,m,'drift discrepancy',np.round((a0-drift)[~clean],6))
