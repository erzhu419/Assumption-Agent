import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as m
I=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=I['dataset']; tr,te,g=d['train'],d['test'],I['campaign_diagnostics']; t,dt=d['eval_times'],d['past_dt']; labs=g['lab_id']
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; loads=cal['true_channel_loads']; clean=np.max(abs(raw-np.median(raw,axis=0)),axis=1)<8
errs=[]
for f in range(7):
 j=np.arange(len(raw))%7==f; use=~j; c=m._axial_load_map(dict(bridge_raw=raw[use],true_channel_loads=loads[use])); j &=clean; errs.extend((m._channel_loads(raw[j],c)-loads[j]).sum(1))
print('sparse bridge OOF axial RMSE/max',np.sqrt(np.mean(np.square(errs))),max(abs(np.array(errs))))
bridge=m._axial_load_map(cal); xt,xe=m._features(tr,te); sp=m._specimen(tr,xt,bridge)
V=lambda s: np.column_stack((s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']))
tau,w=m._fit(sp,dt,t,V(tr)[:,:-1],V(tr)[:,-1]); R=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); T=np.r_[t,1.]; y=V(g)
c,b=m._branches(sp,dt,t,tau); e=np.c_[c@w[:2]+(sp[2]@w[2:])[:,None]*t,b@w[:2]+sp[2]@w[2:]]-V(tr)
print('material',tau,w,'train MAD',np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])))
def merged(*groups):return {k:np.concatenate([s[k] for s in groups]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
def metrics(e):
 def rms(x):return np.array([np.sqrt(np.mean(x[:,:-1]**2)),np.sqrt(np.mean(x[:,-1]**2))])
 return np.round(rms(e),5),np.round(np.max([rms(e[labs==l]) for l in np.unique(labs)],axis=0),5)
for name,x in [('diag',m._features(g,g)[0]),('joint',m._features(merged(tr,te),g)[1]),('train',m._features(tr,g)[1])]:
 past,h,F=m._specimen(g,x,bridge); q=F@w[2:]; E=y-(h[:,None]*w[:2])@R.T
 print(name,'residual singular values',np.linalg.svd(E,compute_uv=False)[:5]); print('simple law',[(tag,metrics(q[:,None]*K-E)[0]) for tag,K in [('linear',T),('log',np.r_[np.log1p(t),1]),('exp',np.r_[-np.expm1(-t),1])]])
 print('per-input free T+log',np.round(np.linalg.lstsq(np.c_[np.ones(len(h)),q,h],np.linalg.lstsq(np.stack([T,np.r_[np.log1p(t),1]],1),E.T,rcond=None)[0].T,rcond=None)[0],4))
 for label,A in [('q',q[:,None]),('q+h',np.c_[q,h,h*0+1]),('q2+h',np.c_[q,q*q,h,h*0+1]),('q+h*q',np.c_[q,q*(h-2),h,h*0+1]),('features+h',np.c_[F,h])]:
  for tag,K in [('log',np.r_[np.log1p(t),1]),('exp',np.r_[-np.expm1(-t),1]),('rat',np.r_[t/(1+.6*t),1]),('sqrt',np.r_[t/np.sqrt(1+t),1])]:
   D=np.einsum('ni,bk->nkib',A,[T,K]).reshape(len(q),len(T),-1); P=np.empty_like(E)
   for l in np.unique(labs):
    ok=labs!=l; co=np.linalg.lstsq(D[ok].reshape(-1,D.shape[-1]),E[ok].ravel(),rcond=None)[0]; P[~ok]=D[~ok]@co
   print(label,tag,*metrics(P-E))
 if name=='diag':print('drift/hold/observed residual rate',np.round(np.c_[q,h,E[:,-1]],5).tolist())
