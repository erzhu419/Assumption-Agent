from m99_probe import *
from itertools import product
E=np.c_[-np.expm1(-t[:,None]/tau),t]; er=np.r_[1/tau,1.]; T=np.r_[E,er[None,:]]; Y=np.c_[yc,rc]; Y0=np.c_[cy,cr]; drift=lambda x:beta[2]+x@beta[3:]; Z=hc[:,None]-bc[:,:2]*tau
coef=np.linalg.lstsq(T,Y.T,rcond=None)[0].T; D=coef-np.c_[bc[:,:2]*tau*beta[:2],drift(xc)]; print('residual coefficients vs predictors: [drift,hold,z0,z1,delta A0,delta A1,delta steady,delta rate]'); print(np.round(np.c_[drift(xc),hc,Z,D,rc-cr],4));
cs, _=m._features(c,c)
_,ct=m._features(tr,c)
for name,xx in [('deployment',xc),('joint',xc_joint),('chamber_only',cs),('train_plus_chamber',ct)]:
 dc=drift(xx); target=Y-np.c_[ac[:,:,:2]@beta[:2],bc[:,:2]@beta[:2]]-dc[:,None]*T[:,2];
 for mode,P in [('drift',dc[:,None]),('const+drift',np.c_[np.ones(len(dc)),dc]),('drift_quad',np.c_[dc,dc**2]),('drift_hold',np.c_[dc,dc*hc]),('drift_hist',np.c_[dc,dc[:,None]*Z]),('drift+hist',np.c_[dc,Z]),('drift+hold+hist',np.c_[dc,hc,Z]),('cubic',np.c_[dc,dc**2,dc**3]),('rational',np.c_[dc,dc/(1+dc**2)]),('reset',np.c_[dc,hc[:,None]-bc[:,:2]*tau])]:
  for keep in [False,True]:
   # Test rate preserving correction separately from unconstrained shared-time amplitudes.
   U=T[:,:2]-T[:,2:3]*er[:2] if keep else T
   Q=(P[:,None,:,None]*U[None,:,None,:]).reshape(len(dc),len(t)+1,-1)
   preds=np.zeros_like(target); group=c['lab_id']
   for lab in np.unique(group):
    u=group!=lab; b=np.linalg.lstsq(Q[u].reshape(-1,Q.shape[-1]),target[u].ravel(),rcond=None)[0]; preds[~u]=Q[~u]@b
   e=preds-target; ce=max(np.sqrt(np.mean(e[group==l,:-1]**2)) for l in np.unique(group)); re=max(np.sqrt(np.mean(e[group==l,-1]**2)) for l in np.unique(group)); b=np.linalg.lstsq(Q.reshape(-1,Q.shape[-1]),target.ravel(),rcond=None)[0]; print(name,mode,'preserve' if keep else 'free',round(ce,4),round(re,4),np.round(b,3) if max(ce,re)<.08 else '')
