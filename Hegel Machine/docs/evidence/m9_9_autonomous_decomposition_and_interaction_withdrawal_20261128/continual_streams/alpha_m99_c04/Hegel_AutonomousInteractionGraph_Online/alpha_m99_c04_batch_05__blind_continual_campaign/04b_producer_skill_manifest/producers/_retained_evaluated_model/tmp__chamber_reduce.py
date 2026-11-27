from chamber_probe import *
import downstream_solver as m
xd,_=m._features(g,g)
q,s=m._chamber_response(g,xd,t,taus,theta)
e=np.c_[g['strain_curves_obs']-q,R-s]
lab=g['lab_id']; complete=np.isfinite(g['coupon_features_raw']).all(1)
k=np.c_[-taus[None,:]*np.expm1(-t[:,None]/taus),t]; k=np.r_[k,[[1.,1.,1.]]]; b=k@np.array([.2,.55,.25]); kb=k[:,0]-k[:,1]
make=lambda a,v: np.asarray(a)[:,None]*v
# Compare rate-conserving load terms with residual history on the same held-out coupons.
z0,z1=Z.T; h0=theta[0]/taus[0]; h1=theta[1]/taus[1]; href=4.
choice={'baseline':[], 'linear hold':[make(H,kb)], 'saturating hold':[make(H/(1+(H/href)**2),kb)], 'constant transfer':[make(np.ones(n),kb)], 'slow history':[make(h1*z1,k[:,1])], 'both history':[make(h0*z0,k[:,0]),make(h1*z1,k[:,1])], 'hold and slow history':[make(H/(1+(H/href)**2),kb),make(h1*z1,k[:,1])]}
for name,cols in choice.items():
 for project in (False,True):
  P=e.copy(); coefficients=[]
  if cols:
   A=np.stack(cols,axis=-1)
   project_drift=lambda u: u-np.einsum('ntj,t,s->nsj',u,b,b)/(b@b)
   W=project_drift(A) if project else A
   y=project_drift(e[...,None])[...,0] if project else e
   for l in np.unique(lab):
    trn=(lab!=l)&complete; te=lab==l
    c=np.linalg.lstsq(W[trn].reshape(-1,len(cols)),y[trn].ravel(),rcond=None)[0]; coefficients.append(c); P[te]-=A[te]@c
  metric=lambda masks: np.max([[abs(P[a,:-1]).mean(),abs(P[a,-2]).mean(),abs(P[a,-1]).mean()] for a in masks],axis=0).round(6)
  print(name,'project',project,'worst',metric([lab==l for l in np.unique(lab)]+[complete,~complete]+[g['stratum']==j for j in np.unique(g['stratum'])]),'coeff',np.round(coefficients,5))
print('archive entries:')
import json
for a in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print({key: (f'{type(value).__name__} len={len(value)}' if isinstance(value,(list,dict)) else value) for key,value in a.items()})
