import numpy as np, pickle
from scipy.optimize import least_squares
z=dict(np.load('/tmp/m98_arrays.npz')); globals().update(z)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; strata=it['campaign_diagnostics']['stratum']; v=th[2]+x@th[3:]; later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; H=past@(-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau)); G=(hold[:,None]-H)*th[:2]; target=np.c_[yc,yr]; scale=np.r_[np.full(len(t),.02),.05]
xs=x/np.std(x,axis=0); specs={'base':(None,0),'x0':(xs[:,[0]],.5),'x2':(xs[:,[2]],.5),'feature_shrink':(xs,1.5),'x02_shrink':(xs[:,[0,2]],1.5),'reset_fast':(None,0),'reset_slow':(None,0),'reset_both':(None,0),'amp_slow':(None,0),'linear_feature':(xs,1.5)}
def pred(p,name):
 T=tau*np.exp(p[:2]); s=np.exp(p[3]); K=-s*np.expm1(-t/s); b=np.c_[-np.expm1(-t[None,:]/T[:,None]),1/T]; A=G.copy(); F,pen=specs[name]; drift=np.exp(p[2])*v; out=0.
 if name.startswith('reset_'):
  j={'reset_fast':[0],'reset_slow':[1],'reset_both':[0,1]}[name]; A[:,j]+=H[:,j]*th[j]*p[4:]
 if name=='amp_slow': A[:,1]*=1+p[4]
 if F is not None:
  if name=='linear_feature': out=(F@p[4:])[:,None]*np.r_[t,1.]
  else: drift+=F@p[4:]
 return A@b+drift[:,None]*np.r_[K,1.]+out

def fit(name,m):
 F,pen=specs[name]; n=F.shape[1] if F is not None else (2 if name=='reset_both' else int(name!='base')); p0=np.r_[[.004,.15,.21,.06],np.zeros(n)];
 def err(p): return np.r_[((pred(p,name)[m]-target[m])/scale).ravel(),pen*p[4:]/.02]
 return least_squares(err,p0,loss='soft_l1',f_scale=1.5,bounds=(-4,4)).x

def cv(name,eligible,groups):
 e=np.full_like(target,np.nan)
 for g in np.unique(groups[eligible]):
  m=eligible&(groups==g); e[m]=pred(fit(name,eligible&~m),name)[m]-target[m]
 return e

def metrics(e):
 a=abs(e); result=[]
 for g in (labs,strata):
  result.extend(np.max([[np.median(a[g==l,:-1]),np.median(a[g==l,-1])] for l in np.unique(g)],axis=0))
 return np.round([*result,max(np.median(a[:,:-1],axis=0)),np.quantile(a[:,-2],.90)],5)
allrows=np.ones(len(labs),bool)
for name in specs:
 p=fit(name,allrows); print(name,'LOLO',metrics(cv(name,allrows,labs)),'LOSO',metrics(cv(name,allrows,strata)),'p',p.round(5))
for outer in (labs,strata):
 e=np.empty_like(target)
 for g in np.unique(outer):
  m=outer==g; scores={}
  for name in specs:
   a=abs(cv(name,~m,outer)[~m]); scores[name]=np.mean(np.log1p((a/scale)**2))
  name=min(scores,key=scores.get); e[m]=pred(fit(name,~m),name)[m]-target[m]; print('nested selection',g,name)
 print('nested',metrics(e))
