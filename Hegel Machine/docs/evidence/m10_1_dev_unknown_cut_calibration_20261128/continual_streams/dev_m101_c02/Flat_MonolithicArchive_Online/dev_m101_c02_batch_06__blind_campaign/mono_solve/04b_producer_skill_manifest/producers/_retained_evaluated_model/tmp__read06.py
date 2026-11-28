from model06 import *
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
tr=s['train']; w=calibration(s)(tr['x']); y=tr['y']; T=tr['traj']; t=s['eval_times']; p=s['panel_b']; u,v=p['u'],p['v']; folds=[(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=4).split(w)]
f=lambda x,p:aff(x**3/(1+p[0]*abs(x)**p[1])+p[2]*x)
pp=least_squares(lambda p:(f(u,p)@ls(f(u,p),v)-v).ravel(),[.53,2.42,-.09]).x
z=f(w,pp)@ls(f(u,pp),v)
a=aff(z); pr=np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i+1,4)]); ds={'linear':a,'paired':np.c_[a,pr[:,1]+pr[:,4]],'all pairs':np.c_[a,pr],'all products':poly(z,2),'cal times B':np.c_[a,z*w,np.roll(z,1,axis=1)*w]}
def fit_link(d,b,kind):
 c=ls(d,b); l=d[:,1:]; sd=np.maximum(l.std(0),1e-8); l=l/sd
 def link(q,p):
  h=p[0]+q@p[1:]
  if kind=='atan': return np.c_[np.ones(len(h)),h,np.arctan(h)]
  if kind=='tanh': return np.c_[np.ones(len(h)),h,np.tanh(h)]
  if kind=='sin': return np.c_[np.ones(len(h)),h,np.sin(h)]
  if kind=='sq': return np.c_[np.ones(len(h)),h,h*h]
  return np.c_[np.ones(len(h)),h,h/(1+h*h)]
 def res(p):
  a=link(l,p); return a@ls(a,b)-b
 init=c[1:]*sd; starts=[np.r_[shift,k*init] for k,shift in [(0.3,0),(1,0),(2,0),(4,0),(1,-1),(1,1),(3,-1),(3,1)]]
 fit=min((least_squares(res,p,max_nfev=110) for p in starts),key=lambda o:o.cost); p=fit.x; c=ls(link(l,p),b)
 return lambda q:link(q[:,1:]/sd,p)@c
for name,d in ds.items():
 for kind in ['ols','atan','tanh','sin','rat','sq']:
  E=[]
  for i,j in folds:
   h=np.linalg.svd(T[i],full_matrices=False)[2][0]; amp=T[i]@h
   if kind=='ols': pred=d[j]@ls(d[i],np.c_[y[i],T[i]]); yp,tp=pred[:,0],pred[:,1:]
   else: yp=fit_link(d[i],y[i],kind)(d[j]); tp=fit_link(d[i],amp,kind)(d[j])[:,None]*h
   E.append([abs(yp-y[j]).mean(),abs(tp-T[j]).mean()]);
  print(name,kind,np.round(E[0],5),np.round(np.max(E,axis=0),5),np.round(np.mean(E[1:],axis=0),5),flush=True)
D=ds['paired']; h=np.linalg.svd(T,full_matrices=False)[2][0]; b=T@h; e=np.c_[y,b]-D@ls(D,np.c_[y,b]); print('coeff',ls(D,np.c_[y,b])); print('corr y and amp residual',np.corrcoef(e.T)); print('largest y residuals',np.round(np.c_[w,z,y,b,e][np.argsort(abs(e[:,0]))[-10:]],3)); print('B residual quantiles',np.quantile((f(u,pp)@ls(f(u,pp),v)-v),[0,.1,.5,.9,1],axis=0))
