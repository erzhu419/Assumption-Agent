import numpy as np,pickle
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; u,v=[s['panel_a'][k] for k in ('u','v')]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; poly=lambda a,d:PolynomialFeatures(d).fit_transform(a)
for _ in range(4):
 a=poly(u,1); ids=np.arange(len(u)) if _==0 else np.argsort(np.linalg.norm(v-a@b,axis=1))[:45]; b=ls(a[ids],v[ids]);
print('A aff',b)
for k in (1,2,3):
 a=poly(u,k); e=v[ids]-a[ids]@ls(a[ids],v[ids]); print('poly A',k,np.std(e,axis=0));
 if k==2: print('A quad',ls(a[ids],v[ids]).round(3))
for names,panel in [('B',s['panel_b']),('C',s['panel_c'])]:
 for n,vv in panel.items():
  if n=='u':continue
  for d in (1,2):
   a=poly(panel['u'],d); print(names,n,d,np.std(vv-a@ls(a,vv),axis=0))
   if d==2:print(ls(a,vv).round(3))
for k in (1,2,3,4,5):
 a=poly(x,k)
 for alpha in (0,0.001,0.01,0.1,1):
  yz=np.c_[y,z]; est=Ridge(alpha=alpha,fit_intercept=False).fit(a[:52],yz[:52]).predict(a[52:]); e=np.abs(est-yz[52:]).mean(0);print('CV poly',k,alpha,e[0],e[1:].mean(),e[-1])
for name,g in [('t',t),('log',np.log1p(t)),('one',np.ones_like(t)),('sqrt',np.sqrt(t))]:
 def r(k, ret=False):
  h=np.array([-np.expm1(-k*t),g]); a=z[:52]@np.linalg.pinv(h); return (h,a) if ret else np.linalg.norm(z[:52]-a@h)
 f=minimize_scalar(r,bounds=(.03,7),method='bounded'); h,a=r(f.x,True); print('temporal',name,f.x,f.fun, 'basis',h.round(5))
_,sv,p=np.linalg.svd(z[:52],full_matrices=False);print('svd',sv,p[:2])
