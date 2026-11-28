import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(suppress=True,precision=5,linewidth=140)
def poly(u,d):
 keys=[k for n in range(d+1) for k in combinations_with_replacement(range(4),n)]
 return np.column_stack([np.prod(u[:,k],axis=1) for k in keys]),keys
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.max(abs(v),1)<3; print('A keep',sum(keep),np.max(abs(v),1)[~keep]); u=u[keep];v=v[keep]
for d in [1,2,3]:
 X,k=poly(u,d); b=np.linalg.lstsq(X,v,rcond=None)[0]; r=X@b-v; h=(X*np.linalg.pinv(X).T).sum(1); print('A',d,'fit',abs(r).mean(0),'loo',np.mean(abs(r)/np.maximum(1-h,1e-8)[:,None],0));
 if d==2: print(list(zip(k,b)))
for pan,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[pan]['u'],s[pan][key]; print(pan,key)
 X,k=poly(u,2); b=np.linalg.lstsq(X,v,rcond=None)[0]; print(list(zip(k,b)))
 for use_sin in [False,True]:
  def design(p,x):
   a=x*p[:4]; return np.column_stack((np.ones(len(x)),np.cos(a),np.sin(a) if use_sin else np.empty((len(x),0))))
  def fun(p):
   X=design(p,u);return (X@np.linalg.lstsq(X,v,rcond=None)[0]-v).ravel()
  sols=[least_squares(fun,np.full(4,i),bounds=(.05,12),max_nfev=200) for i in [0.6,1.5,2.5,3.5,4.5,5.5,7.5]]; a=min(sols,key=lambda a:a.cost);print('axis trig',use_sin,a.x,np.mean(abs(a.fun)))
 if pan!='panel_b':continue
 for ns in [1,2,3,4]:
  def design(w,x):
   z=x@w.reshape(ns,4).T; return np.column_stack((np.ones(len(x)),np.cos(z),np.sin(z)))
  def fun(w):
   X=design(w,u);return (X@np.linalg.lstsq(X,v,rcond=None)[0]-v).ravel()
  rng=np.random.default_rng(843); sols=[]
  for i in range(30):
   w=(np.eye(4)[:ns]*2 if i==0 else rng.normal(0,[.7,1.3,2.2][i%3],(ns,4))); sols.append(least_squares(fun,w.ravel(),max_nfev=180))
  a=min(sols,key=lambda a:a.cost); X=design(a.x,u); print('shared trig',ns,np.mean(abs(a.fun)),a.x.reshape(ns,4),'coef',np.linalg.lstsq(X,v,rcond=None)[0])
