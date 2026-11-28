import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement as comb
sys.path.insert(0,'/tmp/hegel_continual_archive');import conf_m102_a01_batch_00_solver as prior
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; z,h=prior._coordinates(s,x); p=s['panel_c']; u=p['u']; n=52
np.set_printoptions(precision=4,suppress=True)
def affine(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.stack([np.prod(x[:,i],axis=1) for j in range(d+1) for i in comb(range(x.shape[1]),j)],1)
def loo(A,b):
 P=np.linalg.pinv(A);r=b-A@(P@b); d=np.maximum(1-np.sum(A*P.T,1),1e-10); return np.sqrt(np.mean((r/d[:,None])**2))
models={}
for k in ['v1','v2']:
 out=[]
 for name,f,lb,ub in [('sin',np.sin,.2,3),('tanh',np.tanh,.2,4),('arctan',np.arctan,.2,4),('cos',np.cos,.2,3),('softsign',lambda a:a/(1+abs(a)),.1,4),('rational',lambda a:a/(1+a*a),.1,3),('sqrt',lambda a:np.sqrt(1+a*a),.1,3)]:
  def A(w,t):return np.c_[affine(w),f(t*w)]
  t=minimize_scalar(lambda t:loo(A(u,t),p[k]),bounds=(lb,ub),method='bounded').x
  out.append((loo(A(u,t),p[k]),name,t,f))
 out.sort(key=lambda a:a[0]); print(k,[(e,name,t) for e,name,t,f in out]); e,name,t,f=out[0]; model=np.linalg.lstsq(A(u,t) if False else np.c_[affine(u),f(t*u)],p[k],rcond=None)[0]; print('model',model); models[k]=lambda w,t=t,f=f,c=model:np.c_[affine(w),f(t*w)]@c
for nm,w in [('x',x),('z',z),('h',h)]:
 v1=models['v1'](w);v2=models['v2'](w); a=affine(w); q=poly(w,2)
 for name,F in [('v1',affine(v1)),('v2',affine(v2)),('v1+v2',np.c_[affine(v1),v2]),('lin+v1v2',np.c_[a,v1,v2]),('quad+v1v2',np.c_[q,v1,v2]),('quad(v1)',poly(v1,2)),('quad(v2)',poly(v2,2)),('lin+quad(v1)',np.c_[a,poly(v1,2)[:,1:]]),('lin+quad(v2)',np.c_[a,poly(v2,2)[:,1:]]),('quad+quad(v1)',np.c_[q,poly(v1,2)[:,1:]]),('quad+quad(v2)',np.c_[q,poly(v2,2)[:,1:]])]:
  B=np.c_[y,T]; c=np.linalg.lstsq(F[:n],B[:n],rcond=None)[0]; E=F@c-B; print(nm,name,'val',np.mean(abs(E[n:,0])),np.mean(abs(E[n:,1:])), 'train', np.sqrt(np.mean(E[:n]**2,0))[[0,1,8]])
