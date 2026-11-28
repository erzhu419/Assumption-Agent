import pickle,itertools,numpy as np
from scipy.optimize import minimize, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_b']['u']; v=s['panel_b']['v']; np.set_printoptions(precision=5,suppress=True)
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(A,y):return np.linalg.lstsq(A,y,rcond=None)[0]
def design(x,f,typ):
 a=aff(x) if typ in ('lin','lincos') else np.ones((len(x),1)); h=np.sin(x*f); return np.c_[a,h,np.cos(x*f)] if 'cos' in typ else np.c_[a,h]
for typ in ('lin','lincos','sin','sincos'):
 def loss(f,idx=np.arange(len(u))):
  A=design(u[idx],f,typ); r=v[idx]-A@fit(A,v[idx]);return np.mean(r*r)
 grid=np.linspace(.3,10,200); f0=grid[np.argmin([loss(f) for f in grid])]; sc=minimize_scalar(loss,bounds=(f0-.06,f0+.06),method='bounded'); sp=minimize(loss,np.full(4,sc.x),method='L-BFGS-B',bounds=[(.15,12)]*4, options={'ftol':1e-16,'gtol':1e-12}); print(typ,'shared',sc.x,np.sqrt(sc.fun),'separate',sp.x,np.sqrt(sp.fun)); print(fit(design(u,sp.x,typ),v))
 if typ in ('sin','lin'):
  errors=[]
  for j in range(len(u)):
   idx=np.flatnonzero(np.arange(len(u))!=j); r=minimize(lambda f:loss(f,idx),sp.x,method='L-BFGS-B',bounds=[(.15,12)]*4, options={'ftol':1e-14,'gtol':1e-10});errors.append(v[j]-design(u[j:j+1],r.x,typ)@fit(design(u[idx],r.x,typ),v[idx]))
  print('loo',np.mean(np.abs(errors)))
