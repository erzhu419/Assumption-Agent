import numpy as np,pickle
from itertools import combinations_with_replacement as cw
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def poly(x,k):return np.column_stack([np.prod(x[:,j],axis=1) for j in cw(range(x.shape[1]),k)])
def design(x,k):return np.column_stack([poly(x,i) for i in range(k+1)])
def robust(X,y):
 rng=np.random.default_rng(123); m=X.shape[1]; best=None
 for _ in range(900):
  j=rng.choice(len(X),m,replace=False); w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]; r=np.linalg.norm(y-X@w,axis=1); i=np.argsort(r); score=r[i[int(.75*len(X))]]
  if best is None or score<best[0]:best=score,i
 j=best[1][:int(.8*len(X))]; w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]
 for _ in range(3):
  r=np.linalg.norm(y-X@w,axis=1); j=r<max(4*np.median(r),.0005); w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]
 return w,np.linalg.norm(y-X@w,axis=1)
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=140)
 print('t,tel',S['eval_times'],S['telemetry_vec'])
 for key in 'abc':
  p=S['panel_'+key];u=p['u']
  for name,v in p.items():
   if name=='u':continue
   for k in (1,2):
    X=design(u,k);w=np.linalg.lstsq(X,v,rcond=None)[0]; r=v-X@w; H=np.linalg.qr(X)[0]; print(key,name,k,'rms',np.mean(r*r)**.5,'loo',np.mean((r/(1-(H*H).sum(1))[:,None])**2)**.5); print(w.T)
   if key=='a':
    w,r=robust(design(u,1),v); print('robust',w.T,'residual',np.sort(r))
 x,y,z=[S['train'][k] for k in ('x','y','traj')]; t=S['eval_times'];
 for kind in (0,1,2):
  def b(q):return (-np.expm1(-t[:,None]*q)) if kind==0 else np.column_stack((-np.expm1(-t*q[0]),t if kind==1 else np.sin(q[-1]*t)))
  def fun(q):
   B=b(q);return (z-z@np.linalg.pinv(B).T@B.T).ravel()
  r=least_squares(fun,[1.5,.25] if kind==0 else ([1.5] if kind==1 else [1.5,1.])); print('time basis',kind,r.x,'rms',np.mean(r.fun**2)**.5)
 for k in range(1,5):
  D=design(x,k);w=np.linalg.lstsq(D[:56],np.c_[y,z[:,-1]][:56],rcond=None)[0];print('poly',k,'MAE',np.abs(D[56:]@w-np.c_[y,z[:,-1]][56:]).mean(0));print(w[:15].T if k<=2 else '')
