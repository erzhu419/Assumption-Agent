import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; T=tr['traj']; t=s['eval_times']; x=tr['x']; y=tr['y']; n=52
np.set_printoptions(precision=5,suppress=True)
def rms(x):return np.mean(x*x,axis=0)**.5
def mae(x):return np.mean(abs(x),axis=0)
def pol(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in combinations_with_replacement(range(4),k)])
for key in ('panel_a','panel_b','panel_c'):
 p=s[key]; u=p['u']; F=pol(u,1)
 for j in p.keys()-{'u'}:
  v=p[j]; w=np.linalg.lstsq(F,v,rcond=None)[0]; r=np.linalg.norm(v-F@w,axis=1); keep=r<3
  w=np.linalg.lstsq(F[keep],v[keep],rcond=None)[0]; r=np.linalg.norm(v-F@w,axis=1); print(key,j,'affine',w,'res',np.sort(r));
  for d in (1,2,3):
   ff=pol(u[keep],d); h=ff@np.linalg.pinv(ff); print('deg',d,'loo',rms((v[keep]-h@v[keep])/np.maximum(1e-10,1-h.diagonal())[:,None]))
print('T SVD',np.linalg.svd(T,compute_uv=False)); print('Y joined SVD',np.linalg.svd(np.column_stack([T,y]),compute_uv=False));
for k in (1,2,3):
 def fun(r):
  q=-np.expm1(-t[:,None]*np.exp(r)); return (T[:n]-(T[:n]@np.linalg.pinv(q.T))@q.T).ravel()
 z=least_squares(fun,np.linspace(-1,1,k),max_nfev=250); print('rates',np.exp(z.x),'rms',rms(fun(z.x)))
for k in range(1,5):
 f=pol(x,k); pred=f[n:]@np.linalg.lstsq(f[:n],np.c_[y,T][:n],rcond=None)[0]; print('raw poly',k,mae(pred-np.c_[y,T][n:]))
