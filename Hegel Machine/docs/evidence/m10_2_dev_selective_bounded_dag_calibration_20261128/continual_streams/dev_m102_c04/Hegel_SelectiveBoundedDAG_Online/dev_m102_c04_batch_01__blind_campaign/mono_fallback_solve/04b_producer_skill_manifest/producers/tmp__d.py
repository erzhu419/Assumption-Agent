import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];x=tr['x'];y=tr['y'];T=tr['traj'];t=s['eval_times']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; A=lambda x:np.c_[np.ones(len(x)),x]; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in cwr(range(x.shape[1]),k)])
if __name__=='__main__':
 print('times, telemetry',t,s['telemetry_vec']);print('traj first',T[:7]);print('ranges',np.min(x,0),np.max(x,0),np.quantile(y,[0,.25,.5,.75,1]),np.quantile(T,[0,.25,.5,.75,1]));print('SVD',np.linalg.svd(T[:52],compute_uv=False));print('D SVD',np.linalg.svd(np.diff(T[:52],axis=1),compute_uv=False))
 for kind in range(5):
  def f(k):
   q=-np.expm1(-np.outer(t,k));return [q,np.c_[np.ones(len(t)),q],np.exp(-np.outer(t,k)),np.c_[t,q],np.c_[np.ones(len(t)),t,q]][kind]
  for k in (1,2,3):
   def fun(p):
    q=f(p);return (q@ls(q,T[:52].T)-T[:52].T).ravel()
   o=least_squares(fun,np.geomspace(.3,2,k),bounds=(.001,30),max_nfev=250);print('temporal',kind,k,o.x,np.mean(np.abs(o.fun)))
 for k in ('panel_a','panel_b','panel_c'):
  p=s[k];u=p['u']; print(k,'u v first',np.c_[u,*[v for j,v in p.items() if j!='u']][:6]);print('ranges',[(j,np.min(v,0),np.max(v,0)) for j,v in p.items()]);
 for d in range(1,5):
  f=poly(x,d);a=f@ls(f[:52],np.c_[y,T][:52]); print('raw poly',d,np.mean(abs(a-np.c_[y,T])[:52],0),np.mean(abs(a-np.c_[y,T])[52:],0))
