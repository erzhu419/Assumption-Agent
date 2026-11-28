import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52
ls=lambda x,y: np.linalg.lstsq(x,y,rcond=None)[0]
aff=lambda x:np.column_stack((np.ones(len(x)),x))
pairs=lambda x,k=2:np.column_stack([np.prod(x[:,i],axis=1) for i in it.combinations_with_replacement(range(4),k)])
dead=lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1]
for key in ['panel_a','panel_b','panel_c']:
 p=s[key]; u=p['u'];
 for k,v in p.items():
  if k=='u':continue
  print(key,k,'median/global residuals')
  for name,d in [('aff',aff(u)),('quad only',aff(pairs(u))),('quad',aff(np.c_[u,pairs(u)])),('cubic',aff(np.c_[u,pairs(u,3)]))]:
   b=ls(d,v); r=d@b-v; print(name,d.shape,np.median(abs(r)),np.std(r),np.round(b,3) if name=='aff' else '')
  for trans in [dead,lambda x,p:np.tanh(x*p[0]),lambda x,p:np.arcsinh(x*p[0]),lambda x,p:np.sin(x*p[0])]:
   for polynomial in [False,True]:
    def design(p):
     z=trans(u,p);return aff(np.c_[z,pairs(z)] if polynomial else z)
    def fun(p):
     d=design(p);return (d@ls(d,v)-v).ravel()
    opt=least_squares(fun,[.35,1.5],bounds=([0,.25],[5,5])); print('trans',trans.__name__,polynomial,opt.x,np.std(opt.fun))
print('times',s['eval_times'],'SVD',np.linalg.svd(tr['traj'],compute_uv=False));print('y from traj',ls(aff(tr['traj'][:n]),tr['y'][:n]),np.mean(abs(aff(tr['traj'][n:])@ls(aff(tr['traj'][:n]),tr['y'][:n])-tr['y'][n:])))
