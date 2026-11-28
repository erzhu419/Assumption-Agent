import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
from glob import glob
from runpy import run_path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=56
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
a=lambda x:np.c_[np.ones(len(x)),x]
poly=lambda x,d: np.c_[np.ones(len(x)),*[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)]]
def cv(u,v):
 p=np.linalg.pinv(u); e=v-u@(p@v); return np.mean(abs(e/np.maximum(1e-9,1-(u*p.T).sum(1))[:,None])),np.mean(abs(e))
for k in ['panel_b','panel_c']:
 p=s[k]; u=p['u'];
 for j,v in p.items():
  if j=='u':continue
  print(k,j,'poly', [cv(poly(u,d),v) for d in (1,2,3)])
  for label,inv in [('atanh',lambda x:np.arctanh(x)),('asin',lambda x:np.arcsin(x))]:
   for scale in (1,1.2,1.5,2,3):
    if np.max(abs(v))<scale: print(label,scale,cv(a(u),inv(v/scale)),end='; ')
   print()
  for label,g in [('tanh',np.tanh),('sin',np.sin),('square',np.square),('cube',lambda x:x**3),('relu',lambda x:np.maximum(x,0)),('dead',lambda x:np.sign(x)*np.maximum(abs(x)-.3,0)**1.5)]:
   print(label,cv(a(g(u)),v),cv(np.c_[a(u),g(u)],v),end='; ')
  print('\nlin coeff', ls(a(u),v).round(3))
for f in sorted(glob('/tmp/hegel_continual_archive/*solver.py')):
 st=dict(s,train={**tr,**{k:tr[k][:n] for k in ('x','y','traj')}},test={'x':tr['x'][n:]}); o=run_path(f)['solve_batch'](st); print(f.split('/')[-1],np.mean(abs(o['y_hat']-tr['y'][n:])),np.mean(abs(o['state_pred']-tr['traj'][n:])))
