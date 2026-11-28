import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=(s['train'][k] for k in ['x','y','traj']); ts=s['eval_times']; np.set_printoptions(precision=4,suppress=True,linewidth=180)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def poly(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for i in range(d+1) for j in combinations_with_replacement(range(x.shape[1]),i)])
def fit(a,b,rob=True):
 m=np.ones(len(a),bool)
 for _ in range(10 if rob else 1):
  w=ls(a[m],b[m]); e=np.linalg.norm(a@w-b,axis=1); n=e<max(4*np.median(e),1e-8)
  if n.sum()<a.shape[1]+2 or np.array_equal(n,m): break
  m=n
 return w,e,m
if __name__=='__main__':
 print('times',ts)
 for p in ['panel_a','panel_b','panel_c']:
  u=s[p]['u']; print(p,'u range',u.min(0),u.max(0),'first',u[:4]);
  for k,v in s[p].items():
   if k=='u':continue
   print(k,'first',v[:7]);
   for label,a in [('linear',poly(u,1)),('quadratic',poly(u,2)),('abs',np.c_[poly(u,1),abs(u)]),('squares',np.c_[poly(u,1),u*u]),('saturation',np.c_[poly(u,1),u/(1+abs(u))]),('sin',np.c_[poly(u,1),np.sin(u)])]:
    w,e,m=fit(a,v);print(label,'med',np.median(e),'inliers',m.sum(),'sorted tail',np.sort(e)[-8:]);
    if p=='panel_a' and label in ['linear','quadratic']:print(w)
 for d in [1,2,3,4]:
  a=poly(x,d); vv=np.c_[y,t]; w=ls(a[:52],vv[:52]); print('raw poly',d,'in',np.mean(abs(a[:52]@w-vv[:52]),0),'out',np.mean(abs(a[52:]@w-vv[52:]),0))
 for j in [1,2,3]:
  def basis(r):return -np.expm1(-ts[:,None]*r)
  def err(r):
   a=basis(r);return (a@ls(a,t.T)-t.T).ravel()
  r=least_squares(err,np.geomspace(.2,3,j),bounds=(.001,25)).x;print('global exp',r,'rmse',np.std(err(r)))
 print('X y and traj samples');print(np.c_[x,y,t][:20]);
