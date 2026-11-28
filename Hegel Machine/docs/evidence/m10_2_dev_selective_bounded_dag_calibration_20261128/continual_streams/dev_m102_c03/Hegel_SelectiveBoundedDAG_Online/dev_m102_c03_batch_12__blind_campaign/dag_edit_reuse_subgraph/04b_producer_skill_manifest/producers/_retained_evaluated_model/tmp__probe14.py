import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda h,y:np.linalg.lstsq(h,y,rcond=None)[0]
def one(x):return np.column_stack([np.ones(len(x)),x])
def lib(u):
 w=np.roll(u,-1,axis=1); b=np.roll(u,1,axis=1); z={'lin':u,'cyc':u*w,'cycprev':u*b,'pair':np.column_stack([u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)]),'sq':u*u,'cube':u**3,'cymix':np.column_stack([u*w,u*w*w,u*u*w])}
 for k in [.22,1,1.25,2.15,2.4]:
  for name,f in [('cos',np.cos),('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('sat',lambda v:v/(1+abs(v))),('rat',lambda v:v/(1+v*v)),('gauss',lambda v:np.exp(-v*v))]:
   for nm,v in [('u',u),('plus',u+w),('minus',u-w),('edge',u*w)]:z[f'{name}{k}{nm}']=f(k*v)
 return z
for ky in ['panel_a','panel_b','panel_c']:
 p=s[ky];u=p['u'];z=lib(u)
 for vk,v in p.items():
  if vk=='u':continue
  scores=[]
  for nam,h in list(z.items())+[(n+'+lin',np.column_stack([z['lin'],h])) for n,h in z.items()]:
   h=one(h); w=np.ones(len(u),bool)
   for _ in range(8):
    r=np.linalg.norm(h@ls(h[w],v[w])-v,axis=1); w=np.argsort(r)[:int(len(u)*.85)]
   scores.append((np.median(r),np.sort(r)[-1],nam))
  print(ky,vk, sorted(scores)[:9]);print('raw ranges',np.round(np.min(v,0),3),np.round(np.max(v,0),3))
  if ky=='panel_a':
   w=np.roll(u,-1,axis=1);h=one(np.column_stack([u,u**2,u**3,u*w]));mask=np.arange(len(u))
   for _ in range(10):
    a=ls(h[mask],v[mask]);r=np.linalg.norm(h@a-v,axis=1);mask=np.argsort(r)[:36]
   print('poly median/max',np.median(r),np.sort(r));print('poly coeff',np.round(a,3))
t=s['eval_times'];y=s['train']['traj'].T;print('trajectory spectrum',np.linalg.svd(y,compute_uv=False))
for n in [2,3,4]:
 def resid(k):
  q=-np.expm1(-t[:,None]*k);return (q@ls(q,y)-y).ravel()
 f=least_squares(resid,np.geomspace(.38,6,n),bounds=(.015,20),max_nfev=120);print(n,f.x,np.std(f.fun))
