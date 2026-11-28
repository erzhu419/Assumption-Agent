import numpy as np, pickle,runpy,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; m=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); z=m['run_node']({'__input__':s}); q=(tr['x']-z['offset'])/z['gain']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def bases(x):
 yield 'aff',aff(x)
 yield 'quad',poly(x,2)
 yield 'cub',poly(x,3)
 for name,fun in [('tanh',np.tanh),('sin',np.sin),('rational',lambda u:u/(1+u*u)),('ss',lambda u:u*np.abs(u)),('sat',lambda u:u/np.hypot(1,u))]:
  for scale in [.5,1,2,3,4]:
   f=fun(x*scale);yield f'{name}{scale}',aff(f);yield f'aff+{name}{scale}',np.c_[aff(x),f];yield f'quad+{name}{scale}',np.c_[poly(x,2),f]
def cv(f,y,folds):
 pred=np.empty_like(y); idx=np.arange(len(y)); out=[]
 for a in folds:
  b=np.setdiff1d(idx,a); pred[a]=f[a]@ls(f[b],y[b]);out.append(np.abs(pred[a]-y[a]).mean())
 return np.mean(out),max(out)
for key,tar in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[key];x=p['u'];y=p[tar];folds=np.array_split(np.arange(len(y)),6);print(key,tar,sorted((cv(f,y,folds),k) for k,f in bases(x) if f.shape[1]<20)[:10]); print('raw pairs',np.round(np.c_[x,y][:8],3))
for tar in ['y','traj']:
 y=tr[tar];folds=np.array_split(np.arange(len(q)),6);print(tar,'direct',sorted((cv(f,y,folds),k) for k,f in bases(q))[:10]);
 if tar=='traj':
  u,d,v=np.linalg.svd(y,full_matrices=False);print('modes',v[:2]); print('scores',np.round(np.c_[q,u[:,:2]*d[:2]][:20],3))
