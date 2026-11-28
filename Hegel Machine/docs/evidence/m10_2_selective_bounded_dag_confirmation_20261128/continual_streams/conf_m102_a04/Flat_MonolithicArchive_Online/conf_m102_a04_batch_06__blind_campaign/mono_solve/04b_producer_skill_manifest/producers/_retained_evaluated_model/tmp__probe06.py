import pickle,numpy as np
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(f,y):return np.linalg.lstsq(f,y,rcond=1e-10)[0]
def bias(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for degree in range(d+1) for j in combinations_with_replacement(range(x.shape[1]),degree)])
def pterms(x):
 return np.column_stack((x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]))
if __name__=='__main__':
 u=s['panel_a']['u']; v=s['panel_a']['v']; keep=np.linalg.norm(v-u,axis=1)<5
 print('A rejected',np.where(~keep)); u=u[keep];v=v[keep]
 for d in range(1,4):
  f=poly(u,d); w=ls(f,v); print('A poly',d,np.sqrt(np.mean((f@w-v)**2))); print(np.round(w,4))
 for key in ['v1','v2']:
  u=s['panel_c']['u'];v=s['panel_c'][key]; print('C',key)
  D={'linear':bias,'prod':lambda x:bias(pterms(x)),'poly2':lambda x:poly(x,2)}
  for name,fun in D.items():
   f=fun(u);w=ls(f,v);print(name,np.sqrt(np.mean((f@w-v)**2)),np.round(w,3))
  for name,fn in [('cos',lambda x,q:np.cos(q*x)),('sin',lambda x,q:np.sin(q*x)),('tanh',lambda x,q:np.tanh(q*x)),('curv',lambda x,q:np.c_[x,x*x,x*x/(1+q*x*x)]),('odd',lambda x,q:x/(1+q*abs(x))),('rational',lambda x,q:x/(1+q*x*x))]:
   def err(q):
    f=bias(fn(u,q));return np.sqrt(np.mean((f@ls(f,v)-v)**2))
   grid=np.geomspace(.07,9,80); k=np.argmin([err(q) for q in grid]); res=minimize_scalar(err,bounds=(grid[max(k-1,0)],grid[min(k+1,len(grid)-1)]),method='bounded');print(name,res.x,res.fun)
