import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from scipy.special import hyp2f1,erf
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-9)[0]
def err(d,v):return np.sqrt(np.mean((d@fit(d,v)-v)**2))
i,j=np.triu_indices(4); l,r=np.triu_indices(4,1)
funcs={'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'tanh':lambda x,k:np.tanh(k*x),'erf':lambda x,k:erf(k*x),'atan':lambda x,k:np.arctan(k*x),'soft':lambda x,k:x/(1+k*abs(x)),'soft2':lambda x,k:x/(1+k*abs(x))**2,'rat':lambda x,k:x/(1+k*x*x),'sqrt':lambda x,k:x/np.sqrt(1+k*x*x),'asinh':lambda x,k:np.arcsinh(k*x),'exp':lambda x,k:np.exp(-k*x*x)}
if __name__=='__main__':
 for pkey in ('panel_a','panel_b','panel_c'):
  p=s[pkey]; u=p['u']; n=len(u); f=aff(u)
  for vkey in p.keys()-{'u'}:
   v=p[vkey]; print(pkey,vkey)
   if pkey=='panel_a':
    for label,d in [('aff',f),('diag cubic',np.c_[f,u**3]),('diag quad cubic',np.c_[f,u*u,u**3]),('quad',np.c_[f,u[:,i]*u[:,j]])]:
     w=fit(d,v)
     for _ in range(20):
      q=np.argsort(np.linalg.norm(d@w-v,axis=1))[:36]; w=fit(d[q],v[q])
     print(label,np.round(np.sort(np.linalg.norm(d@w-v,axis=1))[::3],5)); print(np.round(w,4))
    continue
   bases={'aff':f,'quad':np.c_[f,u[:,i]*u[:,j]],'squares':np.c_[f,u*u],'interactions':np.c_[f,u[:,l]*u[:,r]],'cubic':np.c_[f,u**3],'powers':np.c_[f,u*u,u**3]}
   out=[]
   for label,d in bases.items():out.append((err(d,v),label,0))
   for name,g in funcs.items():
    for mode in range(3):
     def d(k):
      z=g(u,k)
      return [aff(z),np.c_[f,z],np.c_[aff(z),z*z]][mode]
     grid=np.geomspace(.04,25,70); vals=[err(d(k),v) for k in grid]; a=np.argmin(vals); opt=minimize_scalar(lambda k:err(d(k),v),bounds=(grid[max(0,a-1)],grid[min(len(grid)-1,a+1)]),method='bounded'); out.append((opt.fun,name+str(mode),opt.x))
   print(sorted(out)[:17]); print('all algebraic',[(k,err(d,v)) for k,d in bases.items()])
 t=s['eval_times']; z=s['train']['traj'][:52]
 def modes(p):return -np.expm1(-np.exp(p)[:,None]*t)
 def residual(p):
  d=modes(p).T;return (d@fit(d,z.T)-z.T).ravel()
 p=least_squares(residual,[-1,1],bounds=(-8,4));print('time roots',np.exp(p.x),'rms',np.std(p.fun))
