import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']
def D(*x): return np.column_stack([np.ones(len(x[0])),*x])
def ls(a,v): return np.linalg.lstsq(a,v,rcond=1e-10)[0]
def prod(x,k): return np.column_stack([np.prod(x[:,ij],axis=1) for ij in combinations(range(4),k)])
a=s['panel_a']; u,v=a['u'],a['v']; good=np.max(abs(u-v),axis=1)<3
for label,f in [('diag',None),('affine',D(u)),('cubic',D(u,u**3)),('pair',D(u,prod(u,2))),('all',D(u,u**2,u**3,prod(u,2)))]:
 if f is None:
  w=np.array([ls(D(u[good,j]),v[good,j]) for j in range(4)]); pred=w[:,0]+u*w[:,1]; print(w)
 else: pred=f@ls(f[good],v[good])
 print('A',label,good.sum(),np.sqrt(np.mean((pred[good]-v[good])**2)),np.sort(np.max(abs(pred-v),axis=1))[-12:])
FUN={'soft':lambda x,k:x/(1+k*abs(x)),'rat':lambda x,k:x*x/(1+k*x*x),'oddrat':lambda x,k:x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'cos':lambda x,k:np.cos(k*x),'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'exp':lambda x,k:np.exp(k*x),'absrat':lambda x,k:abs(x)/(1+k*abs(x))}
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][key]; res=[]
 for label,fun in [('aff',lambda x:D(x)),('paironly',lambda x:D(prod(x,2))),('pair',lambda x:D(x,prod(x,2))),('quad',lambda x:D(x,x*x,prod(x,2))),('triple',lambda x:D(x,prod(x,3)))]:
  f=fun(u); w=ls(f,v); res.append((np.mean((f@w-v)**2)**.5,label,0))
 for label,fun in FUN.items():
  for aff in [False,True]:
   def obj(k):
    h=fun(u,k); f=D(u,h) if aff else D(h); return np.mean((f@ls(f,v)-v)**2)**.5
   out=min([minimize_scalar(obj,bounds=b,method='bounded') for b in [(.001,.8),(.8,1.8),(1.8,3),(3,5)]],key=lambda o:o.fun)
   res.append((out.fun,label+(' +lin' if aff else ''),out.x))
 print(panel,key,sorted(res)[:15])
def tm(p):return -np.expm1(-t[:,None]*p)
p=least_squares(lambda p:(tm(p)@ls(tm(p),tr['traj'][:52].T)-tr['traj'][:52].T).ravel(),[1.8,.5],bounds=([.8,.02],[8,.79])).x
print('rates',p,'noise',np.std(tm(p)@ls(tm(p),tr['traj'][:52].T)-tr['traj'][:52].T))
