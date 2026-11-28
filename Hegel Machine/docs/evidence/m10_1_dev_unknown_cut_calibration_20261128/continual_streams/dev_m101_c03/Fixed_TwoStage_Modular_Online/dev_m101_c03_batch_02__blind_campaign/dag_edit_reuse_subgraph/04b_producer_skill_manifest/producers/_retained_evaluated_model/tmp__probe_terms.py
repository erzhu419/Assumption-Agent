exec(open('/tmp/probe_routes.py').read().split('names=list(sources)')[0])
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
print('times and rates:',s['eval_times'],R)
base=aug(x)
print('linear target coefficients and RMSE',ls(base,target),np.sqrt(np.mean((target-base@ls(base,target))**2,axis=0)))
for name in list(sources)[:9]:
 z=sources[name]; f=vf(z)@C1; g=aug(np.cos(2.4*z))@C2
 q=np.column_stack([z[:,l]*z[:,m] for l,m in combinations_with_replacement(range(4),2)])
 for nm,extra in [('v1',f),('v2',g),('v1v2',np.c_[f,g]),('v2 and v2 squared',np.c_[g,g*g]),('all 16 products',np.einsum('ni,nj->nij',f,g).reshape(-1,16)),('v2 plus all products',np.c_[g,np.einsum('ni,nj->nij',f,g).reshape(-1,16)]),('v2 plus squares and crosses',np.c_[g,q]),('v2 plus z*cos(z)',np.c_[g,np.einsum('ni,nj->nij',z,np.cos(2.4*z)).reshape(-1,16)])]:
  H=np.c_[base,extra]; C=ls(H[i],target[i]); e=np.mean(abs(H[j]@C-target[j]),axis=0)
  print(name,nm,np.round(e,5))
  if np.min(e)<.015:print('coeff',np.round(C[:,e<.015],4).T)
for nm,fn in [('softsign',lambda z,k:z/(1+k*abs(z))),('exp',lambda z,k:np.sign(z)*(-np.expm1(-k*abs(z)))),('log',lambda z,k:np.sign(z)*np.log1p(k*abs(z))),('pade',lambda z,k:z/(1+k*z*z))]:
 for bias in (False,True):
  def err(k):
   f=fn(u,k); H=aug(np.c_[u,f] if bias else f);return np.sqrt(np.mean((H@ls(H,v)-v)**2))
  opt=minimize_scalar(err,bounds=(.05,5),method='bounded');print('v1',nm,bias,opt.x,opt.fun)
