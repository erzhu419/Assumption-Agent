import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, eigh
from scipy.optimize import minimize
from graph_study import data, base, evalf

def profile(C, S):
 p=len(S)
 try: H=cholesky(S,lower=True,check_finite=False)
 except np.linalg.LinAlgError: return None
 D=cho_solve((H,True),np.eye(p),check_finite=False)
 eig,Q=eigh(H.T@C@H,subset_by_index=(p-2,p-1),check_finite=False)
 V=solve_triangular(H.T,Q,check_finite=False)
 t=np.maximum(eig-1,0)
 R=C-(V*t)@V.T
 obj=np.sum(S*C)-2*np.log(H.diagonal()).sum()+sum(np.log1p(t)-t)
 return obj,R-D,R

def fit(C, S, lam, mask=None):
 p=len(C); d=np.arange(p); i,j=np.triu_indices(p,1)
 if mask is not None: i,j=i[mask],j[mask]
 def unpack(x):
  out=np.diag(x[:p]); out[i,j]=out[j,i]=x[p:]; return out
 def f(x):
  q=profile(C,unpack(x))
  if q is None: return 1e8, np.zeros_like(x)
  y,g,_=q
  t=x[p:]; v=np.hypot(t,.012)
  return y+2*lam*np.sum(v-.012),np.r_[g[d,d],2*(g[i,j]+lam*t/v)]
 x=np.r_[S.diagonal(),S[i,j]]
 S0=unpack(x); gap=np.linalg.eigvalsh(S0)[0]
 if gap<.15: x[:p]+=.15-gap
 opt=minimize(f,x,method='L-BFGS-B',jac=True,options=dict(maxiter=120,ftol=2e-11,gtol=2e-6))
 return unpack(opt.x)

def methods(C):
 F=base(C,.06,'raw'); G=base(C,0,'cov'); p=len(C); i,j=np.triu_indices(p,1)
 for n,U in [('prec',F),('cov',G)]:
  for lam in (.019,.028,.037):
   S=fit(C,U,lam); _,_,R=profile(C,S); T=np.linalg.inv(R)
   yield f'{n}{lam}S',S
   yield f'{n}{lam}T',T
   yield f'{n}{lam}blend',.7*S+.3*T

if __name__=='__main__':
 z=np.load('/holdout_task/public_graph.npz'); X=z['X']; C=np.cov(X,rowvar=False)
 S=base(C,.06,'raw'); G=np.random.default_rng(42).normal(size=S.shape); G=(G+G.T)/2
 f,g,R=profile(C,S); print('gradient',((profile(C,S+1e-5*G)[0]-profile(C,S-1e-5*G)[0])/2e-5),np.sum(g*G))
 for n,S in methods(C):print('public',n,np.round(evalf(S,z['gt_adj']),3))
 for mode in ('publiclike','uneven','weakfactor','mixed','random'):
  out={}
  for seed in range(20):
   X,a,_=data(seed,mode); C=np.cov(X,rowvar=False)
   for n,S in methods(C):out.setdefault(n,[]).append(evalf(S,a))
  print(mode)
  for n,v in out.items():
   if 'cov' in n: continue
   print(n,np.round(np.mean(v,axis=0),3),np.round(np.min(v,axis=0),3))
