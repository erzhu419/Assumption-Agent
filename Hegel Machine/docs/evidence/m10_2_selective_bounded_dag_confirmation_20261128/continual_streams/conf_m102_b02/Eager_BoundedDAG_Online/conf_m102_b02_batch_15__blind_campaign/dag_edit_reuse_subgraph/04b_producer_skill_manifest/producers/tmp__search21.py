import sys
sys.path.insert(0,'/holdout_task')
from node_00 import _coordinates
from probe21 import s,np,D,reg,least_squares,combinations
from itertools import combinations_with_replacement
from sklearn.linear_model import Lars, Ridge
from scipy.special import erf
from scipy.optimize import minimize_scalar
tr=s['train']; n=len(tr['x']); X=tr['x']; z,b,c,d=_coordinates(s,X); h=np.cos(2.6*z); t=s['eval_times']; E=lambda p:-np.expm1(-t[:,None]*np.array(p)); T=tr['traj']; W=np.c_[tr['y'],reg(E([1.8,.42]),T.T).T]; ix=np.arange(52); jx=np.arange(52,n)
base={'z':z,'x':X,'b':b,'c':c,'d':d,'h':h,'p':z*np.roll(z,-1,axis=1)}
def lib(base):
 F={}
 def add(k,x):
  if x.ndim==1:F[k]=x
  else:
   for j,v in enumerate(x.T):F[f'{k}{j}']=v
 for k,x in base.items():
  add(k,x)
  for a in [.5,1.25,2.15,2.6,4.3,5.2]:
   for fun in [np.sin,np.cos,np.tanh]:add(f'{fun.__name__}{a}({k})',fun(a*x))
 for k,l in combinations_with_replacement(base,2):
  for j,i in ((j,i) for j in range(4) for i in range(j if k==l else 0,4)):
   f=base[k][:,j];g=base[l][:,i];add(f'{k}{j}*{l}{i}',f*g)
   if k in ('z','b','h') and l in ('z','b','h') and (k,j)!=(l,i):
    for r in [-2,-1,-.5,.5,1,2]:
     for a in [1.25,2.15,2.6]:
      for fun in [np.sin,np.cos,np.tanh]:add(f'{fun.__name__}{a}({k}{j}+{r}{l}{i})',fun(a*(f+r*g)))
 for k,x in base.items():
  for ix in combinations_with_replacement(range(4),3):add(f'{k}^{ix}',x[:,ix].prod(1))
 for k in ('b','c','d','z'):
  for l in ('b','c','d','z'):
   for a in [.22,.5,1.25,2.15]:
    for i in range(4):
     for j in range(4):add(f'{k}{i}/(1+{a}{l}{j}^2)',base[k][:,i]/(1+a*base[l][:,j]**2))
 return F
F=lib(base); keys=np.array(list(F)); G=np.column_stack(list(F.values())); G=(G-G[ix].mean(0))/np.maximum(G[ix].std(0),1e-8)
for H in range(3):
 y=W[:,H]; y0=y-y[ix].mean(); corr=G[ix].T@y0[ix]; top=np.argsort(-abs(corr))[:8]; print('TOP',H,list(zip(keys[top],np.round(corr[top],2))))
 paths=[]
 for gs in [np.arange(G.shape[1]),np.where(np.array([not any(w in k for w in ('x','d','c')) for k in keys]))[0]]:
  path=Lars(n_nonzero_coefs=32,fit_intercept=False).fit(G[ix][:,gs],y0[ix]);
  for j,coef in enumerate(path.coef_path_.T):
   sel=gs[np.flatnonzero(coef)]; f=D(G[:,sel]); w=reg(f[ix],y[ix]); p=f@w; err=(abs(p[ix]-y[ix]).mean(),abs(p[jx]-y[jx]).mean());paths.append((err,keys[sel].tolist(),w))
 for q in sorted(paths,key=lambda q:q[0][1])[:3]:print('LARS',H,q[0],q[1],np.round(q[2],3))
 for k,g in base.items():
  f=D(g);w=reg(f[ix],y[ix]);print('LIN',H,k,np.round(w,3),np.round([abs((f@w-y)[j]).mean() for j in [ix,jx]],4))
# Profile small multi-index models with calibrated coordinates and nonlinear readout.
Z=np.c_[z,b,c,d]; Z=(Z-Z[ix].mean(0))/np.maximum(Z[ix].std(0),1e-10); rng=np.random.default_rng(441)
for H in range(3):
 y=W[:,H]; opts=[]
 for kk in [4,8,12,16]:
  for fn in [np.sin,np.tanh,np.cos,lambda t:t*t,lambda t:t/(1+t*t)]:
   for typ in [0,1,2]:
    P=Z[:,:kk]; f0=D(P[:,:min(kk,8)] if typ else P[:,:0]); m=kk+1
    def f(q):return np.c_[f0,fn(P@q[:kk]+q[kk]), fn(P@q[m:m+kk]+q[m+kk])] if typ==2 else np.c_[f0,fn(P@q[:kk]+q[kk])]
    def e(q):
     g=f(q)[ix]; return g@reg(g,y[ix])-y[ix]
    for a in range(3):
     sol=least_squares(e,rng.normal(0,.5,m*(2 if typ==2 else 1)),max_nfev=100); g=f(sol.x); w=reg(g[ix],y[ix]); rr=abs(g@w-y); opts.append((rr[ix].mean(),rr[jx].mean(),kk,getattr(fn,'__name__','fn'),typ,sol.x,w))
 for row in sorted(opts,key=lambda q:q[1])[:3]:print('INDEX',H,row[:5],np.round(row[5],3),np.round(row[6],3))
