import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): v=runpy.run_path('/tmp/chamber_structure.py')
p,l,a,h,t,dt,Y,g,tr,te,align=(v[k] for k in ('p','l','a','h','t','dt','Y','g','tr','te','align'))
np.set_printoptions(precision=5,suppress=True,linewidth=150)
X=align([g]); r=h[2]+X@h[3:]; N,K=Y.shape
D=np.c_[np.ones(N),l,p,X]; B=np.linalg.lstsq(D,Y,rcond=None)[0]; print('linear coefficient trajectories (offset, hold, six intervals, four features)');print(B)
def mem(q):
 return np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0]/q)*-np.expm1(-dt/q)
def temp(q): return np.r_[-np.expm1(-t/q),1/q]
lin=np.r_[t,1.]; ones=np.ones(N)
def fit(mode,X,itr):
 r=h[2]+X@h[3:]; rr=np.c_[ones,X] if mode.endswith('free') else np.c_[ones,r]
 outer=lambda z,f:z[:,None]*f
 def basis(q):
  if mode.startswith('same'): u,w=q[:2]
  else: u,w=q[2:4]
  G1=l-p@mem(u);G2=l-p@mem(w); F1,F2=map(temp,q[:2]);
  cols=[outer(G1,F1),outer(G2,F2),outer(l,lin)]
  if 'cross' in mode: cols.extend([outer(G1,F2),outer(G2,F1),outer(l,F1),outer(l,F2)])
  if 'drift' in mode: cols.extend([outer(z,lin) for z in rr.T]);
  cols.extend([outer(z,F2) for z in rr.T]);
  if 'fast' in mode: cols.extend([outer(z,F1) for z in rr.T]);
  return np.stack(cols,axis=-1)
 def calc(q,out=False):
  A=basis(np.exp(q)); C=np.linalg.lstsq(A[itr].reshape(-1,A.shape[-1]),Y[itr].ravel(),rcond=1e-9)[0]; P=A@C
  return (np.exp(q),C,P) if out else (P[itr]-Y[itr]).ravel()
 u0=[.65,1.8] + ([] if mode.startswith('same') else list(a)); z=least_squares(calc,np.log(u0),bounds=(np.log(.08),np.log(25)),max_nfev=180);return calc(z.x,True)
for mode in ['same-drift','separate-drift','separate-driftfast','separate-crossdrift','separate-driftfree','separate-driftfastfree','separate-crossdriftfree']:
 q,C,P=fit(mode,X,np.arange(N)); E=P-Y
 errors=np.array([fit(mode,X,np.delete(np.arange(N),i))[2][i]-Y[i] for i in range(N)])
 print(mode,'q',q,'C',C,'rms in,loo',np.sqrt(np.mean(E**2)),np.sqrt(np.mean(errors**2)),'worst time,rate,max',np.max(np.sqrt(np.mean(errors[:,:-1]**2,axis=0))),np.sqrt(np.mean(errors[:,-1]**2)),np.max(abs(errors)))
 if mode in ('separate-drift','separate-crossdriftfree'): print('residuals last and rate',E[:,[-2,-1]].T)
