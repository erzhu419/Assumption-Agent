from pulsecheck import y,rm
from wallstudy import *

def axis(k,u,h,r,typ):
 n=len(k); A=arc._axis(k,u,h,(0,0)).tolil(); name,ff,ad=typ
 for i,j,p in [(0,1,n-1),(n-1,n-2,0)]:
  out = u[i] if i==n-1 else -u[i]
  if name=='open': A[i,i]+=r/h
  if name=='sink': A[i,i]+=ff*k[i]/h**2
  if name=='reflect':
   A[i,j]*=ff; A[i,i]=-A[i,j]
  if name in ('wrap','wrap_full','wrap_rep'):
   a=k[i] if ff==0 else (k[i]+k[p])/2
   if name=='wrap_full': A[i,j]=-(k[i]+k[j])/(2*h*h)
   if name=='wrap_rep': A[i,j]=0
   A[i,p]=-a/(h*h)
   if ad:
    A[i,j]-=max(out,0)/h; A[i,p]-=max(-out,0)/h
   A[i,i]=-A[i,j]-A[i,p]
 return A.tocsc()

def fw(tx,ty):
 A=axis(s['kappa_x'],s['vel_x'],dx,rob[0],tx); B=axis(s['kappa_y'],s['vel_y'],dy,rob[2],ty)
 L=sparse.kron(sparse.eye(ny),A,format='csc')+sparse.kron(B,sparse.eye(nx),format='csc')+vdiag
 return np.vstack([splu(L+u*eye).solve(rhs)[rows] for u in s['mu_shifts']])

types=[('closed',0,0),('open',0,0),('sink',1,0),('sink',2,0),('reflect',2,0),('reflect',.5,0)]+list(itertools.product(['wrap','wrap_full','wrap_rep'],[0,1],[0,1]))
a=[]
for tx,ty in itertools.product(types,repeat=2):
 e=rm(w@fw(tx,ty).T-y); a.append((np.linalg.norm(e),tx,ty,e.tolist()))
print('best pairs',*sorted(a)[:12],sep='\n');
for t in range(3):
 best=min(a,key=lambda x:sum(e**2 for j,e in enumerate(x[3]) if j!=t)); print('holdout',t,best)
