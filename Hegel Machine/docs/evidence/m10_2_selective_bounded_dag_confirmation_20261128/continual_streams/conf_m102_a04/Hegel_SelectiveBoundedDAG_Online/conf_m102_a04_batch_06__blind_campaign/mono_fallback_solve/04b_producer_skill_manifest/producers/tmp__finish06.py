from cv06 import *
from itertools import combinations,product
from scipy.optimize import least_squares
T=np.c_[y,t]; n=len(z); one=np.ones(n); Z=z*np.roll(c,1,axis=1); D=c*d
base=np.c_[one,Z,D,c,d]; a=np.arange(52); b=np.arange(52,72)
Q=np.linalg.qr(base[a])[0]; R=T[a]-Q@(Q.T@T[a]); print('base residual svd',np.linalg.svd(R,compute_uv=False)); pr=lambda f:f-Q@(Q.T@f)
cols=[]; names=[]
def add(name,h):names.append(name);cols.append(h)
blocks={'z':z,'c':c,'d':d,'zc':Z,'cd':D}; vv=[(f'{k}{i}',v[:,i]) for k,v in blocks.items() for i in range(4)]+[('sumzc',Z.sum(1)),('sumcd',D.sum(1)),('sumc',c.sum(1)),('sumd',d.sum(1))]
for (k,x),(l,h) in combinations(vv,2):
 add(k+'*'+l,x*h)
 for g in (-1,-.5,.5,1):
  q=x+g*h; nm=f'{k}{g:+g}{l}';
  for f in (0.5,1,1.25,2.4,3.2):
   add(f'cos{f}({nm})',np.cos(f*q)); add(f'sin{f}({nm})',np.sin(f*q)); add(f'tanh{f}({nm})',np.tanh(f*q));
  add(f'rat({nm})',q/(1+.22*q*q)); add(f'sqr({nm})',q*q/(1+.22*q*q));
for k,x in vv:
 for g in [.5,1,1.25,2.4,3.2]:
  add(f'cos{g}({k})',np.cos(g*x));add(f'sin{g}({k})',np.sin(g*x));add(f'tanh{g}({k})',np.tanh(g*x))
A=np.column_stack(cols); H=pr(A[a]); N=np.linalg.norm(H,axis=0).clip(1e-9); U,ss,V=np.linalg.svd(R,full_matrices=False)
for j in range(3):
 sc=abs(H.T@U[:,j])/N; jj=np.argsort(sc)[-12:][::-1]; print(j,[(names[i],round(sc[i],4)) for i in jj]);
for j in range(9):
 sc=abs(H.T@R[:,j])/N/np.linalg.norm(R[:,j]); jj=np.argsort(sc)[-3:][::-1]; print('out',j,[(names[i],round(sc[i],4)) for i in jj]);
# Separate output: scalar law is already near panel noise. For state keep mechanistic base, then model residual in the dominant temporal directions.
def fn(a,b,lam=0.,k=0,fixed=()):
 B=np.c_[base,A[:,list(fixed)]]; C=m._ls(B[a],T[a]); P=B@C; r=T[a,1:]-P[a,1:]; Q=np.linalg.qr(B[a])[0]; X=A[a]-Q@(Q.T@A[a]); nx=np.linalg.norm(X,axis=0).clip(1e-8); selected=[]
 if lam:
  # Residual GP with a projected normalized operator kernel.
  M=A-B@m._ls(B[a],A[a]); M/=np.maximum(np.linalg.norm(M[a],axis=0),.03); K=M@M[a].T/A.shape[1]; P[b,1:]+=K[b]@np.linalg.solve(K[a]+lam*np.eye(len(a)),r)
 else:
  for _ in range(k):
   sc=np.linalg.norm(X.T@r,axis=1)/nx; sc[selected]=-1;selected.append(sc.argmax()); G=np.c_[B,A[:,selected]]; C=m._ls(G[a],T[a,1:]); P[:,1:]=G@C; r=T[a,1:]-P[a,1:];Q=np.linalg.qr(G[a])[0];nx=np.linalg.norm(A[a]-Q@(Q.T@A[a]),axis=0).clip(1e-8)
 return P[b,0],P[b,1:]
for k in (0,2,4,6,9,12,16):
 E=evaluate(lambda a,b:fn(a,b,k=k));print('k',k,E[0].round(5),E[1:9].max(0).round(5),E[9:].max(0).round(5))
for lam in (1e-6,1e-5,1e-4,1e-3,1e-2):
 E=evaluate(lambda a,b:fn(a,b,lam=lam));print('ridge',lam,E[0].round(5),E[1:9].max(0).round(5),E[9:].max(0).round(5))
