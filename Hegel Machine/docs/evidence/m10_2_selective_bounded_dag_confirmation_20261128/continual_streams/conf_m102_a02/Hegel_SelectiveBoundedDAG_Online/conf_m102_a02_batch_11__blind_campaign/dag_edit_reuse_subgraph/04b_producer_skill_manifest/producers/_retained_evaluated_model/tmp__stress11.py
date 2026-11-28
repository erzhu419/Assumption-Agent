import numpy as np,runpy
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=np.load('/tmp/read11.npz'); q,b,c,y,z,t=[s[k][:72] if k!='t' else s[k] for k in ('q','b','c','y','z','t')]; n=len(y); o=np.ones(n); A=np.c_[o,b,c]; Q=np.c_[o,q,*[q[:,i]*q[:,j] for i,j in combinations_with_replacement(range(4),2)]]; ls=lambda f,y:np.linalg.lstsq(f,y,rcond=None)[0]; base=runpy.run_path('/holdout_task/node_01.py'); folds=[np.arange(n-20,n)]+list(np.array_split(np.random.default_rng(123).permutation(n),5)); models=[]
def make(p,kind,key,x,b,c,A,Q):
 h=x@p[1:]+p[0]; g=np.tanh(h) if kind=='tanh' else 1/(1+h*h); o=np.ones(len(x)); g0=1/(1+t[0]*x[:,2]**2); prior=np.c_[o,c,b*g0[:,None],b[:,3]*x[:,2]**3*g0]; bases=[(A,np.c_[o,b]),(Q,Q[:,:5]),(A,np.c_[o,x]),(prior,np.c_[o,b]),(prior,np.c_[o,x]),(np.c_[o,c],np.c_[o,b]),(A,b),(prior,b),(np.c_[o,c],np.c_[b,b*x[:,2,None]**3])]; L,R=bases[key]; return np.c_[L,R*g[:,None]]
def fit(I,J,kind,key):
 arrays=[q,b,c,A,Q]; train=[x[I] for x in arrays]; yy=y[I];
 def residual(p):
  f=make(p,kind,key,*train); return f@ls(f,yy)-yy
 starts=np.r_[np.c_[np.zeros(4),np.eye(4)* (6 if kind=='tanh' else np.sqrt(t[1]))],np.c_[np.zeros(4),np.eye(4)* (2 if kind=='tanh' else np.sqrt(t[0]))]]
 opt=min((least_squares(residual,p,bounds=([-3,-15,-15,-15,-15],[3,15,15,15,15]),max_nfev=65) for p in starts),key=lambda r:r.cost); F=make(opt.x,kind,key,*arrays)
 return abs(F[J]@ls(F[I],yy)-y[J]).mean(),opt.x,np.mean(abs(opt.fun))
by=[]
for J in folds:
 I=np.setdiff1d(np.arange(n),J); v=np.r_[I,J]; by.append(abs(base['_scalar'](q[v],b[v],c[v],t,y[I])-y[J]).mean())
print('current nested splits',np.round(by,6),flush=True)
for kind in ('tanh','rational'):
 for key in range(9):
  res=[fit(np.setdiff1d(np.arange(n),J),J,kind,key) for J in folds]; errs=np.array([r[0] for r in res]); print(kind,key,np.round(errs,5), 'worst', round(max(errs),6),'mean',round(errs.mean(),6),'gate',np.round(res[0][1],4),'train',round(res[0][2],6),flush=True)
