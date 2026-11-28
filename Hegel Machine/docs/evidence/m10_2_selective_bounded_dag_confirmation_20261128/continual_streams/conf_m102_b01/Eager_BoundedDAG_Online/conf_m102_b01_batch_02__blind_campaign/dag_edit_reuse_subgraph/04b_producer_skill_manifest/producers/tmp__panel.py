import itertools,pickle,runpy,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; tr=s['train']['traj']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:len(y)]; n=len(y); d=np.arange(n); ids=list(itertools.combinations(range(4),2)); p=np.column_stack([w[:,i]*w[:,j] for i,j in ids]); q=np.c_[np.ones(n),p]; a=np.c_[q,w,w*w]; ls=lambda X,z:np.linalg.lstsq(X,z,rcond=None)[0]
from scipy.optimize import least_squares
c=s['panel_c']; u=c['u']; v1=c['v1']; v2=c['v2'];
def pair(x):return np.column_stack([x[:,i]*x[:,j] for i,j in ids])
def quad(x):return np.c_[np.ones(len(x)),x,pair(x),x*x]
C=ls(quad(u),v1); print('v1 quad coefficient',C.round(5));
F=quad(w)@C; P=quad(u)@C; print('v2 coefficient on affine + v1',ls(np.c_[np.ones(len(u)),u,P],v2).round(5));
folds=[d[52:],np.argsort(np.linalg.norm(w,axis=1))[-18:]]+[vi for seed in range(10) for vi in np.array_split(np.random.default_rng(seed).permutation(d),6)];
def score(A,z=y):
 e=[]
 for v in folds:
  t=np.setdiff1d(d,v); e.append(np.mean(abs(A[v]@ls(A[t],z[t])-z[v])))
 return np.round([*e[:2],np.mean(e[2:]),max(e[2:])],6)
def design(x,f):return np.c_[np.ones(len(x)),x,f];
for nm,R in [('v1',F),('v1^2',F**2),('v1^3',F**3),('v1*tanh',F*np.tanh(F)),('v1 atan',np.arctan(F)-F),('v1 tanh',np.tanh(F)-F),('v1 asinh',np.arcsinh(F)-F),('v1 cross',pair(F)),('v1 times w',(F[:,:,None]*w[:,None,:]).reshape(n,-1))]:
 print(nm,score(np.c_[q,R]),score(np.c_[q,R],tr))
for g in [.25,.5,1.,1.25,2.]:
 for nm,fn in [('atan',np.arctan),('tanh',np.tanh),('asinh',np.arcsinh),('sin',np.sin),('rational',lambda x:x/(1+x*x))]:
  for kind in [0,1]:
   def D(x,f):return np.c_[design(x,f),fn(g*f)] if kind==0 else np.c_[design(x,f),fn(g*x)];
   M=D(u,P); fit=ls(M,v2); e=np.mean((M@fit-v2)**2)**.5
   print('v2',g,nm,kind,round(e,6),score(np.c_[q,D(w,F)@fit]),score(np.c_[q,D(w,F)@fit,fn(g*F)]))
print('v1 coefficient fit tests',score(np.c_[np.ones(n),F]),score(np.c_[np.ones(n),F],tr)); print('w train/test range',w.min(0),w.max(0))
