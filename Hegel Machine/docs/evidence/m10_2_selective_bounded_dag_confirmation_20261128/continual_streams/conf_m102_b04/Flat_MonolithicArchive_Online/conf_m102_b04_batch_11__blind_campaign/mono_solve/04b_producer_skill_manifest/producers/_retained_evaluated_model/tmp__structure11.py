import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u,v=a['u'],a['v']; keep=np.max(abs(v-u),axis=1)<1
ls=lambda A,b:np.linalg.lstsq(A,b,rcond=None)[0]
D=np.array([ls(np.c_[np.ones(keep.sum()),u[keep,j]],v[keep,j]) for j in range(4)])
tr=s['train']; X=(tr['x']-D[:,0])/D[:,1]; T=tr['traj']; Y=tr['y']; t=s['eval_times'];
def poly(x,d):
 return np.column_stack([np.ones(len(x))]+[x[:,i].prod(1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def cv(A,H):
 return np.mean(abs(A[52:]@ls(A[:52],H[:52])-H[52:]),axis=0)
from itertools import combinations
from scipy.optimize import minimize_scalar
from scipy.special import erf
F=lambda p:-np.expm1(-t[:,None]*np.exp(p)); p=least_squares(lambda p:(F(p)@ls(F(p),T[:52].T)-T[:52].T).ravel(),[-.8,.5]).x; Z=ls(F(p),T.T).T; H=np.c_[Y,Z];
L=list(combinations(range(4),2));
def prod(w): return np.column_stack([w[:,i]*w[:,j] for i,j in L]);
def feat(x,kind,k):
 if kind=='cos':return np.cos(k*x)
 if kind=='exp':return np.exp(-k*x*x)
 if kind=='rat':return 1/(1+k*x*x)
 if kind=='sin':return np.sin(k*x)
 if kind=='tanh':return np.tanh(k*x)
 if kind=='odd':return x/(1+k*x*x)
 if kind=='cosh':return np.logaddexp(k*x,-k*x)
 if kind=='sinc':return np.sinc(k*x/np.pi)
 if kind=='erf':return erf(k*x)
 if kind=='sqsin':return x*np.sin(k*x)
 if kind=='silu':return x*np.tanh(k*x)
def fitprof(design, target, n, bounds=(.2,6.)):
 def goal(k):
  a=design(k)[:n]; z=target[:n]; return np.mean((a@ls(a,z)-z)**2)
 g=np.linspace(*bounds,100); ix=min(range(1,len(g)-1),key=lambda i:goal(g[i])); k=minimize_scalar(goal,bounds=(g[ix-1],g[ix+1]),method='bounded').x; A=design(k); b=ls(A[:n],target[:n]); e=A@b-target; return k,np.mean(abs(e[:n]),0),np.mean(abs(e[n:]),0) if n<len(e) else '', b, A@b
A=np.c_[np.ones(72),X**2,X**4,prod(X**2)]; print('even quartic',cv(A,H),np.mean(abs(A[:52]@ls(A[:52],H[:52])-H[:52]),0));print('even coefficients',ls(A[:52],H[:52]).round(4))
for kind in ('cos','exp','rat','cosh','sinc','sqsin','silu'):
 for typ in range(4):
  def de(k):
   W=feat(X,kind,k); return np.c_[np.ones(len(X)),W,*[(),(prod(W),),(prod(X**2),),(prod(W), X**2)][typ]]
  r=fitprof(de,H,52); print('readout',kind,typ,round(r[0],4),r[1].round(5),r[2].round(5))
for key in ['panel_b','panel_c']:
 P=s[key]; U=P['u']; A=poly(U,1); Q=poly(U,2); V=np.column_stack([v for k,v in P.items() if k!='u']); print(key,'poly2',np.std(Q@ls(Q,V)-V,0));
 for kind in ('cos','sin','tanh','odd','erf','rat','silu'):
  for typ in range(4):
   def de(k):
    W=feat(U,kind,k); return np.c_[A if typ!=0 else A[:,:1],W,*[(),(),(prod(U),),(prod(W),)][typ]]
   r=fitprof(de,V,len(U)); print(key,kind,typ,round(r[0],3),r[1].round(4))
 # Shared nonlinear dictionary of ridge-free coordinate pair arguments.
 for form in ('sin','cos','tanh'):
  dif=np.column_stack([U[:,i]-U[:,j] for i,j in L]); plus=np.column_stack([U[:,i]+U[:,j] for i,j in L]); times=prod(U);
  for D in (np.c_[U,dif,plus],np.c_[U,times],np.c_[dif,plus]):
   r=fitprof(lambda k: np.c_[A,feat(D,form,k)],V,len(U));print(key,form,D.shape[1],round(r[0],3),r[1].round(4))
