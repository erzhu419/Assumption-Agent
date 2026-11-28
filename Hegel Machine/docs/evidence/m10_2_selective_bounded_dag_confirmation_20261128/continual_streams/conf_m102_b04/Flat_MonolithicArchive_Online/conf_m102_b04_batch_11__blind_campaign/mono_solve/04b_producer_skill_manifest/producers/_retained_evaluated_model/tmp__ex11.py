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
for d in range(1,5):
 A=poly(X,d); print('poly',d,cv(A,Y),cv(A,T),np.std(Y-A@ls(A,Y)))
for name,F in [('exp',lambda p:-np.expm1(-t[:,None]*np.exp(p))),('log',lambda p:np.log1p(t[:,None]*np.exp(p))),('frac',lambda p:t[:,None]/(1+t[:,None]*np.exp(p))),('expmix',lambda p:np.c_[-np.expm1(-t*np.exp(p[0])),np.log1p(t*np.exp(p[1]))]),('exp0',lambda p:np.c_[t,-np.expm1(-t*np.exp(p[0]))])]:
 r=least_squares(lambda p:(F(p)@ls(F(p),T.T)-T.T).ravel(),[-.8,.5] if name!='exp0' else [.5],bounds=(-8,6)); print('time',name,np.exp(r.x),np.std(r.fun))
for key in ['panel_b','panel_c']:
 P=s[key]; U=P['u']; A=poly(U,1)
 for k in P:
  if k=='u':continue
  V=P[k]; print(key,k,'range',V.min(0),V.max(0)); print(np.c_[U,V][:8].round(3));
  for name,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('erf',__import__('scipy.special',fromlist=['erf']).erf),('silu',lambda z:z/(1+np.exp(-z))),('rational',lambda z:z/(1+z*z)),('quadratic',lambda z:z*z),('softplus',lambda z:np.logaddexp(0,z)),('atan',np.arctan)]:
   rms=[]
   for j in range(4):
    y=V[:,j]; w=ls(A,y); rng=np.random.default_rng(11); starts=[w,w*3,-w*3]+list(rng.normal(0,1,(12,5))); solutions=[least_squares(lambda z:f(A@z[:5])+z[5]-y,np.r_[p,0.],max_nfev=160) for p in starts]; rms.append(min(np.std(z.fun) for z in solutions));
   print(name,np.round(rms,6))
