import pickle, numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; n=len(x)
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
mono=lambda x,j: np.column_stack([np.prod(x[:,q],1) for q in combinations_with_replacement(range(4),j)])
P=np.c_[np.ones(n),x,mono(x,2)]; C=mono(x,3); D=mono(x,4)
splits={'fixed':np.arange(52,72),'radial':np.argsort(np.linalg.norm(x,axis=1))[-18:]}
for j in range(4):
 order=np.argsort(x[:,j]); splits[f'low{j}']=order[:18]; splits[f'high{j}']=order[-18:]
for i,ix in enumerate(np.array_split(np.random.default_rng(712).permutation(n),4)): splits[f'rand{i}']=ix
scores={};
for label,va in splits.items():
 tr=np.setdiff1d(np.arange(n),va); Z=z[tr].T
 def bas(k):return -np.expm1(-t[:,None]*k)
 rates=least_squares(lambda k:(bas(k)@ls(bas(k),Z)-Z).ravel(),[1.5,.25],bounds=([.7,.02],[4,.65])).x
 U=np.linalg.qr(bas(rates))[0]; Y=np.c_[y[tr],Z.T@U]
 def add(key,p):
  e=np.abs(p[:,1:]@U.T-z[va]); scores.setdefault(key,[]).append([np.mean(abs(p[:,0]-y[va])),e.mean(),e.mean(0).max()])
 def readout(A,pen):
  return A[va]@ls(np.r_[A[tr],np.diag(pen)],np.r_[Y,np.zeros((A.shape[1],3))])
 # Test smooth extrapolation via global cubic and quartic terms plus interpolated residuals.
 g=np.exp(-.23*np.sum((x[:,None]-x[None,:])**2,axis=2)); p=P[tr]
 for trend in ('quad','cubic','quartic'):
  for scale in (.05,.25,1.,4.):
   k=g.copy()
   if trend!='quad': k+=scale*(C@C.T)
   if trend=='quartic': k+=scale*.2*(D@D.T)
   pp=[]
   for j,penalty in enumerate([.001,.006,.006]):
    m=k[np.ix_(tr,tr)]+penalty*np.eye(len(tr)); a=np.block([[m,p],[p.T,np.zeros((15,15))]]); pp.append(np.c_[k[np.ix_(va,tr)],P[va]]@np.linalg.solve(a,np.r_[Y[:,j],np.zeros(15)]))
   add((trend,scale),np.array(pp).T)
 for cpen in (.15,.35,.7): add(('global quartic',cpen),readout(np.c_[P,C,D],np.r_[np.zeros(15),np.full(20,cpen),np.full(35,3*cpen)]))
 # Compare a low-rank quotient and additive ridge terms in projected readouts.
 for law in ('quotient','ridge_tanh','ridge_sin'):
  pred=[]
  for j in range(3):
   target=Y[:,j]; best=None
   for seed in range(7):
    rng=np.random.default_rng(seed)
    def features(q,idx):
     pp=P[idx]; v=pp[:,:5]@q.reshape(2,5).T
     if law=='quotient': return np.c_[pp[:,:5],pp/(1+np.sum(v*v,axis=1,keepdims=True))]
     f=np.tanh if law=='ridge_tanh' else np.sin
     return np.c_[pp,f(v),v*f(v)]
    def res(q):
     A=features(q,tr); return np.r_[A@ls(A,target)-target,.008*q]
    o=least_squares(res,rng.normal(0,.65,10),max_nfev=250,bounds=(-5.,5.))
    if best is None or o.cost<best.cost: best=o
   pred.append(features(best.x,va)@ls(features(best.x,tr),target))
  add(law,np.array(pred).T)
for name,v in scores.items():
 a=np.array(v); print(name,'fixed',np.round(a[0],5),'tail worst',np.round(a[1:10].max(0),5),'tail mean',np.round(a[1:10].mean(0),5),'random worst',np.round(a[10:].max(0),5))
