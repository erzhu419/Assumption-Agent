from probe import *
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from itertools import combinations
x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; n=len(x); Y=np.c_[y,z]; I=np.arange(n)
folds=[I[56:]]+list(np.array_split(np.random.default_rng(42).permutation(n),4))+[np.argsort((x*x).sum(1))[-12:]]
def metrics(pred):
 E=np.array([np.mean(abs(p-Y[v]),axis=0) for v,p in zip(folds,pred)]); return np.r_[E[:5,0].mean(),E[:5,1:].mean(),E[:5,0].max(),E[:5,1:].max(),E[-1,0],E[-1,1:].max()]
def score(F,L):
 return metrics([F[v]@solve(F[t].T@F[t]+np.diag(L),F[t].T@Y[t],assume_a='pos') for v in folds for t in [np.setdiff1d(I,v)]])
P=[poly(x,d) for d in range(1,7)]; deg=np.array([k for k in range(7) for _ in combinations_with_replacement(range(4),k)]); out=[]
for d in range(3,7):
 F=P[d-1]; dd=deg[:F.shape[1]]
 for a in [1e-5,.0001,.001,.01,.04,.16]:
  for m in [3,12,48]:
   l=a*m**(dd-3.);l[dd<2]=1e-10;out.append((score(F,l),('poly',d,a,m)))
D=cdist(x,x,'sqeuclidean'); S=(x@x.T)/4
for d in [1,2,3]:
 F=P[d-1];
 for a in [.25,.5,1.,2.,4.,8.]:
  for tag,K in [('gauss',np.exp(-D/(2*a*a))),('rq',(1+D/(2*a*a))**-2),('m52',(1+np.sqrt(5*D)/a+5*D/(3*a*a))*np.exp(-np.sqrt(5*D)/a)),('expdot',np.exp(S/a))]:
   for l in [1e-6,.0001,.003,.03]:
    pred=[]
    for v in folds:
     t=np.setdiff1d(I,v); Q=F[t]; A=K[np.ix_(t,t)]+100*Q@Q.T+np.eye(len(t))*l; b=solve(A,Y[t],assume_a='pos'); pred.append((K[np.ix_(v,t)]+100*F[v]@Q.T)@b)
    out.append((metrics(pred),(tag,d,a,l)))
for h,m in [(0,4),(1,5)]:
 print('best average head',h,sorted(out,key=lambda q:q[0][h])[:8]);print('best average+edge',h,sorted(out,key=lambda q:q[0][h]+.35*q[0][m])[:8]);
F=P[2]; b=fit(F,Y); inds=[q for d in range(4) for q in combinations_with_replacement(range(4),d)];
for j in [0,1,8]:print('coeff',j,sorted(zip(inds,b[:,j]),key=lambda q:-abs(q[1]))[:24]);
T=np.linalg.svd(z,full_matrices=False)[2][:2]; c=z@T.T
print('rank latent corrs with y',np.corrcoef(np.c_[y,c].T))
