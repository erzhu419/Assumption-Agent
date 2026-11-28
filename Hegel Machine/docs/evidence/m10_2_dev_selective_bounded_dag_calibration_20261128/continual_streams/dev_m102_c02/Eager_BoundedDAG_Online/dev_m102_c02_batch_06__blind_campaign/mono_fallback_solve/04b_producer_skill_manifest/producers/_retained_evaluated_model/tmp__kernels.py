import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
q=np.load('/tmp/coords2.npz'); Y=q['Y'][:,:9]; D={k:q[k] for k in q if k not in ['B','Y']}; n=52; X=D['x']; N=len(X); rng=np.random.default_rng(33)
def score(K,lam,idx=np.arange(52),test=np.arange(52,72)):
 r=K[np.ix_(test,idx)]@solve(K[np.ix_(idx,idx)]+lam*np.eye(len(idx)),Y[idx],assume_a='pos')-Y[test];return np.mean(abs(r[:,0])),np.mean(abs(r[:,1:]))
K={};out=[]
for name,Z in D.items():
 Z=(Z-Z[:n].mean(0))/np.maximum(Z[:n].std(0),.0001); dist=cdist(Z,Z,'sqeuclidean'); dot=Z@Z.T/Z.shape[1];
 for k in [1,2,3,4]:K[name,'poly',k]=(1+dot)**k
 for w in [.3,.6,1,1.5,2.5,4,7]:
  r=np.sqrt(5*dist)/w; K[name,'m52',w]=(1+r+r*r/3)*np.exp(-r);K[name,'rbf',w]=np.exp(-dist/(2*w*w))
for key,K0 in K.items():
 for lam in [1e-7,1e-4,.002,.02,.1]:
  a,b=score(K0,lam); out.append((a,b,key,lam))
print('best kernel y',sorted(out)[:8]); print('best kernel s',sorted(out,key=lambda z:z[1])[:8]);
