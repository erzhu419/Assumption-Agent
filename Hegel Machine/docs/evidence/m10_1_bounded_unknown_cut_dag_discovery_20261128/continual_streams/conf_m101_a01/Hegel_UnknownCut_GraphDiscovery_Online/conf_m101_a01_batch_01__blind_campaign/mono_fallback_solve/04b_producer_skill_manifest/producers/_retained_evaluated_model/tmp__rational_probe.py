from graph_probe import *
from scipy.optimize import least_squares
from scipy.linalg import eigh
q=frames(x)['a-']; L=poly(q,1); P=poly(q,2); T=poly(q,3)
def candidate(g,j,mode,a,b):
 rng=np.random.default_rng(42); V=np.linalg.lstsq(P[a],g[a],rcond=None)[0]; H=np.zeros((4,4))
 for (i,k),v in zip(combinations_with_replacement(range(4),2),V[5:]):H[i,k]=H[k,i]=v/(1+(i!=k))
 _,ev=eigh(H); starts=[np.r_[0,w*f] for w in [ev[:,0],ev[:,-1],ev[:,0]+ev[:,-1],ev[:,0]-ev[:,-1],*np.eye(4)] for f in [.55,1.5]]+[rng.normal(0,.65,5) for _ in range(6)]
 def design(w,i):
  v=L[i]@w; z=v*v
  if mode=='rat2':return P[i]/(1+z[:,None])
  if mode=='rat3':return T[i]/(1+z[:,None])
  if mode=='ratL':return np.c_[L[i],P[i]/(1+z[:,None])]
  if mode=='ratProd':return np.c_[L[i],L[i]*((1+z)**-1)[:,None]]
  if mode=='tanhProd':return np.c_[P[i],L[i]*np.tanh(v)[:,None]]
  if mode=='sinProd':return np.c_[P[i],L[i]*np.sin(v)[:,None]]
  if mode=='softProd':return np.c_[P[i],L[i]*np.logaddexp(v,-v)[:,None]]
  if mode=='ratSqProd':return np.c_[P[i],L[i]/(1+z[:,None])]
 def get(w):
  d=design(w,a); c=np.linalg.solve(d.T@d+.00005*np.eye(d.shape[1]),d.T@g[a]);return d,c
 def fun(w):
  d,c=get(w);return np.r_[d@c-g[a],.007071*c,.0003*w]
 runs=[]
 for w in starts:
  r=least_squares(fun,w,bounds=(-5,5),max_nfev=100); runs.append((r.cost,r.x))
 r=min(runs,key=lambda b:b[0]); d,c=get(r[1]); pred=design(r[1],b)@c
 return np.mean(abs(pred-g[b])),np.mean(abs(d@c-g[a])),r[1]
for j in (0,1):
 for mode in ('rat2','rat3','ratL','ratProd','tanhProd','sinProd','softProd','ratSqProd'):
  o=[candidate(Y[:,j],j,mode,a,b) for a,b in splits[:4]];print(j,mode,np.round([k[:2] for k in o],4).T,'arg',np.round(o[0][2],3),flush=True)
