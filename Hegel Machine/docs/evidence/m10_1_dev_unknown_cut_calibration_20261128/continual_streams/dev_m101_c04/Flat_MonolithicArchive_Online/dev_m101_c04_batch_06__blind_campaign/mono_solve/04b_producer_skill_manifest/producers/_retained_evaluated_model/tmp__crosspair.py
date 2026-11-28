import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; z=np.load('/tmp/stages.npz')['b']; U=np.c_[np.ones(len(z)),z]; pa=np.array(list(itertools.combinations_with_replacement(range(4),2))); Q=np.prod(z[:,pa],axis=2); F=np.c_[U,Q]; pairs=z[:,[0,1]]*z[:,[2,3]]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def qmode(i):
 u=U[i]; q=Q[i]; m=ls(q-u@ls(u,q),T[i]-u@ls(u,T[i])); _,_,v=np.linalg.svd(m,full_matrices=False); return Q@(m@v[0])
def fit(i,j,f,g,method):
 u=U[i]; yy=y[i]; a=f[i]; aux=g[i]; p0=ls(F[i],yy)[1:5]; p0/=np.linalg.norm(p0)
 def fun(u,a,aux,p):
  t=u@p; d=1/(1+t*t)
  if method=='rat': w=t*d
  elif method=='parity': w=np.c_[t*d,d]
  elif method=='linear': w=u*d[:,None]
  elif method=='contrast': w=np.c_[u,aux]*d[:,None]
  else: w=np.tanh(t)
  return np.c_[a,w]
 def res(p):
  b=fun(u,a,aux,p); return b@ls(b,yy)-yy
 sol=min((least_squares(res,np.r_[0,p0*v],bounds=(-6,6),max_nfev=125) for v in [.7,1.2,2.2]),key=lambda r:r.cost); w=ls(fun(u,a,aux,sol.x),yy)
 return np.mean(abs(sol.fun)),np.mean(abs(fun(U[j],f[j],g[j],sol.x)@w-y[j])),sol.x,w
rng=np.random.default_rng(9); folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(rng.permutation(72),5)]; out={}
for k,(i,j) in enumerate(folds):
 q=qmode(i); ps=pairs.sum(axis=1); g=np.c_[q,ps];
 for layout,f in [('pair',np.c_[U,g]),('separate',np.c_[U,q,pairs]),('full',F)]:
  for method in ['rat','parity','tanh','linear','contrast']:
   a,b,p,w=fit(i,j,f,g,method); out.setdefault((layout,method),[]).append(b)
   if k==0:print(layout,method,'fit',round(a,5),'cv',round(b,5),'p/w',np.round(p,4),np.round(w,4))
for k,v in out.items(): print(k,np.round(v,5),'mean/max',np.round([np.mean(v),np.max(v)],5))
