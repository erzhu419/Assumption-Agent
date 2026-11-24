import numpy as np

def data(seed, mode):
 r=np.random.default_rng(seed); p=24; i,j=np.triu_indices(p,1)
 if mode in ('publiclike','uneven','weakfactor','mixed'):
  a=np.eye(p,k=1,dtype=bool); a[0,-1]=True; pool=np.where(~a[i,j])[0]; ex=r.choice(pool,4,replace=False); a[i[ex],j[ex]]=True; a|=a.T
 else:
  a=np.zeros((p,p),bool); ex=r.choice(len(i),26,replace=False); a[i[ex],j[ex]]=True; a|=a.T
 S=np.eye(p)*2.2
 w=np.full(a[i,j].sum(),.52) if mode not in ('mixed','random') else r.uniform(.32,.69,a[i,j].sum())
 S[i[a[i,j]],j[a[i,j]]]=w*r.choice([-1,1],len(w)); S+=np.triu(S,1).T
 if mode=='random': S+=np.diag(r.uniform(-.28,.28,p)); S+=np.eye(p)*max(0,.55-np.linalg.eigvalsh(S)[0]);
 B=r.normal(size=(p,2))*.9
 if mode=='uneven': B[:,0]*=np.exp(r.normal(scale=.7,size=p)); B[:,1]*=np.exp(r.normal(scale=.7,size=p));
 if mode=='weakfactor': B[:,1]*=.25
 X=r.normal(size=(900,p))@np.linalg.cholesky(np.linalg.inv(S)).T+r.normal(size=(900,2))@B.T
 return X,a,S

def base(C, lam, typ):
 T=np.linalg.inv(C); p=len(T); d=np.diag(T).copy(); E=np.zeros_like(T); k=np.median(d)
 if typ=='cov':
  l,q=np.linalg.eigh(C); return np.linalg.inv(C-(q[:,-2:]*(l[-2:]-1/k))@q[:,-2:].T)
 for it in range(65):
  V=np.diag(d)+E-T; l,q=np.linalg.eigh(V); L=(q[:,-2:]*np.maximum(l[-2:],0))@q[:,-2:].T; F=T+L
  d=np.diag(F).copy()
  if typ=='norm': th=lam*np.sqrt(np.outer(d,d))
  else: th=lam*k
  E=np.sign(F)*np.maximum(np.abs(F)-th,0); np.fill_diagonal(E,0)
 return F

def evalf(M,a):
 i,j=np.triu_indices(len(M),1); d=np.diag(M); v=np.abs(M[i,j])/np.sqrt(d[i]*d[j]); y=a[i,j];
 fs=[]
 for pred in [v>t for t in (.105,.12,.135,.15)] + [v>=np.partition(v,-k)[-k] for k in (26,y.sum())]:
  fs.append(2*(y&pred).sum()/(y.sum()+pred.sum()))
 return fs

if __name__=='__main__':
 for mode in ('publiclike','uneven','weakfactor','mixed','random'):
  xx=[data(s,mode) for s in range(35)]
  for typ,lam in [('cov',0),('raw',.06),('raw',.10),('raw',.14),('norm',.10)]:
   out=np.array([evalf(base(np.cov(x,rowvar=False),lam,typ),a) for x,a,S in xx]); print(mode,typ,lam,np.round(out.mean(0),3),np.round(out.min(0),3))
