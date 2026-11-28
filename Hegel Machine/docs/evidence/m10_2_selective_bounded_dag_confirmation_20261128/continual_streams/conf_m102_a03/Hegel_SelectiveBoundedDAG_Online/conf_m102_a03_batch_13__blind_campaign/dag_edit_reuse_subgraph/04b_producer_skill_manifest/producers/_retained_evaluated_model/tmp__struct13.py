import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; n=len(y)
def fit(F,Y):return np.linalg.lstsq(F,Y,rcond=1e-10)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d,start=0):return np.column_stack([np.prod(x[:,p],axis=1) for k in range(start,d+1) for p in it.combinations_with_replacement(range(x.shape[1]),k)])
def design(u,j):
 a=u[:,j];b=u[:,(j+1)%4];return np.c_[np.ones(len(u)),a,b,a*b,a**3]
u,v=s['panel_a'].values(); co=[]
for j in range(4):
 F=design(u,j); ok=np.linalg.norm(v,axis=1)<4; c=fit(F[ok],v[ok,j]); good=abs(F@c-v[:,j])<.01; co.append(fit(F[good],v[good,j]));print('A good',good.sum(),co[-1]) if __name__=='__main__' else None
co=np.array(co)
def inverse(x):
 return np.array([least_squares(lambda z:np.array([design(z[None],j)[0]@co[j] for j in range(4)])-r,r,gtol=1e-13).x for r in x])
z=inverse(x); rng=np.random.default_rng(642); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(n) for i in range(5)]]
def scores(F,Y,lam=0.):
 Y=np.asarray(Y).reshape(n,-1); out=[]
 for a,b in splits:
  f=F[a]; sc=np.sqrt(np.mean(f*f,0)).clip(1e-8); q=f/sc; U,S,V=np.linalg.svd(q,full_matrices=False); c=(V.T*(S/(S*S+lam+1e-20)))@(U.T@Y[a]); err=abs(F[b]/sc@c-Y[b]); out.append(err.mean())
 return np.round([out[0],np.median(out), max(out)],4)
def pursuit(A,D,Y,k,keep=(0,)):
 ix=list(keep); norms=np.linalg.norm(D,axis=0)
 for _ in range(k):
  F=np.c_[A,D[:,ix]] if ix else A; R=Y-F@fit(F,Y); V=D-F@fit(F,D); score=np.linalg.norm(V.T@R,axis=1)/np.maximum(np.linalg.norm(V,axis=0),1e-8); score[ix]=-1; ix.append(int(np.argmax(score)))
 return ix
if __name__=='__main__':
 for d in (1,2,3,4):
  F=poly(z,d); print('z poly',d,*[(l,scores(F,y,l),scores(F,T,l)) for l in (.001,.05,1)])
 labels=[str(p) for k in (2,3) for p in it.combinations_with_replacement(range(4),k)]
 for pp in ('panel_b','panel_c'):
  u=s[pp]['u']
  for key,v in s[pp].items():
   if key=='u':continue
   print(pp,key)
   D=poly(u,3,2); A=aff(u)
   for k in (4,8,12):
    ix=pursuit(A,D,v,k,()); F=np.c_[A,D[:,ix]]; print(k,[labels[i] for i in ix],np.mean(abs(F@fit(F,v)-v)))
   for j in range(v.shape[1]):
    ix=pursuit(A,D,v[:,j:j+1],7,()); F=np.c_[A,D[:,ix]]; print(j,[labels[i] for i in ix],np.mean(abs(F@fit(F,v[:,j])-v[:,j])))
