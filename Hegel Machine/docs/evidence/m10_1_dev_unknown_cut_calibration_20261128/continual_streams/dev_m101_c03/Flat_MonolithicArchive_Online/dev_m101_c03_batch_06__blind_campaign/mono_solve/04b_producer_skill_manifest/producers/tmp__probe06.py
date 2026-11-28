import pickle,numpy as np,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=145)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def trim(X,Y):
 k=int(.85*len(X)); rng=np.random.default_rng(51);best=None
 for _ in range(350):
  ix=rng.choice(len(X),max(X.shape[1]+1,8),False);a=ls(X[ix],Y[ix]);
  for j in range(4):
   r=np.sum((X@a-Y)**2,axis=1) if Y.ndim==2 else (X@a-Y)**2;ix=np.argsort(r)[:k];a=ls(X[ix],Y[ix]);
  e=np.mean((X[ix]@a-Y[ix])**2)**.5
  if best is None or e<best[0]:best=(e,a,ix)
 return best
u,v=s['panel_a'].values()
for j in range(4):
 z=u[:,j];f=np.c_[z,z*z,z**3];e,a,ix=trim(aff(f),v[:,j]);print('diagonal a',j,e,a,'all res quantiles',np.quantile(abs(aff(f)@a-v[:,j]),[.1,.5,.8,.9,.95,1]))
for p in (1,2,3):
 f=aff(np.hstack([u**q for q in range(1,p+1)]));e,a,ix=trim(f,v);print('full a',p,e,a)
u=s['panel_c']['u'];prod=np.stack([u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)],1);v=s['panel_c']['v1']
for name,F in [('pair',prod),('quad',np.c_[u,u*u,prod]),('cubic',np.c_[u,u*u,u**3,prod]),('triples',np.c_[prod,np.stack([np.prod(u[:,ix],1) for ix in it.combinations(range(4),3)],1)])]:
 F=aff(F);print('v1',name,'rms',np.mean((F@ls(F,v)-v)**2)**.5,'coef',ls(F,v))
