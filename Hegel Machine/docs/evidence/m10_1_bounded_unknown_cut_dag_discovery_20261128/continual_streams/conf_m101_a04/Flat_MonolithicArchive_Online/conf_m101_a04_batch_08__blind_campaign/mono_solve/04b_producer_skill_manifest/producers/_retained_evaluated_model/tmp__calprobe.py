import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];p=s['panel_a'];u,v=p['u'],p['v'];n=len(u); rng=np.random.default_rng(0)
def robust(f,v):
 d=f.shape[1];best=np.inf
 for _ in range(2200):
  ix=rng.choice(n,min(d+2,n),replace=False);w=np.linalg.lstsq(f[ix],v[ix],rcond=None)[0];e=np.linalg.norm(f@w-v,axis=1); val=np.sort(e)[37]
  if val<best:best=val;keep=np.argsort(e)[:39]
 for _ in range(8):
  w=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0];e=np.linalg.norm(f@w-v,axis=1); keep=np.where(e<max(5*np.median(e),.0006))[0]
 return w,e
np.set_printoptions(precision=4,suppress=True)
for name,f in [('aff',u),('sep2',np.c_[u,u**2]),('sep3',np.c_[u,u**2,u**3]),('poly2',np.column_stack([np.prod(u[:,ix],axis=1) for j in (1,2) for ix in itertools.combinations_with_replacement(range(4),j)])),('poly2sep3',np.column_stack([np.prod(u[:,ix],axis=1) for j in (1,2) for ix in itertools.combinations_with_replacement(range(4),j)]+[a**3 for a in u.T])),('cyclic',np.c_[u,u**2,u**3,u*np.roll(u,1,axis=1),u*np.roll(u,2,axis=1)[:,:2].mean(1,keepdims=True)])]:
 w,e=robust(np.c_[np.ones(n),f],v);print(name,np.quantile(e,[0,.25,.5,.75,.85,.9,.95,1]),np.sort(e)[39:]);print(w)
for k in ('v1','v2'):
 p=s['panel_c'];u=p['u'];f=np.c_[np.ones(len(u)),u,np.column_stack([np.prod(u[:,ix],axis=1) for ix in itertools.combinations_with_replacement(range(4),2)])] if k=='v1' else np.c_[np.ones(len(u)),np.cos(2.4*u)];w=np.linalg.lstsq(f,p[k],rcond=None)[0];print(k,w)
