import numpy as np,pickle,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rng=np.random.default_rng(33)
for key in ('panel_a','panel_b'):
 u,v=(s[key][k] for k in ('u','v')); N,d=u.shape; I=np.ones((N,1)); A=np.c_[I,u]; ix,jx=np.array(list(it.combinations_with_replacement(range(d),2))).T; P=u[:,ix]*u[:,jx]; print(key)
 def fit(B, robust=False):
  def fun(idx):return np.linalg.lstsq(B[idx],v[idx],rcond=None)[0]
  best=None
  for _ in range(600 if robust else 1):
   idx=rng.choice(N,min(N,B.shape[1]+3),False) if robust else np.arange(N); C=fun(idx); e=np.linalg.norm(B@C-v,axis=1); score=np.sort(e)[:int(.85*N)].mean();
   if best is None or score<best[0]:best=score,e,C
  E=best[1]; g=E<=max(.02,4*np.median(E)); C=fun(g); R=B@C-v; e=np.linalg.norm(R,axis=1); return np.quantile(e,[.5,.8,.9,1]),C,e
 for tag,B in [('aff',A),('quad',np.c_[A,P]),('cube',np.c_[A,u**3]),('quad+cube',np.c_[A,P,u**3]),('soft',np.c_[A,u/(1+.85*abs(u))]),('rat',np.c_[A,u/(1+.85*u*u)]),('tanh',np.c_[A,np.tanh(1.5*u)]),('sin',np.c_[A,np.sin(2.4*u)])]:
  err,C,e=fit(B,key=='panel_a'); print(tag,err,'coeff',np.round(C,3) if tag in ('aff','quad+cube') else '')
 print('ranges',u.min(0),u.max(0),v.min(0),v.max(0));print('aff resid sorted',np.round(np.sort(fit(A,key=='panel_a')[2]),3))
