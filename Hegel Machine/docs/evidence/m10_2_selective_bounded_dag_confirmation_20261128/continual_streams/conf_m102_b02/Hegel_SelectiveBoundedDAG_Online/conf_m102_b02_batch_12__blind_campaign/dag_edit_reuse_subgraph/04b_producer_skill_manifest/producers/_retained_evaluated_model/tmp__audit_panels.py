from probe import *
from itertools import combinations_with_replacement
ua,va=A['u'],A['v'];print('A shape range',ua.shape,ua.min(),ua.max());print('A sorted affine row errors',np.sort(np.linalg.norm(d@w-va,axis=1)).round(5));print('A rows extreme',np.c_[ua,va,(d@w-va)][np.argsort(np.linalg.norm(d@w-va,axis=1))[-10:]].round(3))
for deg in [1,2,3,4,5]:
 for sep in [True,False]:
  idx=[c for k in range(1,deg+1) for c in combinations_with_replacement(range(4),k) if not sep or len(set(c))==1]; da=np.c_[np.ones(len(ua)),*[np.prod(ua[:,j],axis=1) for j in idx]]
  if da.shape[1]>=len(ua)-5:continue
  rng=np.random.default_rng(1); best=None
  for i in range(350):
   use=rng.choice(len(ua),da.shape[1]+2,replace=False); coeff=np.linalg.lstsq(da[use],va[use],rcond=None)[0];rr=np.linalg.norm(da@coeff-va,axis=1); med=np.median(rr); 
   if best is None or med<best[0]:best=med,rr
  rr=best[1]
  for _ in range(5):
   use=rr<max(.001,3*np.median(rr)); coeff=np.linalg.lstsq(da[use],va[use],rcond=None)[0];rr=np.linalg.norm(da@coeff-va,axis=1)
  print('A',deg,sep,use.sum(),np.quantile(rr,[0,.25,.5,.75,.9]).round(6));
  if deg<=2:print(coeff.round(4))
D=lambda q,*cols:np.column_stack((np.ones(len(q)),*cols))
for key,Q,V in [('B',ub,vb),('C',u,v)]:
 print(key,'shapes',Q.shape,V.shape)
 maps={ 'rat_even':lambda q,p:q*q/(1+np.exp(p[0])*q*q),'rat_odd':lambda q,p:q/(1+np.exp(p[0])*q*q),'rat_abs':lambda q,p:q*abs(q)/(1+np.exp(p[0])*abs(q)),'tanh':lambda q,p:np.tanh(np.exp(p[0])*q),'erf':lambda q,p:erf(np.exp(p[0])*q),'arcsinh':lambda q,p:np.arcsinh(np.exp(p[0])*q),'sin':lambda q,p:np.sin(np.exp(p[0])*q),'erf3core':lambda q,p:erf(np.exp(p[0])*q+p[1]*q**3),'signed_power':lambda q,p:q*abs(q)**np.exp(p[0])}
 for name,fn in maps.items():
  for powers in [[1,3],[1,2,3]]:
   try:
    e,p,cf=vp(Q,V,lambda q,p:D(q,fn(q,p),*[q**j for j in powers]),[[0,0],[.6,4],[-.4,0]] if name=='erf3core' else [[0],[-1],[1]]);print(key,name,powers,round(e,7),p.round(5))
   except Exception as ex: print(ex)
