import numpy as np,pickle
from scipy.special import erf,expit
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def check(name,d,v):
 h=d@np.linalg.pinv(d);r=v-h@v; loo=r/np.maximum(1e-9,1-h.diagonal())[:,None]; print(name,d.shape[1],np.round([np.sqrt(np.mean(r*r)),np.sqrt(np.mean(loo*loo)),np.max(abs(r))],6))
def ops(key,field):
 u=s[key]['u'];v=s[key][field]; n=len(u); one=np.ones(n)
 for k in range(1,6):check('power'+str(k),np.c_[one,*[u**j for j in range(1,k+1)]],v)
 def feat(u,p,typ):
  k,a,b,c=p
  z={'erf':lambda:erf(np.exp(k)*u+a*u**3)+b*u**3+c*u,'tanh':lambda:np.tanh(np.exp(k)*u+a*u**3)+b*u**3+c*u,'rational':lambda:u*abs(u)/(1+np.exp(k)*abs(u))+a*u**3+b*u,'sin':lambda:np.sin(np.exp(k)*u+a*u**3)+b*u**3+c*u,'rat2':lambda:u/(1+np.exp(k)*u**2)+a*u**3+b*u}[typ]()
  return np.c_[np.ones(len(u)),z]
 for typ in ['erf','tanh','rational','sin','rat2']:
  def fun(p):
   d=feat(u,p,typ);return (d@np.linalg.lstsq(d,v,rcond=None)[0]-v).ravel()
  rng=np.random.default_rng(2); best=None
  for p in [[.63,4,-.62,-2.13],[-.42,-.02,-1.08,0],[0,0,0,0],*rng.uniform(-2,2,(6,4))]:
   sol=least_squares(fun,p,max_nfev=200,bounds=(-10,10));
   if best is None or sol.cost<best.cost:best=sol
  print(typ,np.round(best.x,5),end=' '); check('',feat(u,best.x,typ),v)
for k,f in [('panel_b','v'),('panel_c','v2')]:print(k,f);ops(k,f)
print('A');u=s['panel_a']['u'];v=s['panel_a']['v'];keep=np.max(abs(v-u),axis=1)<3; print('outliers',np.where(~keep)[0]);u=u[keep];v=v[keep];one=np.ones(len(u)); neg=np.minimum(u,0)
for name,d in [('affine',np.c_[one,u]),('poly2',np.c_[one,u,*[np.prod(u[:,j],axis=1) for j in combinations_with_replacement(range(4),2)]]),('coord3',np.c_[one,u,u**2,u**3]),('hinges',np.c_[one,u,neg,neg**2,neg**3]),('hingemix',np.c_[one,u,neg,*[np.prod(neg[:,j],axis=1) for j in combinations_with_replacement(range(4),2)]]),('mixed cubics',np.c_[one,u,*[np.prod(u[:,j],axis=1) for deg in (2,3) for j in combinations_with_replacement(range(4),deg)]])]:check(name,d,v)
np.set_printoptions(suppress=True,precision=3); print('u, residual v-u');print(np.c_[u,v-u]);print('linear coef',np.linalg.lstsq(np.c_[one,u],v,rcond=None)[0]);
for p in ['panel_c']:
 u=s[p]['u']; v=s[p]['v1']; pairs=list(combinations_with_replacement(range(4),2)); d=np.c_[np.ones(len(u)),u,*[np.prod(u[:,j],axis=1) for j in pairs]];print('C1',pairs,np.linalg.lstsq(d,v,rcond=None)[0]);
