import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']
def aff(x):return np.column_stack((np.ones(len(x)),x))
def fit(a,y,loo=False):
 p=np.linalg.pinv(a);w=p@y;r=a@w-y
 if loo:r=r/(1-np.sum(a*p.T,axis=1)[:,None])
 return r,w
p=s['panel_c'];u=p['u'];
funs={'atan':lambda x,q:np.arctan(q*x),'softsign2':lambda x,q:x/np.sqrt(1+q*x*x),'rat':lambda x,q:x/(1+q*x*x),'tanh':lambda x,q:np.tanh(q*x),'asinh':lambda x,q:np.arcsinh(q*x),'erf':lambda x,q:__import__('scipy.special',fromlist=['erf']).erf(q*x),'sin':lambda x,q:np.sin(q*x)}
def search(y,features,family):
 best=None
 for n in [1,4]:
  for v in [.3,1.,2.]:
   q=least_squares(lambda q:fit(features(family(u,q)),y,True)[0].ravel(),[v]*n,bounds=(.001,10),max_nfev=130).x
   r,w=fit(features(family(u,q)),y,True);e=np.sqrt(np.mean(r*r))
   if best is None or e<best[0]:best=e,q,w
 return best
best=(np.inf,None)
for k,f in funs.items():
 e,q,w=search(p['v1'],aff,f);print('C1',k,round(e,7),np.round(q,5))
 if e<best[0]:best=e,(k,q,w)
print('C1 winner',best)
for k,f in {**funs,'saturating_square':lambda x,q:x*x/(1+q*x*x),'cosh':lambda x,q:np.cosh(q*x),'log_square':lambda x,q:np.log1p(q*x*x),'sqrt_square':lambda x,q:np.sqrt(1+q*x*x),'tanh_square':lambda x,q:np.tanh(q*x*x)}.items():
 for sym in ([0,1] if k in funs else [0]):
  e,q,w=search(p['v2'],lambda b:np.c_[aff(u),b*b if sym else b],f);print('C2',k,sym,round(e,7),np.round(q,5))
print('C2 quad coefficients',fit(np.c_[aff(u),*[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]],p['v2'])[1].round(5))
t=s['train'];x=t['x'];y=t['y'];z=t['traj'];print('ranges',x.min(0),x.max(0));print('time-y residual correlations')
for d in (1,2,3):
 a=np.c_[aff(x),*[np.prod(x[:,ix],axis=1) for k in range(2,d+1) for ix in combinations_with_replacement(range(4),k)]]
 r,w=fit(a,np.c_[y,z],True);print(d,np.sqrt(np.mean(r*r,0)),np.corrcoef(r.T)[0,1:].round(3))
 if d==3:print('cubic coefs y, first and last t',[(ix,np.round(row[[0,1,-1]],5)) for ix,row in zip(combinations_with_replacement(range(4),3),w[15:])])
