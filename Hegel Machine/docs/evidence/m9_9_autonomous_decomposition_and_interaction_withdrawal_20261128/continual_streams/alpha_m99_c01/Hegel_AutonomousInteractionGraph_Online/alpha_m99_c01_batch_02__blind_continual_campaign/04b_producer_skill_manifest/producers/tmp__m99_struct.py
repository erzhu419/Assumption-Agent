from m99_probe import *
from m99_probe import _features,_design
from scipy.optimize import least_squares
from itertools import product
xg,_=_features(dg,dg)
a,b=_design(tau,xg,pd,hd,dt,t)
g=b[:,:2]*tau
D=beta[2]+xg@beta[3:]
V=np.c_[g,D]; O=np.c_[Y,R]; lab=dg['lab_id']; good=~np.isnan(dg['coupon_features_raw']).any(1)
base=np.r_[beta[:2]/tau,1.]
print('rate baseline all/complete',*[np.sqrt(np.mean((V[m]@base-R[m])**2)) for m in [np.ones(len(R),bool),good]])
print('missing residual', (V@base-R)[~good]); print('t,dt',t,dt)
def fit(ix, fixed=None, constrained=False):
 def prof(ts, ret=False):
  K=np.r_[np.c_[-np.expm1(-t[:,None]/ts),t],np.r_[1/ts,1][None,:]]
  if constrained:
   W=K[:,:2]-K[:,2:]/ts; off=V@base; M=(V[:,None,:,None]*W[None,:,None,:]).reshape(len(V),len(t)+1,6); y=O-off[:,None]*K[:,2];
  else:
   M=(V[:,None,:,None]*K[None,:,None,:]).reshape(len(V),len(t)+1,9);y=O
  c=np.linalg.lstsq(M[ix].reshape(-1,M.shape[-1]),y[ix].ravel(),rcond=None)[0]; E=M@c-y
  if ret:
   c=c.reshape(3,-1)
   if constrained:c=np.c_[c,base-c@(1/ts)]
   return ts,c,E
  return E[ix].ravel()
 if fixed is None:fixed=np.exp(least_squares(lambda z:prof(np.exp(z)),np.log(tau*[.9,.25]),max_nfev=100).x)
 return prof(fixed,True)
def rms(E):return np.array([np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2))])
for complete,constrained,clock in product([False,True],[False,True],[False,True]):
 fs=good if complete else np.ones(len(lab),bool); fixed=tau*[.9,.25] if clock else None
 ts,C,E=fit(fs,fixed,constrained);print('complete/constrained/fixed',complete,constrained,clock,'tau',ts,'fit',rms(E[fs]),'missing',rms(E[~good])); print('C',C.round(5))
 es=np.zeros_like(O)
 for l in np.unique(lab):
  _,_,ei=fit(fs&(lab!=l),fixed,constrained); es[lab==l]=ei[lab==l]
 print('CV worst',np.max([rms(es[fs&(lab==l)]) for l in np.unique(lab)],axis=0),'all',rms(es))
