import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; c=d['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']; n=len(x)
bad=np.max(abs(y),axis=1)>8
print('gross rows',np.flatnonzero(bad),'median shift',np.round(np.median(y[bad],0),3))
def poly(u,degree,diag=False):
 return np.column_stack([np.ones(len(u)),u]+[np.prod(u[:,i],axis=1) for k in range(2,degree+1) for i in combinations_with_replacement(range(4),k) if not diag or len(set(i))==1])
for mode in ('mix_then_sinh','mix_then_cubic','mix_then_sinh_shift','own_sinh','own_cubic','own_sinh_shift','forward_poly2','forward_poly3','inverse_poly2','inverse_poly3'):
 def fit(u,v):
  if 'poly' in mode:
   f=lambda q:poly(q,int(mode[-1])); b=np.linalg.lstsq(f(u),v,rcond=None)[0];return lambda q:f(q)@b
  affine=np.linalg.lstsq(np.c_[np.ones(len(u)),u],v,rcond=None)[0]
  if mode.startswith('mix'):
   def pred(p,q):
    w=q@p[4:20].reshape(4,4); o=p[:4]; s=p[20:24];
    if mode.endswith('cubic'): return o+w+s*w**3
    if mode.endswith('shift'):
     h=p[24:28]; return o+(np.sinh(w*s+h)-np.sinh(h))/s
    return o+np.sinh(w*s)/s
  else:
   def pred(p,q):
    w=q@p[4:20].reshape(4,4); o=p[:4]; s=p[20:24];
    if mode.endswith('cubic'): return o+w+s*q**3
    if mode.endswith('shift'):
     h=p[24:28]; return o+w+p[28:32]*(np.sinh(q*s+h)-np.sinh(h)-q*s*np.cosh(h))
    return o+w+p[24:28]*(np.sinh(q*s)-q*s)
  p=np.r_[affine.ravel(),[.5]*4];
  if mode.endswith('shift'):p=np.r_[p,[0.]*4]
  if mode.startswith('own') and not mode.endswith('cubic'):p=np.r_[p,[1.]*4]
  opt=least_squares(lambda p:(pred(p,u)-v).ravel(),p,max_nfev=350); return lambda q:pred(opt.x,q)
 u,v=(y,x) if mode.startswith('inverse') else (x,y); keep=np.flatnonzero(~bad); f=fit(u[keep],v[keep]); e=f(u)-v
 cv=[]
 for ids in np.array_split(np.random.default_rng(6).permutation(keep),5):
  tr=np.setdiff1d(keep,ids); cv.append(fit(u[tr],v[tr])(u[ids])-v[ids]);
 print(mode,'clean RMS',np.round(np.sqrt(np.mean(e[keep]**2,0)),6),'worst CV rms',round(max(np.sqrt(np.mean(z*z)) for z in cv),6),'gross offsets',np.round(e[bad],3).tolist() if mode in ('mix_then_sinh','own_cubic') else '')
