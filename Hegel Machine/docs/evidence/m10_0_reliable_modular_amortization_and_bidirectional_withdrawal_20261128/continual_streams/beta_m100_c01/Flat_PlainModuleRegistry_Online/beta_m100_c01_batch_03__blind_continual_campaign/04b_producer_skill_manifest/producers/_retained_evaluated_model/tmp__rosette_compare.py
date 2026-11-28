import itertools,pickle
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Y=c['raw_rosette']; b=c['single_bank_sweeps']; B=np.eye(4)[b['bank_id']]*b['ref_signal'][:,None]; R=b['raw_readout']; rows=np.arange(len(X)); axis=np.count_nonzero(X,axis=1)<=1
np.set_printoptions(precision=6,suppress=True,linewidth=150)
print('axis rows:',np.where(axis)[0]); print('axis references and raw:',np.c_[X[axis],Y[axis]])
for j in range(4):
 k=b['bank_id']==j; xx=b['ref_signal'][k]; yy=R[k]; cc=least_squares(lambda p: np.polynomial.polynomial.polyval(xx,p)-yy,[0,1,0,.2],loss='cauchy',f_scale=.03).x; print('single',j,cc,'resid',np.round(np.polynomial.polynomial.polyval(xx,cc)-yy,5))
def terms(d): return list(itertools.combinations_with_replacement(range(4),d))
def mat(x,t): return np.c_[np.ones(len(x)),x,*[np.prod(x[:,u],axis=1) for u in t]]
models={'affine':[], 'component_cubic':[(j,j,j) for j in range(4)], 'all_cubic':terms(3),'full_d3':terms(2)+terms(3),'mixed_cubic':None,'mixed_shift_cubic':None,'mixed_rational':None}
def fit(x,y,name):
 a=mat(x,[]); C=np.linalg.lstsq(a,y,rcond=None)[0]; e=np.linalg.norm(a@C-y,axis=1)
 for _ in range(8):
  w=1/np.sqrt(1+(e/.6)**2); C=np.linalg.lstsq(a*w[:,None],y*w[:,None],rcond=None)[0]; e=np.linalg.norm(a@C-y,axis=1)
 keep=e<3.5; x=x[keep]; y=y[keep]; a=a[keep]
 if models[name] is not None:
  a=mat(x,models[name]); C=np.linalg.lstsq(a,y,rcond=None)[0]; return (lambda z:mat(z,models[name])@C),C
 def fun(z,p):
  u=z@p[1:5]; q=p[5]; off=p[0];
  if name=='mixed_cubic': return off+u+q*u**3
  if name=='mixed_shift_cubic': return off+u+p[6]*u**2+q*u**3
  return off+u+q*u**3/(1+p[6]**2*u**2)
 P=np.column_stack([least_squares(lambda p:fun(x,p)-y[:,j],np.r_[C[:,j],.1,*([.1] if name!='mixed_cubic' else [])],loss='soft_l1',f_scale=.04).x for j in range(4)])
 return (lambda z:fun(z,P)),P
rng=np.random.default_rng(861); folds=np.zeros(len(X),int)
for inds in [rows[axis],rows[~axis]]: folds[rng.permutation(inds)]=np.arange(len(inds))%5
for name in models:
 pred=np.zeros_like(Y); inv=[]
 for f in range(5):
  tr=folds!=f; fn,P=fit(X[tr],Y[tr],name); pred[~tr]=fn(X[~tr]);
  for j in rows[~tr]:
   if np.linalg.norm(Y[j])>8 or np.max(abs(X[j]))>.7: continue
   z=least_squares(lambda q:fn(q[None])[0]-Y[j],np.zeros(4)).x; inv.append(np.linalg.norm(z-X[j]))
 e=np.linalg.norm(pred-Y,axis=1); good=np.linalg.norm(Y,axis=1)<8; clean_bank=abs(R)<8; fn,P=fit(X,Y,name); bank_err=fn(B)[np.arange(len(B)),b['bank_id']]-R
 print(name,'OOF mixed/axis RMS',[np.sqrt(np.mean(e[good & k]**2)) for k in [~axis,axis]],'near-origin inverse max',max(inv,default=np.nan),'bank RMS',np.sqrt(np.mean(bank_err[clean_bank]**2)))
 if name in ('mixed_cubic','mixed_shift_cubic','mixed_rational','all_cubic'): print(P)
