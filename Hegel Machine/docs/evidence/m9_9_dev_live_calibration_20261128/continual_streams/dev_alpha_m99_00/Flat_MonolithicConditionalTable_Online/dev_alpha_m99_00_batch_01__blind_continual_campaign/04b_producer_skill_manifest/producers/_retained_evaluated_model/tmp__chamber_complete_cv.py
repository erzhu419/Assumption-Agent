import contextlib, io, runpy
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/chamber_probe.py')
globals().update({k:p[k] for k in ('m','di','t','tau','theta','bridge','v','vr')})
x=m._features(di); _,hold,_=m._coupon(di,x,bridge); lab=di['lab_id']; complete=~np.isnan(di['coupon_features_raw']).any(axis=1)
T=np.r_[t,1.]; E=lambda a:np.r_[-np.expm1(-t/a),1/a]; H=sum(w*E(a) for w,a in zip(theta[:2],tau)); Y=np.c_[v,vr]-hold[:,None]*H; W=np.r_[np.ones(len(t)),1.4]
print('Intrinsic taus and weights:',tau,theta)
for mode in ('current_family','two_complete','one_complete','rational_complete','intrinsic_complete'):
 def fit(rows):
  init={'current_family':[.32,1.50],'two_complete':[.32,1.50],'one_complete':[1.1],'rational_complete':[1.9],'intrinsic_complete':[]}[mode]
  k=len(init)
  def fun(q, ret=False):
   a=np.exp(q[:k]); B=([E(b) for b in a] if mode!='rational_complete' else [np.r_[t/(a[0]+t),1/a[0]]]) if k else [E(b) for b in tau]; B.append(T)
   f=theta[2]+x@q[k:]; A=np.stack([f[:,None]*b for b in B]+[np.broadcast_to(T,Y.shape)],axis=-1)
   coef=np.linalg.lstsq((A[rows]*W[:,None]).reshape(-1,A.shape[-1]),(Y[rows]*W).ravel(),rcond=None)[0]; r=A@coef-Y
   return (r,coef,q[k:]) if ret else (r[rows]*W).ravel()
  q=least_squares(fun,np.r_[np.log(init),theta[3:]],bounds=(np.r_[np.full(k,np.log(.05)),[-np.inf]*4],np.r_[np.full(k,np.log(25)),[np.inf]*4]),max_nfev=80).x
  return (*fun(q,True),np.exp(q[:k]))
 eligible=np.ones(len(x),bool) if mode=='current_family' else complete
 errors=[]
 for l in np.unique(lab):
  r,*_=fit((lab!=l)&eligible); q=r[(lab==l)&complete]; errors.append([np.sqrt(np.mean(q[:,:-1]**2)),np.sqrt(np.mean(q[:,-2]**2)),np.sqrt(np.mean(q[:,-1]**2))])
 r,c,b,a=fit(eligible); print(mode,'worst complete-coupon lab strain/final/rate RMS',np.max(errors,axis=0).round(6),'poles, coeff, direction',a.round(6),c.round(6),b.round(6),'missing residual final/rate means',r[~complete,-2:].mean(0).round(6))
