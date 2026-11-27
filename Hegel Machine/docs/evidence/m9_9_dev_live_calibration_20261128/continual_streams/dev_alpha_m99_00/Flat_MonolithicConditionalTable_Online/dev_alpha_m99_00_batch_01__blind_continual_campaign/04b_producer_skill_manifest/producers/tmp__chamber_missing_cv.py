import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/chamber_probe.py')
globals().update({k:p[k] for k in ('m','tr','te','di','t','tau','theta','bridge','v','vr')})
x=m._features(di); mask=np.isnan(di['coupon_features_raw']); lab=di['lab_id']; _,hold,_=m._coupon(di,x,bridge)
T=np.r_[t,1.]; E=lambda a:np.r_[-np.expm1(-t/a),1/a]; Y=np.c_[v,vr]-hold[:,None]*sum(w*E(a) for w,a in zip(theta[:2],tau)); W=np.r_[np.ones(len(t)),1.4]
print('Diagnostics missing-column patterns:',mask[mask.any(1)].astype(int).tolist())
for split in ((di,),(tr,te)):
 xx=m._features(*split); mm=np.concatenate([np.isnan(s['coupon_features_raw']) for s in split]); print('Imputation by column:',[round(xx[mm[:,j],j][0],6) if mm[:,j].any() else None for j in range(4)])
for mode in ('base','zero','batch','indicator'):
 xx=x.copy()
 if mode=='zero': xx[mask]=0
 if mode=='batch':
  xb=m._features(tr,te); mb=np.concatenate([np.isnan(s['coupon_features_raw']) for s in (tr,te)])
  for j in range(4):
   if mb[:,j].any(): xx[mask[:,j],j]=xb[mb[:,j],j][0]
 def fit(rows):
  def fun(q,ret=False):
   f=theta[2]+xx@q[2:6]; L=np.stack([E(np.exp(q[0])),E(np.exp(q[1])),T]); D=[f[:,None]*b for b in L]+[np.broadcast_to(T,Y.shape)]
   if mode=='indicator': D.append(mask.any(1)[:,None]*(.060623*E(.322898)+.962508*E(1.496942)+.16156*T))
   A=np.stack(D,axis=-1); a=(A[rows]*W[:,None]).reshape(-1,len(D)); coef=np.linalg.lstsq(a,(Y[rows]*W).ravel(),rcond=None)[0]; r=A@coef-Y
   return (r,coef) if ret else (r[rows]*W).ravel()
  q=least_squares(fun,np.r_[np.log([.32,1.5]),theta[3:]],max_nfev=70).x
  return (*fun(q,True),q)
 err=np.empty_like(Y)
 for l in np.unique(lab):
  k=lab==l; err[k]=fit(~k)[0][k]
 def scores(e): return np.sqrt(np.mean(e**2,axis=0))[[0,-2,-1]]
 r,c,q=fit(np.ones(len(x),bool)); print(mode,'worst lab',np.max([scores(err[lab==l]) for l in np.unique(lab)],axis=0).round(6),'missing/complete',scores(err[mask.any(1)]).round(6),scores(err[~mask.any(1)]).round(6),'coeff',c.round(6))
