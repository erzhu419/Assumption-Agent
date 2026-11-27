import numpy as np
from scipy.optimize import least_squares
D=dict(np.load('/tmp/chamber_probe.npz')); globals().update(D)
print('times',t,'past_dt',dt,'taus',tau)
Y=y-m-drift[:,None]*t; R=r-mr-drift
print('discrepancy baseline RMSE curves/rates',np.sqrt(np.mean(Y**2)), np.sqrt(np.mean(R**2)))
print('correction singular values',np.linalg.svd(np.c_[Y,R],compute_uv=False))
print('diagnostic per-row [lab,hold,past,drift,rate residual]'); print(np.round(np.c_[labs,hold,past,drift,R],3))
A=lambda ts: np.column_stack([np.r_[np.ones(1)/s,-np.expm1(-t/s)] for s in ts]+[np.r_[1,t]])
def vp(z, observations):
 a=A(np.exp(z)); b=np.linalg.lstsq(a,observations.T,rcond=None)[0]; return (a@b-observations.T).ravel()
for name,obs in [('correction',np.c_[R,Y]),('full minus drift',np.c_[r-drift,y-drift[:,None]*t])]:
 for start in [tau,[.2,1,4],[.14,.4,1.8,5]]:
  fit=least_squares(vp,np.log(start),args=(obs,),bounds=(np.log(.035),np.log(35)),max_nfev=450)
  ts=np.exp(fit.x); a=A(ts); b=np.linalg.lstsq(a,obs.T,rcond=None)[0].T
  print(name,'basis',np.round(ts,5),'RMSE',np.sqrt(np.mean(fit.fun**2)))
  if len(start)==2:
   age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; mem=lambda ts:past@(-np.expm1(-dt[:,None]/ts)*np.exp(-age[:,None]/ts))
   gl=hold[:,None]-mem(tau); gn=hold[:,None]-mem(ts)
   f=np.c_[hold,past,gl,gn,drift,x]; names=['hold',*[f'past{i}' for i in range(past.shape[1])],'gap0','gap1','newgap0','newgap1','drift',*[f'x{i}' for i in range(x.shape[1])]]
   print('coefficients',np.round(b,3)); print('correlations'); print(list(zip(names,np.round(np.corrcoef(np.c_[f,b],rowvar=False)[:-3,-3:],3).tolist())))
   lin=np.c_[np.ones(len(hold)),hold,past,drift]; err=b-lin@np.linalg.lstsq(lin,b,rcond=None)[0]; print('linear stress+drift residual std',err.std(axis=0),'vs',b.std(axis=0))
