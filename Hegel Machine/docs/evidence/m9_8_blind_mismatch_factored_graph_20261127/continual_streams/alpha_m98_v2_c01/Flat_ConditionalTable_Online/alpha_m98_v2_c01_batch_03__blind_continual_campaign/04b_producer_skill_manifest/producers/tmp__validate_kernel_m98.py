import runpy,numpy as np
from scipy.optimize import least_squares
v=runpy.run_path('/tmp/probe_m98.py'); globals().update(v)
complete=np.isfinite(g['coupon_features_raw']).all(1)
print('complete coupon count',complete.sum())
models={'unit exponential':(lambda z:-np.expm1(-t),[]),'fit exponential':(lambda z:-np.expm1(-z[0]*t)/z[0],[1]),'fit tanh':(lambda z:np.tanh(z[0]*t)/z[0],[1]),'fit rational':(lambda z:t/(1+z[0]*t),[1]),'linear':(lambda z:t,[])}
for name,(kernel,init) in models.items():
 for by in ('lab_id','stratum'):
  e=np.empty_like(dy); pars=[]
  for group in np.unique(g[by]):
   test=g[by]==group; fit=~test & complete
   z=least_squares(lambda z:(dy[fit]-drift[fit,None]*kernel(z)).ravel(),init,bounds=(.01,10),loss='soft_l1',f_scale=.012).x if init else []
   pars.append(np.asarray(z).round(4).tolist()); e[test]=dy[test]-drift[test,None]*kernel(z)
  print(name,by,'parameters',pars,'pooled median/MSE-root',np.median(abs(e)),np.sqrt(np.mean(e**2)),'worst group RMSE',max(np.sqrt(np.mean(e[g[by]==i]**2)) for i in np.unique(g[by])),'worst complete coupon RMSE',max(np.sqrt(np.mean(e[complete]**2,1))),'worst imputed coupon RMSE',max(np.sqrt(np.mean(e[~complete]**2,1))))
print('Observed initial-rate residual, complete/imputed',[(np.sqrt(np.mean((dr[rows]-drift[rows])**2)),np.max(abs(dr[rows]-drift[rows]))) for rows in [complete,~complete]])
# Identify the dominant residual independent of the diagnostic feature normalization.
e=dy-dr[:,None]*(-np.expm1(-t)); print('Rate-anchored strain residual RMSE and worst coupon',np.sqrt(np.mean(e**2)),max(np.sqrt(np.mean(e**2,1))))
