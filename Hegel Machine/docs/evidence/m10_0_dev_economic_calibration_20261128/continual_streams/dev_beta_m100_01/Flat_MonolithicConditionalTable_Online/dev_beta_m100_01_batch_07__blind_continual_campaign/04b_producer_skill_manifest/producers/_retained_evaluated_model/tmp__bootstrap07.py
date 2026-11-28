from probe07 import m,s
import numpy as np
from scipy.optimize import minimize
from itertools import product
y=m._calibrate(s); Z=s['Z_nuisance']; modes=s['source_modes']; cap=s['u_box']; budget=s['total_budget']; nr=len(s['sensor_ij'])//4; nmu=len(s['mu_shifts']); groups=np.tile(np.repeat(np.arange(nr),4),nmu); shift=np.repeat(np.arange(nmu),4*nr)
for walls in product(['periodic','reflect'], repeat=2):
 S,_=m._forward(s,walls)
 print(walls)
 for lam in [0,.004,.04]:
  out=[]
  for te in [np.zeros(len(y),dtype=bool)]+[(groups==a)&(shift==b) for a in range(nr) for b in range(nmu)]:
   tr=~te; z=Z[tr]; t=S[tr]; v=y[tr]; p=np.linalg.lstsq(z,np.c_[t,v],rcond=None)[0]; A=t-z@p[:,:-1]; b=v-z@p[:,-1]; w,_=m._qp(A,b,lam,cap,budget); e=y-S@w-Z@(p[:,-1]-p[:,:-1]@w)
   if not te.any(): print('ridge',lam,'train RMSE',np.sqrt(np.mean(e*e)), 'w',w.round(4))
   else: out.append(np.mean(e[te]**2))
  print('block pooled/worst RMSE',np.sqrt([np.mean(out),max(out)]))
