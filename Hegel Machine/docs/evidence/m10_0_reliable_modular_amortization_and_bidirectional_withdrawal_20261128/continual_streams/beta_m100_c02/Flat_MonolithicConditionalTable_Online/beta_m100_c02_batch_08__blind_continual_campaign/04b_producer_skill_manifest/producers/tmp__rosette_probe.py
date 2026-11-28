import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; cal=s['sensor_calibration_sweeps']; x,y=cal['ref_rosette'],cal['raw_rosette']; one=cal['single_bank_sweeps']; ids=one['bank_id']; sig=one['ref_signal']; val=one['raw_readout'];
print('rows nonzero',np.count_nonzero(x,axis=1)); print('gross rows',np.where(np.max(abs(y),axis=1)>5)[0]); print('single-axis rows'); print(np.c_[x[:16],y[:16]].round(5))
funcs={'tanh':np.tanh,'atan':np.arctan,'softsign':lambda z: z/np.sqrt(1+z*z),'erf':__import__('scipy').special.erf}
for i in range(4):
 print('bank',i)
 for j in range(4):
  u=(np.count_nonzero(x,axis=1)==1)&(x[:,j]!=0)&(abs(y[:,i])<5); t=x[u,j]; z=y[u,i];
  if i==j:
   q=(ids==i)&(abs(val)<5); t=np.r_[t,sig[q]]; z=np.r_[z,val[q]]
  if i!=j: print(j,np.c_[t,z].round(5).tolist()); continue
  for name,f in funcs.items():
   for linear in [False,True]:
    a=least_squares(lambda p:p[0]+p[1]*f(p[2]*t)+(p[3]*t if linear else 0)-z,[0,1,1,0] if linear else [0,1,1],max_nfev=1000)
    print(j,name,linear,np.sqrt(np.mean(a.fun*a.fun)),a.x.round(6))
