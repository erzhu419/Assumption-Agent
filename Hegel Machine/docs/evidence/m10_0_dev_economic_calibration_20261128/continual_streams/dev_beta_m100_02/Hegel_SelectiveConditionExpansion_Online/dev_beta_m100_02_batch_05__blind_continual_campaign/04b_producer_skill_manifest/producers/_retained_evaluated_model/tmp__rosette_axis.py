import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; b=c['single_bank_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; print('single bank:')
for i in range(4):
 m=b['bank_id']==i; u,v=b['ref_signal'][m],b['raw_readout'][m]; print(i,np.round(np.c_[u,v],5).tolist()); m=abs(v-np.median(v))<5; u,v=u[m],v[m]
 for name,f in [('tanh',np.tanh),('arctan',np.arctan),('soft',lambda z:z/np.sqrt(1+z*z))]:
  for linear in [False,True]:
   fun=lambda p:p[0]+p[1]*f(np.exp(p[2])*u)+(p[3]*u if linear else 0)-v
   q=least_squares(fun,[0,1,0]+([0] if linear else []),bounds=([-np.inf,0,-3]+([-np.inf] if linear else []),[np.inf,np.inf,3]+([np.inf] if linear else []))); print(name,linear,np.round(q.x,6),np.std(q.fun))
