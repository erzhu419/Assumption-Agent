import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; good=np.max(np.abs(y),axis=1)<8; x,y=x[good],y[good]; b=c['single_bank_sweeps']; ids,t,z=[b[k] for k in ('bank_id','ref_signal','raw_readout')]
for k in range(4):
 a=np.flatnonzero((np.count_nonzero(x,axis=1)==1)&(x[:,k]!=0)); m=(ids==k)&(abs(z)<8); v=np.r_[x[a,k],t[m]]; r=np.r_[y[a,k],z[m]]; print('\nbank',k)
 for name,func,p0 in [('tanh+linear',lambda v,p:p[0]+p[1]*np.tanh(p[2]*v)+p[3]*v,[0,1,1,.15]),('arctan+linear',lambda v,p:p[0]+p[1]*np.arctan(p[2]*v)+p[3]*v,[0,1,1,.15]),('shifted tanh+linear',lambda v,p:p[0]+p[1]*np.tanh(p[2]*v+p[4])+p[3]*v,[0,1,1,.15,0])]:
  p=least_squares(lambda p:func(v,p)-r,p0,gtol=1e-14).x; print(name,p,'max',max(abs(func(v,p)-r)))
  if name=='tanh+linear': selfp=p; selff=func
 q=x[:,(k+2)%4]; e=y[:,k]-selff(x[:,k],selfp);
 for deg in (1,2,3,4,5,7):
  F=np.column_stack([q**j for j in range(1,deg+1)]); use=np.arange(len(e))%4!=0; p=np.linalg.lstsq(F[use],e[use],rcond=None)[0]; print('cross',deg,'train',np.sqrt(np.mean((F[use]@p-e[use])**2)),'hold',max(abs(F[~use]@p-e[~use])),p.round(6))
 print('single-axis residuals',[(i,np.c_[x[np.count_nonzero(x,axis=1)==1,i],e[np.count_nonzero(x,axis=1)==1]][x[np.count_nonzero(x,axis=1)==1,i]!=0].round(6).tolist()) for i in range(4)])
