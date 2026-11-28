import pickle, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; b=c['single_bank_sweeps']; a=b['ref_signal']; by=b['raw_readout']; ix=b['bank_id']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
print('ref range',x.min(0),x.max(0)); print('raw obs range',s['y_obs_raw'].min(),s['y_obs_raw'].max())
def feat(x,d):
 cols=[np.ones(len(x)),*x.T]
 if d>=2: cols += [x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]
 if d>=3: cols += [x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i,4) for k in range(j,4)]
 return np.column_stack(cols)
for k in range(4):
 use=ix==k; iso=np.zeros((use.sum(),4));iso[:,k]=a[use]; yy=np.r_[y[:,k],by[use]]; xx=np.vstack((x,iso)); print('BANK',k)
 for d in (1,2,3):
  z=feat(xx,d); fit=lambda m: least_squares(lambda t:(z@t-yy)[m],np.linalg.lstsq(z[m],yy[m],rcond=None)[0],jac=lambda t:z[m],loss='cauchy',f_scale=.018).x
  t=fit(np.ones(len(yy),bool)); e=yy-z@t; cv=np.empty_like(e)
  for j in range(5):
   m=np.arange(len(e))%5!=j; cv[~m]=yy[~m]-z[~m]@fit(m)
  print('degree',d,'coeff',t[:15], 'abs residual and CV q',np.quantile(abs(e),[.5,.8,.9,.95,1]),np.quantile(abs(cv),[.5,.8,.9,.95,1])); print('single residual',e[80:])
