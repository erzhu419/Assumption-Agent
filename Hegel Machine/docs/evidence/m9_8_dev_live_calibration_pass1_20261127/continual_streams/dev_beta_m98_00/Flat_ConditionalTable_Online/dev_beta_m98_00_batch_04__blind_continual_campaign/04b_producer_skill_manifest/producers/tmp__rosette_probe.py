import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; b=sw['single_bank_sweeps']; xb=np.eye(4)[b['bank_id']]*b['ref_signal'][:,None]
for j in range(4):
 for degree in (3,4):
  terms=[t for d in range(degree+1) for t in combinations_with_replacement(range(4),d)]
  def design(x): return np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
  sel=b['bank_id']==j; X=np.vstack((x,xb[sel])); Y=np.r_[y[:,j],b['raw_readout'][sel]]; A=design(X); good=abs(Y)<8
  c=least_squares(lambda c:(A@c-Y)[good],np.linalg.lstsq(A[good],Y[good],rcond=None)[0],jac=lambda c:A[good],loss='soft_l1',f_scale=.001).x
  r=A@c-Y; print('channel',j,'degree',degree,'RMSE',np.sqrt(np.mean(r[good]**2)),'max',np.max(abs(r[good])),'large',[(i,round(z,4)) for i,z in enumerate(r) if abs(z)>.03]); print([(t,round(z,6)) for t,z in zip(terms,c) if abs(z)>.002])
 print('single');print(np.round(np.column_stack((x[:16],y[:16,j])),6))
