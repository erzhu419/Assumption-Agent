import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; xb=np.eye(4)[b['bank_id']]*b['ref_signal'][:,None]; yb=b['raw_readout']; ib=b['bank_id']; np.set_printoptions(precision=5,suppress=True)
for degree in [1,2,3,4]:
 for typ in ['separable','full']:
  ts=[()]+[t for d in range(1,degree+1) for t in combinations_with_replacement(range(4),d) if typ=='full' or len(set(t))==1]; X=np.column_stack([np.prod(x[:,t],axis=1) for t in ts]); B=np.column_stack([np.prod(xb[:,t],axis=1) for t in ts]); good=np.max(np.abs(y),axis=1)<8; bank=np.abs(yb)<8; T=np.flatnonzero(good); P=np.zeros_like(y); Pb=np.zeros_like(yb)
  for j,t in enumerate(np.array_split(np.random.default_rng(42).permutation(T),5)):
   tr=np.setdiff1d(T,t); bt=np.arange(len(yb))%5!=j
   for k in range(4):
    use=(ib==k)&bank&bt; C=np.linalg.lstsq(np.r_[X[tr],B[use]],np.r_[y[tr,k],yb[use]],rcond=None)[0]; P[t,k]=X[t]@C; test=(ib==k)&~bt; Pb[test]=B[test]@C
  print(typ,degree,len(ts),'rosette rms',np.sqrt(np.mean((P[good]-y[good])**2,axis=0)),'max',abs(P[good]-y[good]).max(),'bank',abs(Pb[bank]-yb[bank]).max());
  if degree==3:
   C=np.stack([np.linalg.lstsq(np.r_[X[good],B[(ib==k)&bank]],np.r_[y[good,k],yb[(ib==k)&bank]],rcond=None)[0] for k in range(4)],axis=1); print([(t,z.round(5).tolist()) for t,z in zip(ts,C) if abs(z).max()>.004])
print('bank outlier',np.where(~bank)[0], 'rosette outlier',np.where(~good)[0]);print('bank by ch',yb.reshape(4,12))
