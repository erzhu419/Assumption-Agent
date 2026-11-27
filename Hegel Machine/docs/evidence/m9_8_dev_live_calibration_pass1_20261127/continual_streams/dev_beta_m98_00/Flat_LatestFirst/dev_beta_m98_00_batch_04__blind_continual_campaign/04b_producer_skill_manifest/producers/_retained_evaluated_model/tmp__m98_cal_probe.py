import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; np.set_printoptions(precision=5,suppress=True)
for j in range(4):
 print('bank',j,'extreme idx',np.flatnonzero(abs(y[:,j]-x[:,j])>4))
 good=abs(y[:,j]-x[:,j])<4
 for d in range(1,5):
  inds=[p for z in range(d+1) for p in combinations_with_replacement(range(4),z)]; A=np.array([np.prod(x[:,p],axis=1) for p in inds]).T
  e=[]
  for k in range(4):
   t=good&(np.arange(len(x))%4!=k); v=good&~t; coef=np.linalg.lstsq(A[t],y[t,j],rcond=1e-10)[0]; e.append(np.sqrt(np.mean((A[v]@coef-y[v,j])**2)))
  coef=np.linalg.lstsq(A[good],y[good,j],rcond=1e-10)[0]; r=y[:,j]-A@coef
  print(d,'cv',np.round(e,6),'train',np.linalg.norm(r[good])/np.sqrt(good.sum()),'base coefs',coef[:5], 'largest nonlinear', sorted(zip(inds[5:],coef[5:]),key=lambda p:abs(p[1]),reverse=True)[:4])
print('sweep axis inputs, raw outputs'); print(np.c_[x[:20],y[:20]])
