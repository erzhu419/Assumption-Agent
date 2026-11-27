import itertools as it
import pickle
import numpy as np

s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']
x,y = (s[k] for k in ('ref_rosette','raw_rosette'))
terms = [()] + [t for d in (1,2,3,4) for t in it.combinations_with_replacement(range(4),d)]
D = np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
# Public spike rows are separated by a wide gap, so the diagnostic can
# compare model error without counting the EMI offsets as prediction targets.
a = np.linalg.norm(y,axis=1)
print('largest row norms', sorted(a)[-14:]); clean = a < 8
print('spike rows',np.flatnonzero(~clean))
axis=(x!=0).sum(1)==1
models = {
'affine': lambda t:len(t)<=1,
'diagonal cubic': lambda t:len(t)<=1 or len(t)==3 and len(set(t))==1,
'quadratic plus diagonal cubic': lambda t:len(t)<=2 or len(t)==3 and len(set(t))==1,
'full cubic': lambda t:len(t)<=3,
}
for name,allow in models.items():
 f=np.array([allow(t) for t in terms]); d=D[:,f]; p=np.flatnonzero(clean); err=[]
 for j in p:
  tr=p[p!=j]; err.append(d[j]@np.linalg.lstsq(d[tr],y[tr],rcond=None)[0]-y[j])
 e=np.array(err); print(name,'LOO rms',np.sqrt(np.mean(e**2,axis=0)),'worst axis/mixed',np.max(abs(e[axis[p]])),np.max(abs(e[~axis[p]])))
 b=np.linalg.lstsq(d[p],y[p],rcond=None)[0]; print('in sample max',np.max(np.abs(d[p]@b-y[p])))
 if name=='full cubic':
  for t,row in zip(it.compress(terms,f),b):
   if np.max(abs(row))>0.0001: print(t,np.round(row,6))
