import pickle, numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; q=c['single_bank_sweeps']
np.set_printoptions(precision=5,suppress=True,linewidth=140)
clean=np.max(abs(y),axis=1)<9
print('excluded rows',np.where(~clean)[0], 'retained max',abs(y[clean]).max());print('axis rows x,y:',np.c_[x,y][:16]);
for d in (1,2,3,4):
 terms=[()] + [t for j in range(1,d+1) for t in combinations_with_replacement(range(4),j)]
 A=np.stack([np.prod(x[:,t],axis=1) for t in terms],axis=1); ids=np.flatnonzero(clean); er=[]
 for j in range(5):
  train=ids[np.arange(len(ids))%5!=j]; test=ids[np.arange(len(ids))%5==j]; b=np.linalg.lstsq(A[train],y[train],rcond=None)[0]; er.append(np.sqrt(np.mean((A[test]@b-y[test])**2,axis=0)))
 b=np.linalg.lstsq(A[ids],y[ids],rcond=None)[0]
 print('degree',d,'rank',np.linalg.matrix_rank(A[ids]), 'fit RMSE',np.mean((A[ids]@b-y[ids])**2,axis=0)**.5,'worst CV',np.max(er,axis=0))
 if d in (2,3):
  for t,row in zip(terms,b):
   if max(abs(row))>.012:print(t,row)
print('single bank data:');print(np.c_[q['bank_id'],q['ref_signal'],q['raw_readout']]); print('max raw by row:',abs(y).max(1).round(2))
