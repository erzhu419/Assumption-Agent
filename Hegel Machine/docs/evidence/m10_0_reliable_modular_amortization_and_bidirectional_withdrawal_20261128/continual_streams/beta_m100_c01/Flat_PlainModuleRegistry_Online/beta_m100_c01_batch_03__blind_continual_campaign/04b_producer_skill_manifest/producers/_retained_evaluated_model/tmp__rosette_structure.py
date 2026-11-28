import itertools,pickle
import numpy as np
from collections import Counter
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; ids=np.arange(len(x))
t=[u for k in range(4) for u in itertools.combinations_with_replacement(range(4),k)]
def design(x): return np.column_stack([np.prod(x[:,u],axis=1) for u in t])
a=design(x); lin=a[:,:5]; w=np.ones(len(x))
for _ in range(10):
 p=np.linalg.lstsq(lin*w[:,None],y*w[:,None],rcond=None)[0]; e=np.linalg.norm(lin@p-y,axis=1); w=1/np.hypot(1,e/.6)
g=e<3.5
print('candidate EMI rows',ids[~g], 'largest retained affine residual',e[g].max(),'smallest rejected', e[~g].min())
p=np.linalg.lstsq(a[g],y[g],rcond=None)[0]; r=np.linalg.norm(a@p-y,axis=1)
print('residual extrema',r[g].max(),r[~g].min()); print('term coefficients:')
for u,b in zip(t,p): print(u,*['% .7f'%v for v in b])
print('sweep reference channel spans:', x.min(0),x.max(0)); print('calibration conditioning',np.linalg.cond(a[g])); print('midrange clean rows', sum(g&(abs(x).max(1)<.7)))
for j in range(4):
 ix=(c['single_bank_sweeps']['bank_id']==j); z=c['single_bank_sweeps']['ref_signal'][ix]; q=c['single_bank_sweeps']['raw_readout'][ix]; err=design(z[:,None]*np.eye(4)[j])@p[:,j]-q
 print('bank',j,'abs error sorted',np.round(np.sort(abs(err)),7))
