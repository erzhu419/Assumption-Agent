import pickle
import numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; cal=s['sensor_calibration_sweeps']; x=cal['ref_rosette']; y=cal['raw_rosette']
np.set_printoptions(precision=5,suppress=True)
for k,v in cal['single_bank_sweeps'].items():
 a=np.asarray(v); print('single',k,a.shape,a.ravel()[:12])
print('banks',np.unique(s['row_bank_id'].reshape(-1,4),axis=0)); print('pulse key', 'reference_tracer_pulses' in s)
print('paired first rows\n',np.c_[x,y][:5]); print('ranges',x.min(0),x.max(0)); print('obs',s['y_obs_raw'].reshape(-1,4).min(0),s['y_obs_raw'].reshape(-1,4).max(0))

def fit(a,b):
 w=np.ones(len(a))
 for _ in range(25):
  c=np.linalg.lstsq(a*w[:,None]**.5,b*w[:,None]**.5,rcond=None)[0]
  r=np.linalg.norm(b-a@c,axis=1); m=max(np.median(r),1e-10)
  w=1/(1+(r/(2*m))**2)
 return c

def features(x,kind):
 terms=[np.ones(len(x)),*x.T];
 if kind in ('quad','cubic'): terms += [x[:,inds].prod(1) for inds in combinations_with_replacement(range(4),2)]
 if kind in ('pure3','pure5'): terms += [*(x**3).T]
 if kind=='pure5': terms += [*(x**5).T]
 if kind=='cubic': terms += [x[:,inds].prod(1) for inds in combinations_with_replacement(range(4),3)]
 if kind=='tanh': terms += [*np.tanh(x).T]
 if kind=='rat': terms += [*(x/(1+x*x)).T]
 return np.column_stack(terms)

if __name__=='__main__':
 keep=abs(y).max(1)<4
 print('spike IDs',np.flatnonzero(~keep)); block=np.arange(len(x))//10
 for kind in ('linear','quad','pure3','pure5','tanh','rat','cubic'):
  a=features(x,kind); p=np.empty_like(y)
  for f in np.unique(block):
   t=block!=f; p[~t]=a[~t]@fit(a[t],y[t]);
  e=(p-y)**2; near=keep & (abs(x).max(1)<np.quantile(abs(x).max(1),.4)); worst=max(np.sqrt(e[keep & (block==f)].mean()) for f in np.unique(block)); print(kind,'rmse',np.sqrt(e[keep].mean(0)),'near',np.sqrt(e[near].mean()),'worst block',worst)
  if kind in ('linear','pure3','pure5','tanh','rat'): print('coef',fit(a,y).T)
