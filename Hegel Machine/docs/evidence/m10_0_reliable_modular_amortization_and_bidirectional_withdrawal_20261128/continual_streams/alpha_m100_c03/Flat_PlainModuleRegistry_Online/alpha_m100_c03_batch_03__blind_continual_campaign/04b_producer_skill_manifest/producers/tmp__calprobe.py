import pickle,numpy as np
from collections import Counter
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; print('sweeps',[(k,np.shape(v)) for k,v in C['single_channel_sweeps'].items()]); print('keys',D.keys()); print('cal designs',Counter(np.count_nonzero(X,axis=1))); print('axes',np.round(np.c_[X,Y][np.count_nonzero(X,axis=1)<=1],4));
from itertools import combinations_with_replacement

def basis(x,p):
 a=[np.ones(len(x))]
 for d in range(1,p+1):
  a.extend(np.prod(x[:,j],axis=1) for j in combinations_with_replacement(range(4),d))
 return np.stack(a,axis=1)

def fit(A,y):
 w=np.ones(len(A));
 for _ in range(45):
  b=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]; r=np.linalg.norm(A@b-y,axis=1); s=max(np.median(r),1e-8); w=1/np.sqrt(1+(r/(2*s))**2)
 return b,r
for p in [1,2,3]:
 A=basis(X,p); b,r=fit(A,Y); print('p',p,'r sorted',np.round(np.sort(r)[::3],5)); print('coef',np.round(b,4)); o=np.zeros(len(X));
 for i in range(len(X)):
  m=np.arange(len(X))!=i; o[i]=np.linalg.norm(A[i]@fit(A[m],Y[m])[0]-Y[i]);
 print('LOO core',np.quantile(o[r<1],[.5,.85,.95,1])); print('LOO axes',np.quantile(o[(r<1)&(np.count_nonzero(X,axis=1)<=1)],[.5,.95,1]));
for k in ['train','test','chamber_verification_trials']:
 s=D.get(k); print(k, None if s is None else [(a,np.shape(v)) for a,v in s.items()]);
 if s is not None:
  for a in ['past_channel_raw','hold_channel_raw']:
   y=np.asarray(s[a]).reshape(-1,4); print(a,np.round(np.quantile(y,[0,.5,1],axis=0),3))
