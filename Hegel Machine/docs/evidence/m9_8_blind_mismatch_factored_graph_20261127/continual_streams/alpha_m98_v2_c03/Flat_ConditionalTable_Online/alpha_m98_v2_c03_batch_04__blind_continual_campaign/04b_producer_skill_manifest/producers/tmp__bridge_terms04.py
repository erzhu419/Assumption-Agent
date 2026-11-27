import pickle
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares
from bridge04 import forward_fit
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; Z=C['true_channel_loads']; X=C['bridge_raw']; _,good=forward_fit(Z,X,3)
mon=[()]+[(i,) for i in range(4)]+list(combinations_with_replacement(range(4),2))+list(combinations_with_replacement(range(4),3))
A=np.column_stack([np.prod(Z[:,ix],axis=1) for ix in mon]); fold=np.empty(len(Z),int); fold[np.random.default_rng(816).permutation(len(Z))]=np.arange(len(Z))%5
sizes={'affine':lambda s:len(s)<=1,'diag-cubic':lambda s:len(s)<=1 or len(set(s))==1,'pair+cubic':lambda s:len(s)<=2 or len(set(s))==1,'odd-total':lambda s:len(s)!=2,'all':lambda s:True}
for label,sel in sizes.items():
 j=np.array([sel(s) for s in mon]); a=A[:,j]; b=np.linalg.lstsq(a[good],X[good],rcond=None)[0]; e=np.empty_like(X)
 for k in range(5):
  h=fold==k; f=(~h)&good; e[h]=a[h]@np.linalg.lstsq(a[f],X[f],rcond=None)[0]-X[h]
 print(label,'forward CV rms/worst load band',np.sqrt(np.mean(e[good]**2)),max(np.sqrt(np.mean(v**2)) for v in np.array_split(e[np.flatnonzero(good)[np.argsort(Z[good].sum(1))]],5)), 'fit clean med/max',np.median(np.linalg.norm(a[good]@b-X[good],axis=1)),np.max(np.abs(a[good]@b-X[good])))
 if label in ('pair+cubic','all'):
  print('terms',[(s,np.round(v,4).tolist()) for s,v in zip(np.array(mon,dtype=object)[j],b) if max(abs(v))>.008]); r=np.linalg.norm(a@b-X,axis=1); print('rows by residual',[(int(i),round(r[i],4)) for i in np.argsort(r)[-15:]])
# Fit clean isolated sweeps to estimate the self-cubic independently of mixed terms.
for i in range(4):
 j=good&(np.count_nonzero(Z,axis=1)==1)&(Z[:,i]!=0); a=np.c_[np.ones(j.sum()),Z[j,i],Z[j,i]**3]; b=np.linalg.lstsq(a,X[j],rcond=None)[0]; print('axis',i, 'coefficients',np.round(b,5).tolist(),'maxerr',np.max(abs(a@b-X[j])))
