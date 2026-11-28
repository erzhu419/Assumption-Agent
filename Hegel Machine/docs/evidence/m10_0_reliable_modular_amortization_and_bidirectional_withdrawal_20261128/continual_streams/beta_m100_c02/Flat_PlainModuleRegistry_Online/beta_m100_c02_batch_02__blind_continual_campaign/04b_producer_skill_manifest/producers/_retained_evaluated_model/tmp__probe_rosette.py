import pickle,itertools
import numpy as np
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; single=sw['single_bank_sweeps']; print('single bank rows'); print(np.c_[single['bank_id'],single['ref_signal'],single['raw_readout']].reshape(4,-1,3).round(6)); print('zero rows',np.c_[x,y][np.all(x==0,axis=1)]); print('operator scale');
for spec in ['affine','diag3','cross2_diag3','full3','full4','cross2_diag5']:
 p=[()] + [(i,) for i in range(4)];
 if spec in ['diag3','cross2_diag3','cross2_diag5']: p += [(i,)*d for d in ([2,3,4,5] if spec.endswith('5') else [2,3]) for i in range(4)]
 if spec.startswith('cross'): p += list(itertools.combinations(range(4),2))
 if spec.startswith('full'): p += [t for d in range(2,int(spec[-1])+1) for t in itertools.combinations_with_replacement(range(4),d)]
 A=np.column_stack([np.prod(x[:,t],axis=1) for t in p]); pred=np.zeros_like(y)
 def fit(train):
  w=np.ones(train.sum()); a,b=A[train],y[train];
  for _ in range(22):
   c=lstsq(a*w[:,None],b*w[:,None])[0]; r=np.linalg.norm(a@c-b,axis=1); w=1/np.sqrt(1+(r/max(2*np.median(r),1e-10))**2)
  return c
 c=fit(np.ones(len(x),bool)); main=np.linalg.norm(A@c-y,axis=1); clean=main<3
 for fold in range(5):
  test=np.arange(len(x))%5==fold; pred[test]=A[test]@fit(~test)
 r=np.linalg.norm(pred-y,axis=1); print(spec,'rank',np.linalg.matrix_rank(A),'fit',np.quantile(main[clean],[.5,.9,1]),'OOF',np.quantile(r[clean],[.5,.9,1]),'outliers',np.flatnonzero(~clean));
 if spec in ('cross2_diag3','full3'): print('coefficients',list(zip(p,c.round(6).tolist())))
