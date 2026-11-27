import pickle, numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; sb=c['single_bank_sweeps']; np.set_printoptions(precision=6,suppress=True,linewidth=140)
print('axis-only rows raw:\n',np.c_[x[:16],y[:16]]); print('standalone sweeps:\n', np.column_stack([sb[k] for k in ('bank_id','ref_signal','raw_readout')]))
for name, cols in [('affine',[()]+[(i,) for i in range(4)]),('separable cubic',[()]+[(i,)*j for j in range(1,4) for i in range(4)]),('full quadratic',[()]+[z for j in range(1,3) for z in combinations_with_replacement(range(4),j)]),('full cubic',[()]+[z for j in range(1,4) for z in combinations_with_replacement(range(4),j)])]:
 a=np.array([np.prod(x[:,p],axis=1) for p in cols]).T
 errors=np.empty_like(y)
 for fold in np.array_split(np.random.default_rng(5).permutation(len(x)),5):
  tr=np.setdiff1d(np.arange(len(x)),fold); errors[fold]=a[fold]@np.linalg.lstsq(a[tr],y[tr],rcond=None)[0]-y[fold]
 b=np.linalg.lstsq(a,y,rcond=None)[0]
 print(name,'OOF per-channel RMSE',np.sqrt(np.mean(errors**2,axis=0)),'max',abs(errors).max(),'training max',abs(a@b-y).max()); print([(p,row.tolist()) for p,row in zip(cols,b.round(6)) if abs(row).max()>1e-5])
print('telemetry correlations with raw and raw ranges',np.corrcoef(np.c_[s['aux_manifold_telemetry'],s['y_obs_raw'].reshape(-1,4)].T)[:4,4:],s['y_obs_raw'].reshape(-1,4).min(0),s['y_obs_raw'].reshape(-1,4).max(0))
