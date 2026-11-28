import pickle,numpy as np,importlib.util
from itertools import combinations_with_replacement
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; b=c['single_bank_sweeps'];print('ref/raw ranges',x.min(0),x.max(0),y.min(0),y.max(0));print('single bank',[(k,np.shape(v),np.asarray(v)[:12]) for k,v in b.items()]);
for deg in range(1,5):
 for sparse in [True,False]:
  ids=[i for d in range(1,deg+1) for i in combinations_with_replacement(range(4),d) if not sparse or len(set(i))==1]; X=np.column_stack([np.ones(len(x))]+[x[:,i].prod(1) for i in ids]); err=[]
  for test in np.array_split(np.arange(len(x)),5):
   train=np.setdiff1d(np.arange(len(x)),test); fit=np.linalg.lstsq(X[train],y[train],rcond=None)[0];err.append(np.sqrt(np.mean((X[test]@fit-y[test])**2,axis=0)))
  print(deg,sparse,np.max(err,axis=0));
  if deg==3 and sparse: print(np.round(np.linalg.lstsq(X,y,rcond=None)[0],5))
sp=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m100_c03_s00_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m); S,U=m._responses(s)
for p in item['campaign_diagnostics']:
 yc=m._calibrate(s,p['y_pulse_raw']); yp=S@p['w_pulse']; r=(yc-yp).reshape(len(s['mu_shifts']),-1);print('pulse scale/RMSE',np.linalg.norm(yc),np.sqrt(np.mean(r*r,axis=1)));print('pulse residual',r[0].reshape(-1,4).round(4))
