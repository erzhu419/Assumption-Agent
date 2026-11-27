import importlib.util,pickle,numpy as np
from pathlib import Path
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; print('calibration',x.shape,y.shape,'bank',type(b).__name__,list(b) if isinstance(b,dict) else [(type(v).__name__,list(v) if isinstance(v,dict) else np.shape(v)) for v in b[:4]]); print('bank ids',s['row_bank_id'][:20]); A=np.c_[np.ones(len(x)),x]; B=np.linalg.lstsq(A,y,rcond=None)[0]; print('affine rmse',np.mean((A@B-y)**2)**.5,'coeff',B.round(5)); print('pulse weights',[p['w_pulse'].round(3) for p in a['campaign_diagnostics']]);
for p in sorted(Path('/tmp/hegel_continual_archive').glob('*solver.py')):
 spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); S,U=m._forward(s);
 for j,d in enumerate(a['campaign_diagnostics']):
  pred=S@d['w_pulse']; actual=m._calibrate({**s,'y_obs_raw':d['y_pulse_raw']}); err=actual-pred; errn=err-s['Z_nuisance']@np.linalg.lstsq(s['Z_nuisance'],err,rcond=None)[0]; print(p.stem,j,'relative',np.linalg.norm(err)/np.linalg.norm(actual),'projected',np.linalg.norm(errn)/np.linalg.norm(actual),'per shift rmse',np.sqrt(np.mean(err.reshape(3,-1)**2,axis=1)))
