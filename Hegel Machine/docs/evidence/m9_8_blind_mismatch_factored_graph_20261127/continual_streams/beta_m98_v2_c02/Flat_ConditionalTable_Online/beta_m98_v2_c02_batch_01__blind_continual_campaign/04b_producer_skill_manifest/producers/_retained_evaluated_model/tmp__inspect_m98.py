import glob,importlib.util,pickle
import numpy as np
from itertools import combinations_with_replacement
p=glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]
spec=importlib.util.spec_from_file_location('prev',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=d['study']; sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; bank=sw['single_bank_sweeps']; print('ranges',x.min(0),x.max(0),y.min(0),y.max(0)); print('sensors',s['sensor_ij'].reshape(-1,4,2)[:4]);print('banks',s['row_bank_id'][:40]);print('bank arrays',[(k,np.shape(v),np.asarray(v).ravel()[:16]) for k,v in bank.items()]); print('telemetry',s['aux_manifold_telemetry'][:6]); print('pulses',[(p['w_pulse'],np.min(p['y_pulse_raw']),np.max(p['y_pulse_raw'])) for p in d['campaign_diagnostics']]);
for deg in range(1,5):
 terms=[t for q in range(deg+1) for t in combinations_with_replacement(range(4),q)]; F=np.array([np.prod(x[:,t],axis=1) for t in terms]).T; coeff=np.linalg.lstsq(F[:60],y[:60],rcond=None)[0]; e=F@coeff-y; print('poly',deg,'train/test rms',np.sqrt(np.mean(e[:60]**2)),np.sqrt(np.mean(e[60:]**2))); 
 if deg<=2: print(np.round(coeff,6))
S,U=m._forward(s); Z=s['Z_nuisance']; print('pulse baseline: raw affine vs calibrated errors')
C=np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]
for p in d['campaign_diagnostics']:
 pred=S@p['w_pulse']; raw=p['y_pulse_raw'].reshape(-1,4); r=(np.c_[np.ones(len(raw)),pred.reshape(-1,4)]@C-raw); cal=m._calibrate(dict(s,y_obs_raw=raw)); e=cal-pred; print('pred range',pred.min(),pred.max(),'raw rms',np.mean(r**2)**.5,'cal rms',np.mean(e**2)**.5,'cal projected rms',np.mean((e-Z@np.linalg.lstsq(Z,e,rcond=None)[0])**2)**.5);print('first residual',e.reshape(-1,4)[:3]);
