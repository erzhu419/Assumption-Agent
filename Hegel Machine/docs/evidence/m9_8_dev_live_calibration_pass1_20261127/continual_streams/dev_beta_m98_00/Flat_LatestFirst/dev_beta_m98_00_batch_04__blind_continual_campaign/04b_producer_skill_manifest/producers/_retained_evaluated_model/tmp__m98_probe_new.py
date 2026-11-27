import importlib.util,json,pickle,numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/'
spec=importlib.util.spec_from_file_location('arch','/holdout_task/batch_solver.py'); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; src=s['source_modes']; c=a._calibration(s['sensor_calibration_sweeps']); pulses=s['reference_tracer_pulses']; m=np.array([np.tensordot(t['w_pulse'],src,axes=1) for t in pulses]); y=np.column_stack([c(t['y_pulse_raw']) for t in pulses]); print('pulse weights', [t['w_pulse'] for t in pulses]); print('pulse observed RMS', np.sqrt((y*y).mean(axis=0))); print('sweep inverse RMS by bank',np.sqrt(np.mean((c(s['sensor_calibration_sweeps']['raw_rosette'])-s['sensor_calibration_sweeps']['ref_rosette'])**2,axis=0)))
for groups in ([],[[0,1],[4,5]],[[0],[1],[4],[5]],[[i] for i in range(8)]):
 def expand(x):
  w=np.ones(8)
  for g,v in zip(groups,x): w[g]=v
  return w
 def res(w): return a._response(s,expand(w),m)[0]-y
 def fit(idx):
  return least_squares(lambda w:res(w)[:,idx].ravel(),np.ones(len(groups)),bounds=(0,10),gtol=1e-14).x if groups else []
 w=fit(slice(None)); print('wall',np.round(expand(w),6),'rms',np.sqrt(np.mean(res(w)**2,axis=0)),'max',abs(res(w)).max(),'LOPO',[np.sqrt(np.mean(res(fit([1-i]))[:,i]**2)) for i in range(2)])
for i,t in enumerate([np.array(t['facility_telemetry_trace']) for t in json.load(open(p+'archive_apparatus_log.json'))]+[s['aux_manifold_telemetry']]):
 print('TELEMETRY',i,'raw range',np.round(np.ptp(t,axis=0),3),'med',np.round(np.median(t,axis=0),4),'iqr',np.round(np.percentile(t,75,axis=0)-np.percentile(t,25,axis=0),4)); print(np.round(t,3))
