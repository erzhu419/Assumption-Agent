import glob, importlib.util, pickle, numpy as np
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; c=s['sensor_calibration_sweeps']; Z=s['Z_nuisance']
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 spec=importlib.util.spec_from_file_location('m',f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 try:
  S,U=m._design(s); print(f.split('/')[-1],S.shape)
  for p in item['campaign_diagnostics']:
   yy=np.asarray(m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw']))); pred=S@p['w_pulse']; r=yy-pred; rp=r-Z@np.linalg.lstsq(Z,r,rcond=None)[0]; print('  pulse rel residual',np.linalg.norm(r)/np.linalg.norm(yy),np.linalg.norm(rp)/np.linalg.norm(yy),'per shift',np.sqrt(np.mean(r.reshape(3,-1)**2,axis=1)))
 except Exception as e: print(type(e).__name__,e)
