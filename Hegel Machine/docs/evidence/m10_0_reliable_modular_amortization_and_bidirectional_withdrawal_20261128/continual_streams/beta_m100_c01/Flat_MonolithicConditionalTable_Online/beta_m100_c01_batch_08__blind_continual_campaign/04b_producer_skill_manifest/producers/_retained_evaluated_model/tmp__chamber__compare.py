import pathlib,inspect,pickle,sys
import numpy as np
D=pathlib.Path('/tmp/hegel_continual_archive')
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; pulses=it['campaign_diagnostics']; w=np.array([p['w_pulse'] for p in pulses]); y=np.array([p['y_pulse_raw'] for p in pulses]); Z=s['Z_nuisance']; ij=s['sensor_ij']; xx=np.tile(ij[:,0],len(s['mu_shifts'])); yy=np.tile(ij[:,1],len(s['mu_shifts'])); wall=(xx==0)|(xx==s['nx']-1)|(yy==0)|(yy==s['ny']-1)
cal={}; operators={}
for f in sorted(D.glob('*solver.py')):
 m={}; exec(compile(f.read_text(),str(f),'exec'),m); name=f.stem.replace('beta_m100_c01_','').replace('_solver',''); c=m['_calibrate']; t=m['_transport'];
 try:
  operators[name]=t(s)[0]; cal[name]=np.array([c(dict(s,y_obs_raw=r),r) if len(inspect.signature(c).parameters)>1 else c(dict(s,y_obs_raw=r)) for r in y]); sw=s['sensor_calibration_sweeps']; fake=dict(s,y_obs_raw=sw['raw_rosette'].ravel()); a=c(fake,fake['y_obs_raw']) if len(inspect.signature(c).parameters)>1 else c(fake); print(name,'sweep inverse RMSE',np.std(a-sw['ref_rosette'].ravel()))
 except Exception as e: print(name,repr(e))
for cn,obs in cal.items():
 for tn,S in operators.items():
  r=obs-w@S.T; q=r-np.linalg.lstsq(Z,r.T,rcond=None)[0].T@Z.T
  print(cn,tn,'direct',np.round(np.sqrt(np.mean(r*r,axis=1)),5),'nuisance',np.round(np.sqrt(np.mean(q*q,axis=1)),5),'wall/core',round(np.sqrt(np.mean(q[:,wall]**2)),5),round(np.sqrt(np.mean(q[:,~wall]**2)),5))
