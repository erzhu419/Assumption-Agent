import sys,pickle,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m98_v2_c03_batch_02_solver as m
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=x['study']; cs=s['sensor_calibration_sweeps']; r=cs['ref_rosette']; y=cs['raw_rosette']; X=np.c_[np.ones(len(r)),r]; B=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@B; loo=e/(1-np.sum(X*np.linalg.pinv(X).T,axis=1))[:,None]; print('calibration matrix and offset',B.round(6)); print('calibration LOO RMSE by channel',np.sqrt(np.mean(loo**2,0))); print('pulse cal ranges',[(np.min(m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw']))),np.max(m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])))) for p in x['campaign_diagnostics']]); Z=s['Z_nuisance']; W=np.column_stack([p['w_pulse'] for p in x['campaign_diagnostics']]); Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in x['campaign_diagnostics']]); P=lambda e:e-Z@np.linalg.lstsq(Z,e,rcond=None)[0]; scale=np.sqrt(np.mean(Y.reshape(3,32,-1)**2,axis=1)); print('pulse scales by shift',scale.round(6));
for a in (0,1):
 for b in (0,1):
  D,_=m._forward(dict(s,robin_x=a*s['robin_x'],robin_y=b*s['robin_y'])); E=D@W-Y; rms=lambda e:np.sqrt(np.mean(e.reshape(3,32,-1)**2,axis=1)).round(7); print((a,b),'raw error',rms(E),'projected error',rms(P(E)))
print('vel endpoints',s['vel_x'][[0,-1]],s['vel_y'][[0,-1]])
