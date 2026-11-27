import glob,runpy,pickle,numpy as np
m=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); s0=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=s0['study']; P=s0['campaign_diagnostics']; C=s['sensor_calibration_sweeps']; x=C['ref_rosette']; y=C['raw_rosette']; d=np.c_[np.ones(len(x)),x]; b=np.linalg.lstsq(d,y,rcond=None)[0]; print('ref range',x.min(0),x.max(0),'affine rms',np.sqrt(np.mean((d@b-y)**2,axis=0)),'coefficients',b,sep='\n');
for a in [1,2,3]:
 from itertools import combinations_with_replacement
 D=np.c_[d,*[np.prod(x[:,i],axis=1) for j in range(2,a+1) for i in combinations_with_replacement(range(4),j)]]; r=[]
 for k in range(5):
  t=np.arange(len(x))%5==k; r.extend(D[t]@np.linalg.lstsq(D[~t],y[~t],rcond=None)[0]-y[t])
 print('CV',a,np.sqrt(np.mean(np.array(r)**2,axis=0)))
for k,v in C['single_bank_sweeps'].items(): print(k,v.reshape(4,-1))
S,_=m['_forward'](s)
for p in P:
 pred=S@p['w_pulse']; obs=m['_calibrate'](dict(s,y_obs_raw=p['y_pulse_raw'])); err=obs-pred; print('pulse signals/rms by shift',np.linalg.norm(pred),np.sqrt(np.mean(err.reshape(3,-1)**2,axis=1)), 'projected rms',np.std(err-s['Z_nuisance']@np.linalg.lstsq(s['Z_nuisance'],err,rcond=None)[0])); print('first rosettes pred/obs',pred[:12],obs[:12]);
print('sensor coords',s['sensor_ij'].tolist())
