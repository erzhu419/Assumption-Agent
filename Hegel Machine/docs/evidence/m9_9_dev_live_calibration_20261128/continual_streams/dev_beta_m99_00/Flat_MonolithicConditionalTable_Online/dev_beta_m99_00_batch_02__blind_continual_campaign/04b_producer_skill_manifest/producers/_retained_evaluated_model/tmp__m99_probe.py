import pickle, numpy as np, importlib.util, json
from itertools import combinations_with_replacement
from scipy.sparse import diags
p='/tmp/hegel_continual_archive/dev_beta_m99_00_batch_01_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; cal=s['sensor_calibration_sweeps']; x=cal['ref_rosette']; y=cal['raw_rosette']
R=np.array([p['y_pulse_raw'] for p in it['campaign_diagnostics']]); W=np.array([p['w_pulse'] for p in it['campaign_diagnostics']]).T
Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
print('cal ranges',np.round([x.min(0),x.max(0),y.min(0),y.max(0)],4));print('pulse ranges raw',R.min(),R.max());print('single banks',np.round(np.column_stack(list(cal['single_bank_sweeps'].values())),5).tolist())
def features(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for a in range(1,d+1) for j in combinations_with_replacement(range(4),a)])
for d in range(1,5):
 A=features(x,d); C=np.linalg.lstsq(A,y,rcond=None)[0]; E=y-A@C; lev=np.sum(A*np.linalg.pinv(A).T,axis=1)
 print('cal degree',d,'ncols',A.shape[1],'train',np.sqrt(np.mean(E**2,axis=0)),'worst LOO',np.max(np.abs(E/(1-lev[:,None]+1e-15)),axis=0));print('coeff',[(j,np.round(v,5).tolist()) for j,v in enumerate(C) if np.max(np.abs(v))>0.002][:40])
axis=m._axis
for leakx,leaky in [(0,0),(1,0),(0,1),(1,1)]:
 params=iter([leakx,leaky]); keys=iter('xy')
 def boundary(k,v,h):
  q=next(params)*s['robin_'+next(keys)]/h
  return axis(k,v,h)+diags(np.r_[q,np.zeros(len(k)-2),q],format='csc')
 m._axis=boundary
 S,U=m._forward(s); P=(S@W).T; b=np.linalg.lstsq(features(x,1),y,rcond=None)[0]; e=m._calibrate(dict(s,y_obs_raw=R.ravel())).reshape(R.shape)-P
 print('leak',leakx,leaky,'pred range',np.min(P),np.max(P),'calibrated pulse rms',np.sqrt(np.mean(e**2,axis=1)),'after nuisance',np.sqrt(np.mean((e-e@Q@Q.T)**2,axis=1)))
for a in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 T=np.array(a['facility_telemetry_trace']);print('archive',a['completed_campaign_short_name'],'means/std',np.round(T.mean(0),3),np.round(T.std(0),3))
