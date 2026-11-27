import glob, importlib.util, pickle
import numpy as np
from itertools import combinations_with_replacement
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=item['study']; sweeps=s['sensor_calibration_sweeps']; x=sweeps['ref_rosette']; y=sweeps['raw_rosette']
mods={}
for name in ('01','02','03','06'):
 p=glob.glob('/tmp/hegel_continual_archive/*batch_'+name+'_solver.py')[0]; sp=importlib.util.spec_from_file_location('b'+name,p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); mods[name]=m
Z=s['Z_nuisance']; P=lambda a:a-Z@np.linalg.lstsq(Z,a,rcond=None)[0]
for name in ('01','02','03'):
 S,_=mods[name]._forward(s)
 for cal in ('02','06'):
  for i,d in enumerate(item['campaign_diagnostics']):
   pred=S@d['w_pulse']; target=mods[cal]._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])); r=target-pred
   print('pulse',name,cal,i,'raw-cal rms / projected rms / max',*[round(float(v),8) for v in (np.linalg.norm(r)/np.sqrt(len(r)),np.linalg.norm(P(r))/np.sqrt(len(r)),max(abs(r)))])
for degree in (1,2,3):
 powers=[idx for d in range(degree+1) for idx in combinations_with_replacement(range(4),d)]
 D=np.column_stack([np.prod(x[:,idx],axis=1) for idx in powers]); E=[]
 for i in range(len(y)):
  train=np.arange(len(y))!=i; E.append(y[i]-D[i]@np.linalg.lstsq(D[train],y[train],rcond=None)[0]);
 E=np.linalg.norm(E,axis=1); print('sweep LOO degree',degree,'quantiles',np.quantile(E,[0,.5,.9,.95,1]),'worst rows',np.argsort(E)[-8:])
for k,v in sweeps['single_bank_sweeps'].items():print('single bank',k,np.shape(v),np.asarray(v)[:4])
