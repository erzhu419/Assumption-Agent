import glob, importlib.util, itertools, pickle
import numpy as np

def load(pattern):
    spec=importlib.util.spec_from_file_location('prior',glob.glob('/tmp/hegel_continual_archive/*'+pattern+'_solver.py')[0]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; d=p['campaign_diagnostics']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; m=load('batch_06'); poly=load('batch_03'); w=np.stack([a['w_pulse'] for a in d]); raw=np.stack([a['y_pulse_raw'] for a in d]); learn=poly._calibration(c)
def read(x,coeff):
    return np.column_stack([poly._design(x,j)@coeff[j] for j in range(4)])
e=read(x,learn)-y; print('polynomial absolute residual percentiles',np.percentile(abs(e),[50,80,90,95,99])); print('high-residual rows',np.where(np.max(abs(e),axis=1)>2e-4)[0]);
for walls in itertools.product(('open','reflect','periodic'),repeat=2):
    S,U=m._forward(s,walls); pred=read((w@S.T).reshape(-1,4),learn).reshape(raw.shape); er=pred-raw; print(walls,'RMSE by pulse',np.sqrt((er**2).mean(1)).round(8),'worst by shift',np.sqrt((er.reshape(3,len(s['mu_shifts']),-1)**2).mean(2)).max(0).round(8))
