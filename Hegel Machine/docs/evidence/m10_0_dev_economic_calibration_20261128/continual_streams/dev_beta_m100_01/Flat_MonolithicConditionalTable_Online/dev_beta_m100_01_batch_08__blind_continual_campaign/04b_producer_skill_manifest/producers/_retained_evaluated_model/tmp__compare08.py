import glob, importlib.util, itertools, pickle
import numpy as np

def load(pattern):
    spec=importlib.util.spec_from_file_location('prior',glob.glob('/tmp/hegel_continual_archive/*'+pattern+'_solver.py')[0]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; d=p['campaign_diagnostics']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; idx=np.arange(len(x)); m=load('batch_06'); poly=load('batch_03'); w=np.stack([a['w_pulse'] for a in d]); raw=np.stack([a['y_pulse_raw'] for a in d]); learn=m._fit_calibration(*m._sweeps(s)); print('sin params',learn.round(6)); print('sin residual percentiles',np.percentile(abs(m._readout(x,learn)-y),[50,80,90,95,99])); print('poly params',poly._calibration(c))
for j in range(4):
    print('raw outliers bank',j,idx[abs(y[:,j])>5]);
for walls in itertools.product(('open','reflect','periodic'),repeat=2):
    S,U=m._forward(s,walls); pred=m._readout((w@S.T).reshape(-1,4),learn).reshape(raw.shape); er=pred-raw; print(walls,'raw RMSE per pulse',np.sqrt((er**2).mean(1)).round(7),'max',np.max(abs(er)).round(7))
