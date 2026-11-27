import importlib.util,json,pickle
import numpy as np
from scipy.optimize import least_squares
from pathlib import Path
root=Path('/tmp/hegel_continual_archive')
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; sw=s['sensor_calibration_sweeps']
def load(x):
    sp=importlib.util.spec_from_file_location('m', root/x); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=load('beta_m98_v2_c03_batch_02_solver.py')
ref,raw=sw['ref_rosette'],sw['raw_rosette']; X=np.c_[np.ones(len(ref)),ref]; coef=np.linalg.lstsq(X,raw,rcond=None)[0]; print('affine coefficients',coef.round(7)); print('affine fit RMS by channel',np.sqrt(np.mean((raw-X@coef)**2,axis=0)))
err=[]
for idx in np.array_split(np.arange(len(ref)),5):
    train=np.setdiff1d(np.arange(len(ref)),idx); fit=np.linalg.lstsq(X[train],raw[train],rcond=None)[0]; err.append(np.sqrt(np.mean((raw[idx]-X[idx]@fit)**2,axis=0)))
print('worst held calibration RMS',np.max(err,axis=0)); print('bank sweep examples', {k: v[:8] for k,v in sw['single_bank_sweeps'].items()})
W=np.array([p['w_pulse'] for p in item['campaign_diagnostics']]); Y=np.array([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in item['campaign_diagnostics']]); print('pulse ranges',[(float(y.min()),float(y.max())) for y in Y]); print('pulse weights',W)
for name in ('beta_m98_v2_c03_batch_01_solver.py','beta_m98_v2_c03_batch_02_solver.py'):
    S,_=load(name)._forward(s); E=W@S.T-Y; print(name,'pulse RMS by shift',np.sqrt(np.mean(E.reshape(len(W),len(s['mu_shifts']),-1)**2,axis=2)), 'max',abs(E).max())
def res(p,ids):
    S,_=m._forward(dict(s,robin_x=p[0],robin_y=p[1])); return (W[ids]@S.T-Y[ids]).ravel()
for ids in ([0],[1],[0,1]):
    fit=least_squares(res,[s['robin_x'],s['robin_y']],args=(ids,),bounds=(-0.1,50)); print('fit axes',ids,fit.x,'RMS each',np.sqrt(np.mean(res(fit.x,[0,1]).reshape(2,-1)**2,axis=1)))
log=json.load(open(root/'archive_apparatus_log.json')); print('log layout',[(list(a),{k:np.shape(v) for k,v in a.items()}) for a in log] if isinstance(log,list) else list(log))
