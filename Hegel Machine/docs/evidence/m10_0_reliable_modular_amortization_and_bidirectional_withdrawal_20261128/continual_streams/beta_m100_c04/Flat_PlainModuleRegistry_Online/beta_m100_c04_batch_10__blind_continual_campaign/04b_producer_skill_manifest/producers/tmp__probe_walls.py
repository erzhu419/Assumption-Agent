import sys,json,pickle,itertools,importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m100_c04_batch_09_solver.py'); prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); c=calibrate_upstream(s,pay)
t=s['aux_manifold_telemetry']; print('current telemetry mean/sd',t.mean(0),t.std(0),'range',t.min(0),t.max(0)); print('obs magnitude',np.min(c['y_obs_raw']),np.max(c['y_obs_raw'])); print('shift',s['mu_shifts']); print('banks',s['row_bank_id'][:32])
a=s['sensor_calibration_sweeps']; ref=a['ref_rosette']; raw=a['raw_rosette']; mask=(np.abs(raw)<=8.5).all(1); print('calibration sweep shape, rejected, per-bank affine RMSE',ref.shape,(~mask).sum(),np.sqrt(np.mean((raw[mask]-ref[mask]*pay['gains']-pay['offsets'])**2,0)))
y=c['y_obs_raw']; z=s['Z_nuisance']; sensors=len(s['sensor_ij']); groups=np.tile(np.arange(sensors)//4,len(s['mu_shifts'])); shifts=np.repeat(np.arange(len(s['mu_shifts'])),sensors)
def fit(S, idx):
 X=S[idx]; Z=z[idx]; Y=y[idx]; gram=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(z.shape[1]),Z.T); T=X-Z@(gram@X); R=Y-Z@(gram@Y); w,nu=prior._box_simplex(T.T@T+s['source_ridge']*np.eye(S.shape[1]),-T.T@R,s['u_box'],s['total_budget']); return w,gram@(Y-X@w)
def cv(S, g):
 e=[]
 for j in np.unique(g):
  i=g!=j; w,b=fit(S,i); e.append(np.sqrt(np.mean((y[~i]-S[~i]@w-z[~i]@b)**2)))
 return np.sqrt(np.mean(np.square(e))),max(e)
for walls in itertools.product(('robin','sealed','periodic'),repeat=2):
 S,_=prior._forward(s,walls); w,b=fit(S,np.ones(len(y),bool)); print(walls,'full',round(np.sqrt(np.mean((y-S@w-z@b)**2)),6),'rosette mean/worst',np.round(cv(S,groups),6),'shift mean/worst',np.round(cv(S,shifts),6),'w',w.round(4))
