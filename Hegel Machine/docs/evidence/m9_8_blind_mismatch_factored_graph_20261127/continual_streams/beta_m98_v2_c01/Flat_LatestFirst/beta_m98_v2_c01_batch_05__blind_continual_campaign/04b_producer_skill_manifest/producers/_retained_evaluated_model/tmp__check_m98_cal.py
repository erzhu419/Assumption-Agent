import importlib.util, json, pickle
import numpy as np
from pathlib import Path
p=Path('/tmp/hegel_continual_archive')
def load(tag):
 f=next(p.glob('*'+tag+'_solver.py')); spec=importlib.util.spec_from_file_location(tag,f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; sb=c['single_bank_sweeps']; latest=load('batch_04')
print('latest basis source:')
import inspect
print(inspect.getsource(latest._rosette_basis))
def fit(x,y,t,form):
 if form=='diagonal':
  return np.column_stack([np.column_stack((np.ones(len(t)),t[:,j])) @ np.linalg.lstsq(np.column_stack((np.ones(len(x)),x[:,j])),y[:,j],rcond=None)[0] for j in range(4)])
 basis=(lambda a: np.column_stack((np.ones(len(a)),a))) if form=='affine' else latest._rosette_basis
 return basis(t) @ np.linalg.lstsq(basis(x),y,rcond=None)[0]
for form in ('diagonal','affine','nonlinear'):
 e=np.empty_like(y)
 for k in range(5):
  ix=np.arange(len(x))%5==k; e[ix]=fit(x[~ix],y[~ix],x[ix],form)-y[ix]
 pred=fit(x,y,np.eye(4),form); z=fit(x,y,np.zeros((1,4)),form)[0]
 print(form,'RMSE by bank',np.sqrt(np.mean(e**2,axis=0)),'max',abs(e).max(),'jac diagnostic',np.round(pred-z,6))
bank=sb['bank_id']; xb=sb['ref_signal']; yb=sb['raw_readout']
for j in range(4):
 coef=np.polynomial.polynomial.polyfit(xb[bank==j],yb[bank==j],1); v=np.zeros((sum(bank==j),4)); v[:,j]=xb[bank==j]
 print('single',j,coef,'joint prediction rmse',np.sqrt(np.mean((fit(x,y,v,'diagonal')[:,j]-yb[bank==j])**2)))
log=json.load(open(p/'archive_apparatus_log.json')); a=np.array([e['facility_telemetry_trace'] for e in log]); print('latest log centroid distances',np.linalg.norm(a.mean(axis=1)-a[-1].mean(axis=0),axis=1))
for tag in ('s00','batch_01','batch_02','batch_03','batch_04'):
 m=load(tag); out=m.solve_batch(s); w=out['weights']; kc=out['kkt_certificate']; print(tag,'J',kc['primal_objective'],'gap',kc['primal_objective']-kc['dual_lower_bound'],'w',np.round(w,5))
