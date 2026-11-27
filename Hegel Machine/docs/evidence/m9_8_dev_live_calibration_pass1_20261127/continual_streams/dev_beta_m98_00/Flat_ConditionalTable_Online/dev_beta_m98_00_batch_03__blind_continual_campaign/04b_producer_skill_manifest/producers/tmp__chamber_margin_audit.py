import importlib.util,json,pickle,numpy as np

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
root='/tmp/hegel_continual_archive/'
m=load('solver','/holdout_task/batch_solver.py'); a=load('archive',root+'dev_beta_m98_00_batch_02_solver.py')
for entry in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
 s=entry['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; folds=np.arange(len(x))%4
 b=c['single_bank_sweeps']; banks=np.asarray(b['bank_id']); bf=np.empty(len(banks),int)
 for j in range(4): bf[banks==j]=np.arange(sum(banks==j))%4
 worst={name:0. for name in ('pooled_affine','affine','quadratic','rational')}
 for f in range(4):
  tr=folds!=f; te=~tr
  xx,yy=x[tr],y[tr]; xt,yt=x[te],y[te]
  clipped=dict(ref_rosette=xx,raw_rosette=yy,single_bank_sweeps={k:np.asarray(v)[bf!=f] for k,v in b.items()})
  # Score all candidates in raw coordinates on the same held-out rosettes.
  pool_inverse=m._calibration(clipped); offset=pool_inverse(np.zeros((1,4))); inverse=pool_inverse(np.eye(4)).reshape(4,4)-offset
  preds={'pooled_affine':np.linalg.solve(inverse.T,(xt-offset).T).T}
  for d,name in ((1,'affine'),(2,'quadratic')):
   p=a._powers(d); preds[name]=a._features(xt,p)@np.linalg.lstsq(a._features(xx,p),yy,rcond=None)[0]
  preds['rational']=np.column_stack([a._rational_predict(xt,*a._rational_fit(xx,yy[:,j]-xx[:,j]))+xt[:,j] for j in range(4)])
  for name,pred in preds.items(): worst[name]=max(worst[name],np.sqrt(np.mean((pred-yt)**2,axis=0)).max())
 print('worst held-out calibration RMSE over fold/channel:',worst)
 pulses=s['reference_tracer_pulses']; cal=m._calibration(c); modes=np.tensordot(np.stack([p['w_pulse'] for p in pulses]),s['source_modes'],axes=1)
 obs=np.column_stack([cal(p['y_pulse_raw']) for p in pulses]).reshape(len(s['mu_shifts']),len(s['sensor_ij']),-1)
 from itertools import product
 pred={pair:m._response(s,pair,modes)[0].reshape(obs.shape) for pair in product(('periodic','open','sealed'),repeat=2)}
 selections={}; worst_abs=0.; margin=float('inf')
 for j in range(len(pulses)):
  for i in range(len(s['sensor_ij'])):
   keep_s=np.arange(len(s['sensor_ij']))!=i; keep_p=np.arange(len(pulses))!=j
   scores=sorted((np.sqrt(np.mean((p[:,keep_s][:,:,keep_p]-obs[:,keep_s][:,:,keep_p])**2)),k) for k,p in pred.items())
   (best,k),(runner,_)=scores[:2]; selections[k]=selections.get(k,0)+1; margin=min(margin,runner-best)
   worst_abs=max(worst_abs,np.max(abs(pred[k][:,i,j]-obs[:,i,j])))
 print('joint leave-sensor-and-pulse-out:',selections,'min train RMSE margin',margin,'worst held-out absolute residual',worst_abs)
 Z=s['Z_nuisance']; S=m._response(s,('periodic','periodic'),s['source_modes'])[0]; P=np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
 print('nuisance rank, projected source minimum singular value:',np.linalg.matrix_rank(Z),np.linalg.svd(P@S,compute_uv=False)[-1])
log=json.load(open(root+'archive_apparatus_log.json')); print('log entry metadata:',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in log]); print('log array sizes:',[{k:np.shape(v) for k,v in r.items() if isinstance(v,list)} for r in log])
