import pickle,numpy as np,itertools,json
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Y=c['raw_rosette']; clean=np.max(abs(Y),axis=1)<5
print('gross rows',np.flatnonzero(~clean),'obs ranges',s['y_obs_raw'].reshape(-1,4).min(0),s['y_obs_raw'].reshape(-1,4).max(0)); print('single bank',np.round(np.column_stack(list(c['single_bank_sweeps'].values())),5))
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('apparatus',[(r['completed_campaign_short_name'],np.round(np.mean(r['facility_telemetry_trace'],axis=0)[[0,5]],3)) for r in (a if isinstance(a,list) else next(v for v in a.values() if isinstance(v,list)))]); print('aux mean',s['aux_manifold_telemetry'].mean(0))
for label,cols in [('linear',[X]),('diag cubic',[X,X**3]),('diag poly5',[X,X**2,X**3,X**4,X**5]),('odd to 9',[X,X**3,X**5,X**7,X**9]),('full cubic',[np.prod(X[:,j],axis=1)[:,None] for d in (1,2,3) for j in itertools.combinations_with_replacement(range(4),d)]),('diag tanh',[X,np.tanh(X)]),('diag rational',[X,X/(1+X**2)])]:
 A=np.column_stack([np.ones(len(X)),*cols]); E=np.empty_like(Y)
 for k in np.flatnonzero(clean):
  tr=clean.copy(); tr[k]=False; E[k]=Y[k]-A[k]@np.linalg.lstsq(A[tr],Y[tr],rcond=None)[0]
 B=np.linalg.lstsq(A[clean],Y[clean],rcond=None)[0]; F=Y-A@B
 print(label,'in',np.round(np.sqrt(np.mean(F[clean]**2,axis=0)),7),'loo',np.round(np.sqrt(np.mean(E[clean]**2,axis=0)),7),'max',round(np.max(abs(E[clean])),7))
 if label in ('diag cubic','diag tanh','diag poly5'): print(np.round(B,5))
