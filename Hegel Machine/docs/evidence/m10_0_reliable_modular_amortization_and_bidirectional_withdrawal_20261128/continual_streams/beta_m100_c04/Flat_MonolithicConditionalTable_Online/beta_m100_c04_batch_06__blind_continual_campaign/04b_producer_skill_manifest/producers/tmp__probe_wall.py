import importlib.util, itertools, json, pickle
import numpy as np
p=importlib.util.spec_from_file_location('batch','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'],a.mean(0).round(3),list(r))
y=m._calibrate(s); Z=s['Z_nuisance']; P=lambda z,A:m._project(z,s['nuisance_ridge'],A)
best=[]
for laws in itertools.product(((0,0),(1,1),(1,0),(0,1),'periodic'),repeat=2):
 S,_=m._forward(s,laws); errs=[]
 for idx in np.array_split(np.arange(len(y)),3):
  tr=~np.isin(np.arange(len(y)),idx); B=P(Z[tr],np.c_[S[tr],y[tr]]); w,_=m._qp(B[:,:-1],B[:,-1],s['source_ridge'],s['u_box'],s['total_budget']); gamma=np.linalg.solve(Z[tr].T@Z[tr]+s['nuisance_ridge']*np.eye(Z.shape[1]),Z[tr].T@(y[tr]-S[tr]@w)); errs.append(np.sqrt(np.mean((y[idx]-S[idx]@w-Z[idx]@gamma)**2)))
 B=P(Z,np.c_[S,y]); w,_=m._qp(B[:,:-1],B[:,-1],s['source_ridge'],s['u_box'],s['total_budget']); err=np.sqrt(np.mean((B[:,:-1]@w-B[:,-1])**2)); best.append((err,laws,np.round(errs,6)))
for a in sorted(best,key=lambda a:a[0]): print(a)
