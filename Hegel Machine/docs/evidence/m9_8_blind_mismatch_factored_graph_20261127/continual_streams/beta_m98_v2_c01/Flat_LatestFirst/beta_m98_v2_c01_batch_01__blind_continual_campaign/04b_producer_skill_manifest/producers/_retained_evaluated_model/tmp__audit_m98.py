import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(p+'public_batch.pkl','rb') as f: it=pickle.load(f)[0]
s=it['study']; S,_=m._forward(s); pulses=it['campaign_diagnostics']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; raw=c['raw_rosette']
aff=np.c_[x,np.ones(len(x))]; quad=np.column_stack([aff,*[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]])
for name,X in [('affine',aff),('pairwise products and squares',quad)]:
 worst=0.
 for fold in np.array_split(np.argsort(np.max(np.abs(x),axis=1)),5):
  train=np.setdiff1d(np.arange(len(x)),fold); coef=np.linalg.lstsq(X[train],raw[train],rcond=None)[0]; e=X[fold]@coef-raw[fold]; worst=max(worst,np.sqrt(np.mean(e**2,axis=0)).max())
 print(name,'worst held-out amplitude block and channel RMSE',worst)
 if name=='affine': assert worst<3e-5
bank=c['single_bank_sweeps']; coef=np.linalg.lstsq(aff,raw,rcond=None)[0]; disagreement=[]
for j in range(4):
 mask=bank['bank_id']==j; q=bank['ref_signal'][mask]; fit=np.linalg.lstsq(np.c_[q,np.ones(len(q))],bank['raw_readout'][mask],rcond=None)[0]; disagreement.append(np.max(np.abs(fit-coef[[j,4],j])))
print('Max independent single-bank gain/offset disagreement',max(disagreement)); assert max(disagreement)<5e-5
for tag in 'xy':
 A=m._axis(s['kappa_'+tag],s['vel_'+tag],s['L'+tag]/(s['n'+tag]-1)); err=np.max(np.abs(A@np.ones(A.shape[0]))); print(tag,'axis constant-field residual',err); assert err<1e-10
Z=s['Z_nuisance']; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); banks=np.asarray(s['row_bank_id']).reshape(-1); shifts=np.repeat(np.arange(len(s['mu_shifts'])),len(s['sensor_ij'])); assert len(shifts)==len(banks)==len(Z)
worst=np.zeros(2)
for t in pulses:
 d=m._calibrate(dict(s,y_obs_raw=t['y_pulse_raw']))-S@t['w_pulse']; d_res=d-Z@(P@d)
 for k in np.unique(shifts):
  for b in np.unique(banks):
   mask=(shifts==k)&(banks==b)
   if mask.any(): worst=np.maximum(worst,[np.sqrt(np.mean(v[mask]**2)) for v in (d,d_res)])
print('Worst pulse x shift x bank RMSE, before/after nuisance profiling:',worst); assert np.max(worst)<3e-6
A=S-Z@(P@S); beta=np.linalg.svd(A,compute_uv=False); print('Profiled source design smallest singular value:',beta[-1])
