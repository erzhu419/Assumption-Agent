import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=m._Transport(s); decode=m._calibration(s); ps=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in ps]); rhs=t.sources@W; y=np.column_stack([decode(p['y_pulse_raw']) for p in ps]); full=m._pulse_losses(t,ps,decode)
params=[]; names=[]
for ax in 'xy':
 h=s['L'+ax]/(s['n'+ax]-1); params.append((h,np.asarray(s['kappa_'+ax])[[0,-1]],s['robin_'+ax])); names.extend([ax+'-',ax+'+'])
# Robin resistance in series with a signed effective exchange length; also check shared axis corrections.
def resistance(q):
 return np.concatenate([r/h/(1+q[i]*r*h/(2*k)) for i,(h,k,r) in enumerate(params)])
def plain(q): return t.nominal+np.repeat(q,2)
def antisym(q): return t.nominal+np.array([q[0]+q[2],q[0]-q[2],q[1]+q[3],q[1]-q[3]])
banks=s['row_bank_id']; Z=s['Z_nuisance']; shifts=np.repeat(np.arange(len(t.shifts)),len(t.sensors)); groups=[banks,shifts,np.tile(np.arange(len(t.sensors)),len(t.shifts))]; print('endpoint correction',dict(zip(names,(full-t.nominal).tolist())))
for label,model,num in [('nominal',lambda _:t.nominal,0),('series resistance',resistance,2),('axis shared',plain,2),('axis plus pair differences',antisym,4)]:
 def fit(tr,cols):
  if not num:return model([])
  return model(least_squares(lambda q:(t.response(model(q),rhs)[0]-y)[tr][:,cols].ravel(),np.zeros(num),bounds=(-1.,1.) if label=='series resistance' else (-3.,3.),gtol=1e-12).x)
 worst=[]
 for group in groups:
  vals=[]
  for g in np.unique(group):
   te=group==g; loss=fit(~te,slice(None)); vals.append(np.sqrt(np.mean((t.response(loss,rhs)[0]-y)[te]**2)))
  worst.append(max(vals))
 worst.append(max(np.sqrt(np.mean((t.response(fit(np.ones(len(y),bool),[1-i]),rhs)[0]-y)[:,i]**2)) for i in range(2)))
 print(label,'worst held-out bank/shift/sensor/pulse',np.array(worst))
raw=decode(s['y_obs_raw']); S,U,_=t.response(full,t.sources); w,_=m._source_fit(s,S,raw); before=raw-S@w; after=before-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@before)
for g in np.unique(banks):
 a=banks==g; print('bank',g,'residual RMS before/after nuisance',np.sqrt(np.mean(before[a]**2)),np.sqrt(np.mean(after[a]**2)))
