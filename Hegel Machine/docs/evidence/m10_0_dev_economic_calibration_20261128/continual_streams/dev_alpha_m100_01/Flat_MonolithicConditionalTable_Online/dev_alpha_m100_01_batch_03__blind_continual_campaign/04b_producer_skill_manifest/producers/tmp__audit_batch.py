import importlib.util,pickle,numpy as np
p='/holdout_task/batch_solver.py'
s=importlib.util.spec_from_file_location('batch_solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; raw=c['bridge_raw']; truth=c['true_channel_loads']
b=m._bridge(c); corrupt=np.max(abs(m._transfer(truth,b)-raw),axis=1)>.1
print('corrupted calibration rows:',np.flatnonzero(corrupt))
for fold in np.array_split(np.arange(len(raw)),5):
 mask=~np.isin(np.arange(len(raw)),fold); sub=dict(c, bridge_raw=raw[mask], true_channel_loads=truth[mask]); coeff=m._bridge(sub); ii=fold[~corrupt[fold]]; e=m._invert(raw[ii],coeff)-truth[ii]; print('bridge blocked fold clean axial RMSE/max',np.sqrt(np.mean(e.sum(1)**2)),max(abs(e.sum(1))))
for k in ['train','test']:
 for field in ['past_channel_raw','hold_channel_raw']:
  a=d[k][field]; print('inverse closure',k,field,np.max(abs(m._transfer(m._invert(a,b),b)-a)))
tr,te=d['train'],d['test']; x=m._features(tr,te)[:len(tr['lab_id'])]; dt,t=d['past_dt'],d['eval_times']; past,hold=m._loads(tr,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; out=m.solve_batch(d)
print('taus',out['estimated_taus'],'weights',out['hereditary_weights'])
for lab in np.unique(tr['lab_id']):
 w=tr['lab_id']!=lab; q=~w
 tau,th=m._fit(past[w],hold[w],x[w],dt,t,y[w],r[w]); a,z=m._design(tau,past[q],hold[q],x[q],dt,t); e=a@th-y[q]; g=z@th-r[q]; print('LOLO',lab,'curve/rate abs error median',np.median(abs(e)),np.median(abs(g)),'q90',np.quantile(abs(e),.9),np.quantile(abs(g),.9),'taus',tau)
tau,th=out['estimated_taus'],out['hereditary_weights']; a,z=m._design(tau,*m._loads(te,b),m._features(tr,te)[len(x):],dt,np.array([0.,1e-6])); v=a@th; print('deployment zero-time closure',max(abs(v[:,0])),'max derivative mismatch',max(abs((v[:,1]-v[:,0])/1e-6-out['initial_rate_pred']))); assert np.max(abs(z@th-out['initial_rate_pred']))<1e-12 and 0.05<tau[0]<tau[1]<25.
