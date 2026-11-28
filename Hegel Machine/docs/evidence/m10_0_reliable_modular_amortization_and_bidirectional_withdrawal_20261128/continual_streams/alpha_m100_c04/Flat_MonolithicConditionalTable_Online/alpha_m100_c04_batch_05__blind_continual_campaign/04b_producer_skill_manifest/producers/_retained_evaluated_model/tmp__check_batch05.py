import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; c=d['calibration_load_cells']; q=c['true_channel_loads']; r=c['bridge_raw']; good=np.max(abs(r),axis=1)<8; axis=(q!=0).sum(1)==1
raw_e=np.zeros_like(r); load_e=np.zeros_like(q)
for k in range(5):
 test=good & (np.arange(len(r))%5==k); train=good & ~test
 p=s._calibrate({'true_channel_loads':q[train],'bridge_raw':r[train]}); raw_e[test]=s._bridge_raw(q[test],p)-r[test]; load_e[test]=s._bridge_load(r[test],p)-q[test]
for mask in (good & axis,good & ~axis):
 print('sweep axis/mixed CV raw RMS/max, total-load RMS/max',np.sqrt(np.mean(raw_e[mask]**2)),np.max(abs(raw_e[mask])),np.sqrt(np.mean(load_e[mask].sum(1)**2)),np.max(abs(load_e[mask].sum(1))))
assert np.max(abs(raw_e[good]))<.001 and np.max(abs(load_e[good].sum(1)))<.002
cal=s._calibrate(c)
for part in (tr,te):
 for key in ('past_channel_raw','hold_channel_raw'):
  raw=part[key]; load=s._bridge_load(raw,cal); err=np.max(abs(s._bridge_raw(load,cal)-raw)); print(key,'max inverse residual',err,'loads',load.min(),load.max()); assert err<1e-9
out=s.solve_batch(d); print('material',out['estimated_taus'],out['hereditary_weights']); assert .05<out['estimated_taus'][0]<out['estimated_taus'][1]<25
x=s._features(tr,te); inp=s._inputs(tr,x[:len(tr['lab_id'])],cal); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; ec=np.zeros_like(y); er=np.zeros_like(rate)
for l in np.unique(tr['lab_id']):
 m=tr['lab_id']==l; sub=lambda mask: tuple(z[mask] for z in inp)
 tau,theta=s._fit(sub(~m),dt,t,y[~m],rate[~m]); a,b=s._design(tau,sub(m),dt,t); ec[m]=a@theta-y[m]; er[m]=b@theta-rate[m]; print('held lab',l,'taus',tau,'curve/rate MAE',np.mean(abs(ec[m])),np.mean(abs(er[m])))
for k in np.unique(tr['stratum']):
 m=tr['stratum']==k; print('OOF stratum',k,'curve/rate p90AE',np.quantile(abs(ec[m]),.9),np.quantile(abs(er[m]),.9))
