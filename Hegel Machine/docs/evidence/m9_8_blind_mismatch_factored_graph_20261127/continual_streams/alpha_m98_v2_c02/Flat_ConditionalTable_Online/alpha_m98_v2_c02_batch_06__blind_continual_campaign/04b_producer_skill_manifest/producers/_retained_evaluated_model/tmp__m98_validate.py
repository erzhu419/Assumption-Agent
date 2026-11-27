import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; a,b=d['train'],d['test']; c=d['calibration_load_cells']; dt,t=d['past_dt'],d['eval_times']; out=m.solve_batch(d)
assert set(out)=={'estimated_taus','hereditary_weights','predicted_strain','initial_rate_pred'}
for k,shape in zip(('estimated_taus','hereditary_weights','predicted_strain','initial_rate_pred'),((2,),(7,),(len(b['lab_id']),len(t)),(len(b['lab_id']),))):
 assert out[k].shape==shape and np.isfinite(out[k]).all()
z=out['estimated_taus']; assert .05<z[0]<z[1]<25
raw,true=c['bridge_raw'],c['true_channel_loads']; good=np.max(abs(raw),axis=1)<5; fold=np.arange(len(raw))%5; errs=[]
for f in range(5):
 tr=fold!=f; te=~tr&good; stress=m._stress_calibration({'bridge_raw':raw[tr],'true_channel_loads':true[tr]}); e=stress(raw[te])-true[te].sum(1); errs.append(np.sqrt(np.mean(e**2)))
print('Calibration inverse total-load OOF RMSE:',np.round(errs,7))
stress=m._stress_calibration(c); x,_=m._standardize(a,b); target=a['strain_curves_obs']-a['strain_0'][:,None]; rates=a['instant_rate_obs']; design=m._design(a,x,stress,dt,t)
print('LOLO benchtop: lab taus strain medianAE strain late medianAE rate RMSE')
for lab in np.unique(a['lab_id']):
 tr=a['lab_id']!=lab; te=~tr
 def subset(q): return tuple(v[tr] for v in design(q))
 taus,theta=m._fit(subset,target[tr],rates[tr]); ss,rr=design(taus); e=ss[te]@theta-target[te]; r=rr[te]@theta-rates[te]
 print(lab,taus,np.median(abs(e)),np.median(abs(e[:,-3:])),np.sqrt(np.mean(r*r)))
# Confirm the initial rate is the derivative of the deployed constitutive curve.
eps=1e-6; short=m.solve_batch(dict(d,eval_times=t))
_,xt=m._standardize(a,b); th=out['hereditary_weights']; hold=stress(b['hold_channel_raw']); fd=(hold*(-np.expm1(-eps/z)@th[:2])+(th[2]+xt@th[3:])*np.log1p(eps))/eps
print('Rate derivative max discrepancy:',np.max(abs(fd-short['initial_rate_pred'])))
assert np.max(abs(fd-short['initial_rate_pred']))<1e-5
