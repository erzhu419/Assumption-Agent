import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; l=c['true_channel_loads']; r=c['bridge_raw']; mixed=(l!=0).sum(1)>1
errors=[]
for i in range(len(l)):
 mask=np.arange(len(l))!=i
 f=m._stress_calibration({'true_channel_loads':l[mask],'bridge_raw':r[mask]})
 if np.linalg.norm(r[i])<10:
  errors.append((i,abs(f(r[i:i+1])[0]-l[i].sum())))
e=np.array(errors); print('bridge held-out n, RMSE, worst error and row',len(e),np.sqrt(np.mean(e[:,1]**2)),e[e[:,1].argmax()]); assert e[:,1].max()<.002
train,test=d['train'],d['test']; x,_=m._standardize(train,test); stress=m._stress_calibration(c); design=m._design(train,x,stress,d['past_dt'],d['eval_times']); s=train['strain_curves_obs']-train['strain_0'][:,None]; r=train['instant_rate_obs']; labs=train['lab_id'];
for lab in [None,*np.unique(labs)]:
 fit=np.ones(len(labs),bool) if lab is None else labs!=lab; ev=~fit if lab is not None else fit
 def matrices(tau): return tuple(a[fit] for a in design(tau))
 tau,theta=m._fit(matrices,s[fit],r[fit]); S,R=design(tau); se=S[ev]@theta-s[ev]; re=R[ev]@theta-r[ev]; print('held lab',lab,'taus',tau.round(5),'weights',theta.round(5),'strain median/p90/max-horizon median',np.round([np.median(abs(se)),np.quantile(abs(se),.9),np.max(np.median(abs(se),axis=0))],6),'rate median/p90',np.round([np.median(abs(re)),np.quantile(abs(re),.9)],6)); assert np.isfinite(theta).all() and .05<tau[0]<tau[1]<25
