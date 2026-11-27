import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'batch_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(p+'public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test'];t=d['eval_times']; dt=d['past_dt']; c=d['calibration_load_cells']; raw=c['bridge_raw'];loads=c['true_channel_loads'];stress=m._stress_calibration(c)
print('calibration total residual quantiles',np.quantile(np.abs(stress(raw)-loads.sum(1)),[.5,.9,1]))
linear=np.c_[np.ones(len(raw)),raw]; quad=np.c_[linear,*[raw[:,i]*raw[:,j] for i in range(4) for j in range(i,4)]]
for a in [linear,quad]:
 target=loads.sum(1); e=[]
 for idx in np.array_split(np.arange(len(raw)),5):
  keep=np.setdiff1d(np.arange(len(raw)),idx);e.extend(a[idx]@np.linalg.lstsq(a[keep],target[keep],rcond=None)[0]-target[idx])
 print('calibration OOF columns, RMSE, max abs',a.shape[1],np.sqrt(np.mean(np.square(e))),np.max(np.abs(e)))
print('public estimated taus and weights',*(m.solve_batch(d)[k] for k in ['estimated_taus','hereditary_weights'])); print('eval times',t)
x,_=m._standardize(tr,te)
for group in np.unique(tr['lab_id']):
 mask=tr['lab_id']==group; fit=np.flatnonzero(~mask); hold=np.flatnonzero(mask)
 def sl(ix):return {k:v[ix] for k,v in tr.items()}
 taus,theta=m._fit(m._design(sl(fit),x[fit],stress,dt,t),tr['strain_curves_obs'][fit]-tr['strain_0'][fit,None],tr['instant_rate_obs'][fit]);a,b=m._design(sl(hold),x[hold],stress,dt,t)(taus)
 ec=np.abs(a@theta+tr['strain_0'][hold,None]-tr['strain_curves_obs'][hold]); er=np.abs(b@theta-tr['instant_rate_obs'][hold]); print('held lab',group,'taus',taus,'curve median/q90',np.quantile(ec,[.5,.9]),'rate median/q90',np.quantile(er,[.5,.9]))
