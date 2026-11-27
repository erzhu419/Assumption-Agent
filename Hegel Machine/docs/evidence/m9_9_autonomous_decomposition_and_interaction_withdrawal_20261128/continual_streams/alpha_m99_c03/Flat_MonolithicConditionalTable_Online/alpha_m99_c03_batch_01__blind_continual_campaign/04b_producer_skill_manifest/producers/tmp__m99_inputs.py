import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c03_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; g=item['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; cal=d['calibration_load_cells']; bridge=m._calibration(cal)
a=cal['bridge_raw']; b=cal['true_channel_loads']; aa=np.c_[np.ones(len(a)),a]; au,bu=np.triu_indices(4); ap=np.c_[aa,a[:,au]*a[:,bu]]
print('times',t,'dt',dt)
for v in (aa,ap):
 p=np.arange(len(a))%4!=0; resid=v[~p]@np.linalg.lstsq(v[p],b[p],rcond=None)[0]-b[~p]; print('bridge design/rmse channel and total',v.shape[1],np.mean(resid**2,axis=0)**.5,np.mean(resid.sum(1)**2)**.5)
x=m._features(tr,te); c=m._coupon_inputs(tr,x[:len(tr['lab_id'])],bridge); out=m.solve_batch(d); tau=out['estimated_taus']; th=out['hereditary_weights']; print('tau theta',tau,th)
xx=m._features(tr,{k:np.concatenate((te[k],g[k]),axis=0) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')})[-len(g['lab_id']):]; cg=m._coupon_inputs(g,xx,bridge)
E=np.c_[-np.expm1(-t[:,None]/tau),t]; er=np.r_[1/tau,1]
y=g['strain_curves_obs']-g['strain_0'][:,None]; yr=g['instant_rate_obs']; A0,R0=m._design(tau,cg,dt,t)
