import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c03_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; g=item['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; cal=d['calibration_load_cells']; bridge=m._calibration(cal)
a=cal['bridge_raw']; b=cal['true_channel_loads']; aa=np.c_[np.ones(len(a)),a]; au,bu=np.triu_indices(4); ap=np.c_[aa,a[:,au]*a[:,bu]]
print('times',t,'dt',dt)
for v in (aa,ap):
 p=np.arange(len(a))%4!=0; resid=v[~p]@np.linalg.lstsq(v[p],b[p],rcond=None)[0]-b[~p]; print('bridge design/rmse channel and total',v.shape[1],np.mean(resid**2,axis=0)**.5,np.mean(resid.sum(1)**2)**.5)
x=m._features(tr,te); c=m._coupon_inputs(tr,x[:len(tr['lab_id'])],bridge); out=m.solve_batch(d); tau=out['estimated_taus']; th=out['hereditary_weights']; print('tau theta',tau,th)
xx=m._features(tr,{k:np.concatenate((te[k],g[k]),axis=0) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')})[-len(g['lab_id']):]; cg=m._coupon_inputs(g,xx,bridge)
for label,h,c0 in [('train',tr,c),('diag',g,cg)]:
 cr,ra=m._design(tau,c0,dt,t); y=h['strain_curves_obs']-h['strain_0'][:,None]; yr=h['instant_rate_obs']; E=np.c_[ -np.expm1(-t[:,None]/tau),t]; er=np.r_[1/tau,1]; B=np.linalg.lstsq(np.vstack((E,er)),np.c_[y,yr].T,rcond=None)[0].T; nominal=np.column_stack((ra[:,:2]*tau,ra[:,2:]@th[2:])); nominal[:,:2]*=th[:2]
 for name,r in [('baseline curve',y-cr@th),('baseline rate',yr-ra@th),('free curve',y-B@E.T),('free rate',yr-B@er),('coef delta',B-nominal)]:
  print(label,name,'rmse', np.sqrt(np.mean(r*r,axis=0)).round(5),'abs quantile',np.quantile(abs(r),[.25,.5,.8,.9,.99]).round(5))
 if label=='diag':
  print('lab stratum past_last hold feature1 nominal3 inferred3'); print(np.c_[h['lab_id'],h['stratum'],c0[0][:,-1],c0[1],c0[2][:,1],nominal,B].round(4)); print('diag keys',list(g)); print('dataset keys',list(d))
