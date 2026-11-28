import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=a['dataset']; c=d['calibration_load_cells']; y=c['true_channel_loads']; r=c['bridge_raw']; b=m._bridge(c); print('bridge',b.round(6)); e=np.max(abs(m._bridge_terms(y)@b-r),axis=1); print('residual quantiles',np.quantile(e,[0,.25,.5,.6,.75,1])); good=e<.01; print('intact rows',sum(good),'calibration inverse max',np.max(abs(m._decode(r[good],b)-y[good])))
tr,te=d['train'],d['test']; n=len(tr['lab_id']); x=m._features(tr,te)[:n]; p,h=m._loads(tr,b); t=d['eval_times']; dt=d['past_dt']; z=tr['strain_curves_obs']-tr['strain_0'][:,None]; u=tr['instant_rate_obs'];
for split in (tr,te):
 for k in ('past_channel_raw','hold_channel_raw'):
  r=split[k]; v=m._decode(r,b); print('coupon range and max forward residual',v.min(),v.max(),np.max(abs(m._bridge_terms(v.reshape(-1,4))@b-r.reshape(-1,4))))
for lab in [-1,*np.unique(tr['lab_id'])]:
 ix=tr['lab_id']!=lab; ex=~ix if lab>=0 else ix; taus,theta=m._fit(x[ix],p[ix],h[ix],dt,t,z[ix],u[ix]); aa,v=m._design(taus,x[ex],p[ex],h[ex],dt,t); ec=abs(aa@theta-z[ex]); er=abs(v@theta-u[ex]); print('LAB',lab,'taus',taus,'theta',theta,'MAE',ec.mean(),er.mean(),'q90',np.quantile(ec,.9),np.quantile(er,.9),'horizonMAE',ec.mean(0).round(5))
