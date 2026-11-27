import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_05_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];g=i['campaign_diagnostics'];tr=d['train'];te=d['test'];dt,t=d['past_dt'],d['eval_times'];c=d['calibration_load_cells'];raw=c['bridge_raw'];target=c['true_channel_loads'];cal=m._stress_calibration(c)
print('calibration ranges',raw.min(0),raw.max(0),target.min(0),target.max(0)); er=raw@cal[:-1]+cal[-1]-target.sum(1);print('affine total RMSE/max',np.sqrt(np.mean(er**2)),max(abs(er)),'coef',cal)
for name,q in [('train',tr),('test',te),('diag',g)]:
 print(name,'lab counts',np.unique(q['lab_id'],return_counts=True),'missing',np.isnan(q['coupon_features_raw']).sum(0),'raw min/max',q['hold_channel_raw'].min(0),q['hold_channel_raw'].max(0)); print('raw past/hold stress ranges',*[np.round(np.quantile(q[k]@cal[:-1]+cal[-1],[0,.1,.5,.9,1]),4) for k in ('past_channel_raw','hold_channel_raw')]);print('lab raw/ref avg',[(j,np.round(np.nanmean(q['coupon_features_raw'][q['lab_id']==j],0),3),round(np.nanmean(q['ref_coupons'][q['lab_id']==j]),3)) for j in np.unique(q['lab_id'])])
ct,ce=m._prepare(d); obs=lambda q:(q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']); tau,theta=m._fit(dt,t,ct,*obs(tr)); growth=-np.expm1(-t[:,None]/tau); A=np.c_[growth,t];
for name,q in [('train',tr),('diag',g)]:
 y,r=obs(q);b=np.linalg.lstsq(A,y.T,rcond=None)[0].T; print(name,'free temporal RMSE per coupon quantiles',np.quantile(np.sqrt(np.mean((y-b@A.T)**2,axis=1)),[0,.25,.5,.75,1])); print(name,'rates extrapolation residual quantiles',np.quantile(r-b@np.r_[1/tau,1],[0,.1,.25,.5,.75,.9,1]));
C,R=m._design(tau,dt,t,ct);y,r=obs(tr); e=y-C@theta;f=r-R@theta;print('train residual structure: time med, mad and rate by lab');print(np.median(e,0));print(np.median(abs(e),0));
for lab in np.unique(tr['lab_id']):
 j=tr['lab_id']==lab; print(lab, 'curve',np.median(abs(e[j])),np.median(e[j]),'rate',np.median(abs(f[j])),np.median(f[j]),'rate residual examples',np.round(f[j][:8],3));
for a in (tr,g):
 print('strain0 and obs counts/values',np.quantile(a['strain_0'],[0,.5,1]),np.round(a['strain_curves_obs'][:2],3),np.round(a['instant_rate_obs'][:8],3))
