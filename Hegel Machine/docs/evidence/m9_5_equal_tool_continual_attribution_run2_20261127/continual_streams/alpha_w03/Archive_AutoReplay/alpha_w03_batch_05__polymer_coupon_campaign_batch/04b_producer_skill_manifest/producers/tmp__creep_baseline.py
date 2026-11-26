import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/alpha_w03_batch_04_solver.py'
s=importlib.util.spec_from_file_location('baseline',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w03_batch_05.pkl','rb'))[0]['dataset'];r=d['train'];dt,t,sy=[d[k] for k in ('past_dt','eval_times','yield_stress')]
x,h,p=m._prep(d);n=len(r['lab_id']); x,h,p=x[:n],h[:n],p[:n]; y=r['strain_curves_obs']-r['strain_0'][:,None];z=r['instant_rate_obs']; complete=np.isfinite(r['coupon_features_raw']).all(1); labs=r['lab_id']; strata=r['stratum']; print('history length',dt.sum(),'calibration',m._calibrate(d['calibration_load_cells'],labs),'physical stress max',max(np.abs(h).max(),np.abs(p).max()),'strata',np.unique(strata,return_counts=True)); yc=np.empty_like(y);zc=np.empty_like(z)
for l in range(4):
 i=(labs!=l)&complete; v=labs==l
 tau,b=m._fit(x[i],h[i],p[i],dt,t,sy,y[i],z[i]);c,a=m._design(tau,x[v],h[v],p[v],dt,t,sy);yc[v]=c@b;zc[v]=a@b
 print('held lab',l,'taus',np.round(tau,6),'coef',np.round(b,6))
for l in range(4):
 for mask in (complete,~complete):
  i=(labs==l)&mask; er=yc[i]-y[i]; ez=zc[i]-z[i]; corrected=er-ez[:,None]*t
  print('slice',l,'complete',bool(mask is complete),'n',i.sum(),'curve RMSE',np.sqrt(np.mean(er**2)),'rate RMSE',np.sqrt(np.mean(ez**2)),'rate-anchored curve RMSE',np.sqrt(np.mean(corrected**2)),'curve medians',np.round(np.median(er,axis=0)[[0,-1]],5))
for j in range(2):
 er=yc-y;ez=zc-z; i=complete; e=er[i] if j==0 else ez[i]; print('complete quantiles',np.quantile(np.abs(e),[.25,.5,.9,.99,1.]))
