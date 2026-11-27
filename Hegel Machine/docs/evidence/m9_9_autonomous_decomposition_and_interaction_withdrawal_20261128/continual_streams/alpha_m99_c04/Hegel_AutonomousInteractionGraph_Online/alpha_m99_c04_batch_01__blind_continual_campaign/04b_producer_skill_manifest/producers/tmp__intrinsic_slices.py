import pickle,runpy,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/batch_solver.py'); i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; s=m['_stress_map'](ds['calibration_load_cells']); x,xt=m['_features'](tr,te)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; q,b=m['_fit'](tr,x,s,dt,t); print('intrinsic',q,b)
mask=np.isfinite(tr['coupon_features_raw']).all(1)
for loss in ['soft_l1','cauchy']:
 for scale in [.8,1.5,2.5,6]:
  for complete in (False,True):
   use=mask if complete else np.ones(len(x),bool)
   def res(p):
    A,R=m['_design'](tr,x,s,dt,t,np.exp(p[:2])); return np.r_[((A@p[2:]-y)[use]/.012).ravel(),(R@p[2:]-yr)[use]/.007]
   f=least_squares(res,np.r_[np.log(q),b],loss=loss,f_scale=scale,x_scale='jac'); print(loss,scale,complete,np.round(np.r_[np.exp(f.x[:2]),f.x[2:]],5))
xd=m['_features'](d,{k:v[:0] for k,v in d.items()})[0]; u,v=m['_chamber_response'](d,xd,s,t,q,b); err=u-(d['strain_curves_obs']-d['strain_0'][:,None]); rr=v-d['instant_rate_obs']; print('complete/missing chamber absolute RMSE',[(np.sqrt(np.mean(err[z]**2)),np.sqrt(np.mean(rr[z]**2))) for z in [np.isfinite(d['coupon_features_raw']).all(1),np.isnan(d['coupon_features_raw']).any(1)]])
