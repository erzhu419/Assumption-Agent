import sys, json, pickle, runpy
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
old=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c04_batch_01_solver.py')
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=a['dataset']; g0=a['campaign_diagnostics']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
d=calibrate_upstream(raw,pay); g=calibrate_upstream(dict(raw,test=g0),pay)['test']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; sm=np.r_[0.,np.ones(4)]; feat=old['_features']; X,Z=feat(tr,te); taus,theta=old['_fit'](tr,X,sm,dt,t)
cat=lambda a,b:{k:np.concatenate([a[k],b[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
xs={'local':feat(g,{k:v[:0] for k,v in g.items()})[0], 'train+diag':feat(tr,g)[1], 'all+diag':feat(cat(tr,te),g)[1]}
for f in ['labs','strata']:
 k={'labs':'lab_id','strata':'stratum'}[f]; print(f,[np.unique(q[k],return_counts=True)[1] for q in (tr,te,g)])
cal=raw['calibration_load_cells']; br,tl=cal['bridge_raw'],cal['true_channel_loads']; ii=(np.abs(br)<=8.5).all(1); print('calibration affine',np.linalg.lstsq(np.c_[tl[ii],np.ones(ii.sum())],br[ii],rcond=None)[0].round(6)); print('taus,theta',taus,theta)
y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; h=g['hold_channel_raw'].sum(-1); p=g['past_channel_raw'].sum(-1); age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; z=p@(-np.expm1(-dt[:,None]/taus)*np.exp(-age[:,None]/taus)); L=h*np.sum(theta[:2]/taus); H=-z*theta[:2]/taus
rm=lambda x:np.sqrt(np.mean(x*x)); med=lambda x:np.median(np.abs(x));
print('signal units',rm(y),rm(r), 'past/hold',np.c_[h,p][:4].round(3)); print('reference nan counts',[np.isnan(q['coupon_features_raw']).sum() for q in (tr,te,g)])
import downstream_solver as dn
x=xs['local']; yc,rc=dn._chamber_response(g,x,dt,t,taus,theta); ec=yc-y-g['strain_0'][:,None]; er=rc-r
for field in ('lab_id','stratum'):
 for v in np.unique(g[field]):
  i=g[field]==v; print(field,v,'curve/rate RMSE',rm(ec[i]),rm(er[i]))
for key,i in [('present',np.isfinite(g['coupon_features_raw']).all(1)),('missing',np.isnan(g['coupon_features_raw']).any(1))]:
 print(key,'curve/rate RMSE',rm(ec[i]),rm(er[i]));
 if key=='missing':
  for j in np.flatnonzero(i):
   k=np.flatnonzero(np.isnan(g['coupon_features_raw'][j]))[0]; print('row/col/lab/stratum/current/inferred',j,k,g['lab_id'][j],g['stratum'][j],x[j,k],x[j,k]-er[j]/theta[3+k])
print('hour slices',np.sqrt(np.mean(ec**2,axis=0)).round(5)); print('prediction shape',dn.solve_downstream(d)['predicted_strain'].shape)
