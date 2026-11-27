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
for name,x in xs.items():
 D=theta[2]+x@theta[3:]; c0,r0=old['_chamber_response'](g,x,sm,t,taus,theta); cb,rb=old['_design'](g,x,sm,dt,t,taus); print(name,'archived curve/rate RMSE',rm(y-c0),rm(r-r0),'bench',rm(y-cb@theta),rm(r-rb@theta))
 designs={'L D':np.c_[L,D],'L D H':np.c_[L,D,H],'L D H 1':np.c_[L,D,H,np.ones(len(g['lab_id']))],'L D H x':np.c_[L,H,np.ones(len(g['lab_id'])),x],'L D p':np.c_[L,D,p]}
 for label,A in designs.items():
  b=np.linalg.lstsq(A,r,rcond=None)[0]; print(' rate',label,'rm/med',round(rm(r-A@b),5),round(med(r-A@b),5),'coef',b.round(4))
  if label not in ('L D','L D H','L D H x'):continue
  for drift in [False,True]:
   def basis(poles):
    a=-np.expm1(-t[None,:]/poles[:,None])*poles[:,None]; return np.vstack([a,t]) if drift else a
   n=A.shape[1]; m=2+drift; ini=np.linalg.lstsq(np.einsum('ij,kt->itjk',A,basis(taus)).reshape(-1,n*m),y.ravel(),rcond=None)[0]
   def pred(v):
    f=A@v[2:].reshape(n,m); return f@basis(np.exp(v[:2])),f.sum(1)
   def res(v):
    cy,cr=pred(v); return np.r_[(cy-y).ravel(),cr-r]
   fit=least_squares(res,np.r_[np.log(taus),ini],bounds=(np.r_[np.log([.025,.025]),np.repeat(-np.inf,n*m)],np.r_[np.log([50.,50.]),np.repeat(np.inf,n*m)]),loss='soft_l1',f_scale=.005,x_scale='jac',max_nfev=150)
   cy,cr=pred(fit.x); print(' curve',label,'drift',drift,'poles',np.exp(fit.x[:2]).round(4),'rm/med',np.round([rm(y-cy),med(y-cy),rm(r-cr),med(r-cr)],5),'coef',fit.x[2:].reshape(n,m).round(4).tolist())
