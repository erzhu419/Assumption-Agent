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
from scipy.optimize import least_squares
D=theta[2]+xs['local']@theta[3:]
B=(h[:,None]-z)*theta[:2]/taus
V=np.c_[B,D]
print('rate immutability',np.c_[r-V.sum(1),g['lab_id'],np.isnan(g['coupon_features_raw']).sum(1)].round(4))
phi=lambda u: -u[:,None]*np.expm1(-t[None,:]/u[:,None])
# Each source can have its own saturating response, plus a conserved linear remainder.
for mode in range(7):
 def pred(q):
  if mode==0: return V@phi(np.exp(q[:3]))
  if mode==1: return (V*q[3:])@phi(np.exp(q[:3]))+(V*(1-q[3:])).sum(1)[:,None]*t
  if mode==2: return B@phi(np.exp(q[:2]))+D[:,None]*(q[3]*phi(np.exp(q[2:3]))+(1-q[3])*t)
  if mode==3: return B@phi(np.exp(q[:2]))+D[:,None]*(q[3]*phi(np.exp(q[2:3]))+(1-q[3])*phi(np.exp(q[4:5])))
  if mode==4: return V@phi(taus[[0,1,1]])+(V*q).sum(1)[:,None]*(t-phi(taus[1:]))
  if mode==5: return (B@q[2:6].reshape(2,2)+D[:,None]*q[6:8])@phi(np.exp(q[:2]))+((B@(1-q[2:6].reshape(2,2).sum(1)))+D*(1-q[6:8].sum()))[:,None]*t
  if mode==6: return B@phi(taus*np.exp(q[:2]))+D[:,None]*(q[3]*phi(np.exp(q[2:3]))+(1-q[3])*t)
 p0=[np.log([.55,2,1]),np.r_[np.log([.55,2,1]),[.99,.99,.8]],np.r_[np.log([.55,2,1]),.85],np.r_[np.log([.55,2,1]),.85,np.log(8.)],[0,0,0],np.r_[np.log([.55,2]),1,0,0,1,.25,.65],np.r_[[0,0],np.log(1),.85]][mode]
 def fit(i): return least_squares(lambda q: (pred(q)[i]-y[i]).ravel(),p0,loss='cauchy',f_scale=.009,max_nfev=400).x
 q=fit(np.ones(len(y),bool)); yh=pred(q); print('mode',mode,'par',q.round(5),'RMSE,med',rm(yh-y),med(yh-y),'time med',np.median(y-yh,axis=0).round(4)); print('LOLO',[(round(rm((yh1:=pred(fit(g['lab_id']!=lab)))[g['lab_id']==lab]-y[g['lab_id']==lab]),4),round(med(yh1[g['lab_id']==lab]-y[g['lab_id']==lab]),4)) for lab in range(4)])
print('test vs diag inputs');
for v in (tr,te,g):
 print('ranges hold past',np.quantile(v['hold_channel_raw'].sum(-1),[0,.5,1]),np.quantile(v['past_channel_raw'].sum(-1),[0,.5,1]),'tele corr' if 'aux_chamber_telemetry' in v else '')
