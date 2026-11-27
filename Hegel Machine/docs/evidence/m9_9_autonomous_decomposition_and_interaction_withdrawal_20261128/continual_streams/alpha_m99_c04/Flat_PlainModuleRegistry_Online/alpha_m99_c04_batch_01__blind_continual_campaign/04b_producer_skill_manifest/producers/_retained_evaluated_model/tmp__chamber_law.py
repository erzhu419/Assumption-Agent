import json, pickle, numpy as np
from scipy.optimize import least_squares
import prior as p
arc='/tmp/hegel_continual_archive/'
for row in json.load(open(arc+'archive_apparatus_log.json')):
 print('apparatus', {k:v for k,v in row.items() if k!='facility_telemetry_trace'}); a=np.array(row.get('facility_telemetry_trace',[])); print('telemetry mean/tail',np.round(a.mean(0),5),a[-1])
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; g=it['campaign_diagnostics']; tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; s=p._stress_map(d['calibration_load_cells']); tau,th=p._fit(tr,p._features(tr,te)[0],s,dt,t); h=s[0]+g['hold_channel_raw']@s[1:]; Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; labs=g['lab_id']; miss=np.isnan(g['coupon_features_raw']); x=g['coupon_features_raw'].copy();
for l in np.unique(labs):
 i=labs==l; a=g['ref_coupons'][i]; x[i]=(x[i]-np.median(a.mean(0)))/np.median(a.std(0))
_,cell,cnt=np.unique(np.c_[labs,g['stratum']],axis=0,return_inverse=True,return_counts=True); cell=cell.ravel(); obs=~miss
X={}
for name,w in [('plain',np.ones(len(x))),('cell',1/cnt[cell]),('sqrtcell',1/np.sqrt(cnt[cell]))]:
 fill=np.nansum(x*w[:,None],0)/(obs*w[:,None]).sum(0); print('fill',name,fill); X[name]=np.where(obs,x,fill)
for name,fill in [('median',np.nanmedian(x,0)),('zero',np.zeros(4))]:
 print('fill',name,fill); X[name]=np.where(obs,x,fill)
K=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); E=Y-(h[:,None]*th[:2])@K.T
L=lambda q:np.r_[np.log1p(t/q),1/q]
for name,x2 in X.items():
 r=th[2]+x2@th[3:]; print('feature model',name)
 for kind in ['fixed log','free log','log+intercept','log+power']:
  def model(b,i):
   if kind=='log+power':
    q=np.exp(b[1]); pow=b[2]; v=np.log1p(t/q); F=np.r_[np.expm1(pow*v)/pow if abs(pow)>1e-9 else v,1/q]
   else: F=L(np.exp(b[1]) if kind!='fixed log' else 1.)
   return np.outer(b[0]*r[i]+(b[2] if kind=='log+intercept' else 0),F)
  start={'fixed log':[1.], 'free log':[1.,0.], 'log+intercept':[1.,0.,0.], 'log+power':[1.,0.,0.]}[kind]
  def fit(i):return least_squares(lambda b:(model(b,i)-E[i]).ravel(),start).x
  b=fit(slice(None)); errs=[]
  for lab in np.unique(labs):
   i=labs==lab; er=model(fit(~i),i)-E[i]; errs.append(np.sqrt(np.r_[np.mean(er[:,:-1]**2),np.mean(er[:,-1]**2)]))
  er=model(b,slice(None))-E; print(kind,np.round(b,6),'full complete/missing',np.round([np.mean(er[obs.all(1)]**2)**.5,np.mean(er[miss.any(1)]**2)**.5],5),'worst lab curve/rate',np.round(np.max(errs,0),5))
