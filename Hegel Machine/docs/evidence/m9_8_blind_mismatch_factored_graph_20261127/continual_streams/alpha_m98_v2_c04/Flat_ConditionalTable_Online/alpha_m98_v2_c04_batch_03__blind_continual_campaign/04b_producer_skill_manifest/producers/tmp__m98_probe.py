import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te=d['train'],d['test']; cd=i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; b=m._bridge(d['calibration_load_cells']); xt=m._features(tr,te); n=len(tr['lab_id']); tau,th=m._fit(xt[:n],*m._loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th)
p,h=m._loads(cd,b); y=cd['strain_curves_obs']-cd['strain_0'][:,None]; r=cd['instant_rate_obs']; Y=np.c_[y,r]; print('chamber first row',Y[0].round(5)); print('chamber ranges',np.ptp(Y,axis=0).round(5),'svd',np.linalg.svd(Y,compute_uv=False).round(5)); print('diagnostic labs',np.unique(cd['lab_id'],return_counts=True),'missing',np.isnan(cd['coupon_features_raw']).sum(0))
merged={k:np.concatenate((te[k],cd[k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; xd={'train+diag':m._features(tr,cd)[n:],'all':m._features(tr,merged)[len(xt):]}
labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); x=cd['coupon_features_raw'].copy()
for lab in np.unique(cd['lab_id']):
 q=refs[labs==lab]; mask=cd['lab_id']==lab; x[mask]=(x[mask]-np.nanmedian(np.nanmean(q,0)))/np.nanmedian(np.nanstd(q,0))
_,ix,cnt=np.unique(np.c_[labs,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); w=1/cnt[ix,None]; fill=(xt*w*(~np.isnan(raw))).sum(0)/(w*(~np.isnan(raw))).sum(0); xd['train+test norm']=np.where(np.isnan(x),fill,x)
for k,X in xd.items():
 if not np.isfinite(X).all(): continue
 a,z=m._design(X,p,h,dt,t,tau); print('baseline',k,'median abs',np.median(np.abs((a@th).reshape(y.shape)-y)),np.median(np.abs(z@th-r)))
def timebasis(q,ns):
 u=np.exp(q); V=np.c_[-np.expm1(-t[:,None]/u),t] if ns else t[:,None]; return np.vstack((V,np.r_[1/u,1.]))
for ns in (0,1,2,3):
 def fun(q):
  V=timebasis(q,ns); return (np.linalg.lstsq(V,Y.T,rcond=None)[0].T@V.T-Y).ravel()
 bounds=(np.full(ns,np.log(.025)),np.full(ns,np.log(35.)))
 if ns: fit=min((least_squares(fun,np.log(q),bounds=bounds,loss='soft_l1',f_scale=.001) for q in [np.geomspace(a,b,ns) for a,b in [(.13,6),(.5,2.5),(.9,12)]]),key=lambda x:x.cost); q=fit.x
 else: q=np.array([])
 V=timebasis(q,ns); co=np.linalg.lstsq(V,Y.T,rcond=None)[0].T; print('time fit',ns,np.exp(q), 'median abs',np.median(np.abs(fun(q))),'rms',np.mean(fun(q)**2)**.5)
 if ns != 2: continue
 for k,X in xd.items():
  if not np.isfinite(X).all(): continue
  dr=th[2]+X@th[3:]; gaps=h[:,None]-p@(-np.expm1(-dt[:,None]/tau)*np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0.][:,None]/tau));
  for label,F in [('hold/drift',np.c_[h,dr]),('hold/gaps/drift',np.c_[h,gaps,dr]),('hold/full past/drift',np.c_[h,p,dr]),('hold/full past/features',np.c_[np.ones(len(h)),h,p,X])]:
   c=np.linalg.lstsq(F,co,rcond=None)[0]; resid=F@c@V.T-Y; print(k,label,'med',np.median(abs(resid)).round(6),'rms per output',np.sqrt(np.mean(resid[:,:-1]**2)).round(6),np.sqrt(np.mean(resid[:,-1]**2)).round(6),'coefs',c.T.round(4).tolist() if label=='hold/gaps/drift' else '')
