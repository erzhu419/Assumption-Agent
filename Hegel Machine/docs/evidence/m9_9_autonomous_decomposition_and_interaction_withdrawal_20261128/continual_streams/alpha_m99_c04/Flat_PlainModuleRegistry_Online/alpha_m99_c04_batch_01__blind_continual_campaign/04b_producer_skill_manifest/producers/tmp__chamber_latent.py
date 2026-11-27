import pickle, numpy as np
from scipy.optimize import least_squares
import prior as p
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr,te=d['train'],d['test']; g=item['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; s=p._stress_map(d['calibration_load_cells']); tau,th=p._fit(tr,p._features(tr,te)[0],s,dt,t)
keys=['lab_id','stratum','ref_coupons','coupon_features_raw']; comb=lambda a,b:{k:np.concatenate([a[k],b[k]]) for k in keys}; empty={k:g[k][:0] for k in keys}
labs=g['lab_id']; raw=g['coupon_features_raw']; missing=np.isnan(raw).any(1); h=s[0]+g['hold_channel_raw']@s[1:]; past=s[0]+g['past_channel_raw']@s[1:]; Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]
A=np.r_[np.c_[-np.expm1(-t[:,None]/tau),t], [np.r_[1/tau,1.]]]; E=Y-(h[:,None]*th[None,:2])@A[:,:2].T
u,sv,v=np.linalg.svd(E-E.mean(0),full_matrices=False); latent=u[:,0]*sv[0]; latent*=np.sign(np.corrcoef(latent,th[2]+p._features(tr,g)[1]@th[3:])[0,1]); print('centered remainder singular values',np.round(sv,5)); print('remainder temporal coefficient (mean + latent)',np.round(np.linalg.lstsq(A,np.vstack([E.mean(0),latent@E/(latent@latent)]).T,rcond=None)[0],5)); print('remainder rank-1 RMSE',np.sqrt(np.mean(sv[1:]**2/E.size)*len(sv[1:])))
xs={name:p._features(b,g)[1] for name,b in [('diag only',empty),('train+diag',tr),('test+diag',te),('all',comb(tr,te))]}
x0=p._features(tr,te)[0]; pool=comb(tr,te); xfix=raw.copy()
for lab in np.unique(labs):
 i=labs==lab; ref=pool['ref_coupons'][pool['lab_id']==lab]; xfix[i]=(xfix[i]-np.median(ref.mean(0)))/np.median(ref.std(0))
xall=np.vstack(p._features(tr,te)); _,idx,cnt=np.unique(np.c_[pool['lab_id'],pool['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/cnt.reshape(-1)[idx.reshape(-1)]; obs=np.isfinite(pool['coupon_features_raw']); fill=(xall*obs*w[:,None]).sum(0)/(obs*w[:,None]).sum(0); xs['train+test fixed']=np.where(np.isfinite(xfix),xfix,fill)
for name,x in xs.items():
 r=th[2]+x@th[3:]; err=np.zeros_like(E)
 for lab in np.unique(labs):
  i=labs!=lab; X=np.c_[np.ones(len(r)),r]; err[~i]=X[~i]@np.linalg.lstsq(X[i],E[i],rcond=None)[0]-E[~i]
 print(name,'LOLO y/r',np.round([np.mean(err[:,:-1]**2)**.5,np.mean(err[:,-1]**2)**.5],5),'complete/missing',np.round([np.mean(err[~missing]**2)**.5,np.mean(err[missing]**2)**.5],5))
print('lab, stratum, missing count, latent, six stress moments, candidate drifts'); print(np.round(np.c_[labs,g['stratum'],np.isnan(raw).sum(1),latent,past,*(th[2]+x@th[3:] for x in xs.values())],4))
print('fit complete rows, latent by lab: intercept, scale; residuals')
for lab in np.unique(labs):
 for name in ('diag only','train+test fixed'):
  i=(labs==lab)&~missing; r=th[2]+xs[name]@th[3:]; X=np.c_[np.ones(len(r)),r]; b=np.linalg.lstsq(X[i],latent[i],rcond=None)[0]; print(lab,name,np.round(b,4),np.round((latent-X@b)[labs==lab],4))
