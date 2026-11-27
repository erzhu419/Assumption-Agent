import numpy as np, pickle
c=pickle.load(open('/holdout_task/public_alpha_m96_w01_s00.pkl','rb'))[0]; d=c['dataset']; gt=c['gt']; tr=d['train']; te=d['test']; t=d['eval_times']; lab=np.r_[tr['lab_id'],te['lab_id']]; s=np.r_[tr['stratum'],te['stratum']]; raw=np.r_[tr['coupon_features_raw'],te['coupon_features_raw']]; ref=np.r_[tr['ref_coupons'],te['ref_coupons']]; x=raw.copy(); print('gt theta',gt['theta']);
for j in np.unique(lab):
 m=lab==j; a=np.median(np.nanmean(ref[m],axis=0)); b=np.median(np.nanstd(ref[m],axis=0)); print('refs',j,a,b); x[m]=(raw[m]-a)/b
th=gt['theta']; tau=gt['taus']; z=np.zeros((len(te['lab_id']),2));
for j,dt in enumerate(d['past_dt']):
 a=-np.expm1(-dt/tau); z+=a*(gt['te_hist_true'][:,j,None]-z)
B=(gt['te_hold_true'][:,None]-z)*th[:2]; target=gt['te_true_rate0']-np.sum(B/tau,axis=1)-th[2]; complete=np.isfinite(raw[200:]).all(1)
for name,v in [('raw',raw),('aligned',x)]:
 for fill in ['cell','lab','balanced','pooled','trainbalanced']:
  w=v.copy(); means=np.array([np.nanmean(w[(lab==j)&(s==k)&((np.arange(len(s))<200) if fill=='trainbalanced' else True)],axis=0) for j,k in sorted(set(zip(lab,s)))]); mu=np.nanmean(means,axis=0)
  for j,k in sorted(set(zip(lab,s))):
   m=(lab==j)&(s==k); q=np.nanmean(w[m if fill=='cell' else lab==j if fill=='lab' else np.ones(len(lab),bool)],axis=0) if fill in ['cell','lab','pooled'] else mu; w[m]=np.where(np.isfinite(w[m]),w[m],q)
  err=w[200:]@th[3:]-target; print(name,fill,'complete/missing max',*(np.max(abs(err[m])) for m in [complete,~complete]),'missing errors',np.round(err[~complete],5))
print('test rate noise quantiles',np.quantile(abs(te['instant_rate_obs']-gt['te_true_rate0']),[0,.5,.9,.99,1]))
print('curve/rate exact consistency', np.max(abs(gt['te_clean']-(te['strain_0'][:,None]+B@(-np.expm1(-t[None,:]/tau[:,None]))+(target+th[2])[:,None]*t))))
