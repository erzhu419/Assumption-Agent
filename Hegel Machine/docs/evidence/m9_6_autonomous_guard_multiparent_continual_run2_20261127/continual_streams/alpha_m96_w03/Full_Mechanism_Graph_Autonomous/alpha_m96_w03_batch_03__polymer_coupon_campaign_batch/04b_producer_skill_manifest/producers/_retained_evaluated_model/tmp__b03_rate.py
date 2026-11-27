exec(open('/tmp/b03_diag.py').read().split('p=m._fit(h,tr);')[0])
case=pickle.load(open('/holdout_task/public_alpha_m96_w03_batch_03.pkl','rb'))[0]; gt=case['gt']; target=m._Hold(d['past_dt'],d['eval_times'],te,xe,cal); time=d['eval_times']; miss_tr=~np.isfinite(tr['coupon_features_raw']).all(1); miss_te=~np.isfinite(te['coupon_features_raw']).all(1)
def sweep(H,b,coef):
 a,_=H.design(coef[:2]); v=a@coef[2:]; r=b['instant_rate_obs']-v[:,-1]; return v[:,:-1], r
fit=m._fit(h,tr); cv=np.empty_like(y[:,:-1]); rate=np.empty(len(y))
for l in range(4):
 use=tr['lab_id']==l; v,r=sweep(h,tr,m._fit(h,tr,~use)); cv[use],rate[use]=v[use],r[use]
print('test rate error abs quantiles vs clean:',np.quantile(abs(te['instant_rate_obs']-gt['te_true_rate0']),[0,.5,.9,.95,.99,1]))
print('relative tau error',abs(fit[:2]/gt['taus']-1),'max theta error',np.max(abs(fit[2:]-gt['theta'])))
pred,rt=sweep(target,te,fit)
for strength in (0.,.25,.5,.75,1.):
 for label,mat,r,missing,lab in [('LOLO',cv-y[:,:-1],rate,miss_tr,tr['lab_id']),('public clean',pred+te['strain_0'][:,None]-gt['te_clean'],rt,miss_te,te['lab_id'])]:
  v=mat+np.where(missing,np.nan_to_num(r)*strength,0.)[:,None]*time
  print(label,strength,'missing rmse',np.sqrt(np.nanmean(v[missing]**2)),'missing terminal rmse',np.sqrt(np.nanmean(v[missing,-1]**2)),'complete rmse',np.sqrt(np.nanmean(v[~missing]**2)),'worst lab',max(np.sqrt(np.nanmean(v[lab==l]**2)) for l in range(4)))
print('LOLO rate residual quantiles complete',np.nanquantile(abs(rate[~miss_tr]),[.5,.9,.95,1])); print('LOLO rate residual quantiles missing',np.nanquantile(abs(rate[miss_tr]),[.5,.9,.95,1]))
print('missing reference count',miss_te.sum(),'finite missing rates',np.isfinite(te['instant_rate_obs'][miss_te]).sum())
out=m.solve_alpha_m96_w03_batch_03(d); print('baseline allocation errors rmse/max',np.sqrt(np.mean((out['allocation_weights']-gt['w_alloc_opt'])**2)),np.max(abs(out['allocation_weights']-gt['w_alloc_opt'])))
