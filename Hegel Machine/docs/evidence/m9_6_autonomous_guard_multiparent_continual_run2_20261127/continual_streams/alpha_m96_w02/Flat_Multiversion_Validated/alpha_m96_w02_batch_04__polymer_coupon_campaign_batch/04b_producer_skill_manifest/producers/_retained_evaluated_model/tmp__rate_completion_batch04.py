exec(open('/tmp/check_batch04.py').read().split('out=errors')[0])
rng=np.random.default_rng(24)
for c in range(4):
    m=lab==c
    dp=dict(d,train=subset(tr,~m),test=subset(tr,m)); te=dp['test']
    o=s.solve_alpha_m96_w02_batch_04(dp)
    miss=np.isnan(te['coupon_features_raw']).any(axis=1)
    ri=te['instant_rate_obs']-o['initial_rate_pred']
    rc=(te['strain_curves_obs']-o['predicted_strain'])/d['eval_times']
    print('held lab',c,'missing',sum(miss),'complete rate absolute quantiles',np.round(np.quantile(abs(ri[~miss]),[.5,.9,.95,1]),4))
    print('  missing: observed rate residual / median curve slope residual',np.round(np.c_[ri[miss],np.median(rc[miss],axis=1)],4).tolist())
    print('  complete: median abs rate minus median curve slope',np.median(abs(ri[~miss]-np.median(rc[~miss],axis=1))))
o=s.solve_alpha_m96_w02_batch_04(d); te=d['test']; missing=np.isnan(te['coupon_features_raw']).any(axis=1); r=te['instant_rate_obs']-o['initial_rate_pred']; noise=te['instant_rate_obs']-g['te_true_rate0']; t=d['eval_times'];
print('public missing: observed rate innovation / measurement error',np.round(np.c_[r[missing],noise[missing]],5).tolist()); print('public complete rate measurement absolute quantiles',np.quantile(abs(noise[~missing]),[.5,.9,.95,1]))
hist=np.any(abs(g['te_hist_true'])>d['yield_stress'],axis=1); hold=abs(g['te_hold_true'])>d['yield_stress']
for label, correction in [('baseline', np.zeros(len(r))), ('rate completion',np.where(missing,r,0.))]:
    e=o['predicted_strain']+correction[:,None]*t-g['te_clean']; f=o['initial_rate_pred']+correction-g['te_true_rate0']
    print(label,'overall RMSE',rms(e),rms(f),'worst horizon',max(rms(e[:,k]) for k in range(e.shape[1])))
    for h,j in [(False,False),(True,False),(False,True),(True,True)]:
        idx=(hist==h)&(hold==j)
        if idx.any(): print(' ',h,j,'RMSE',rms(e[idx]),rms(f[idx]))
