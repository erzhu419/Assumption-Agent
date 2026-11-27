exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
labs, strata, raw, refs = (np.concatenate((train[k],test[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons'))
a = np.empty_like(raw)
for lab in range(4):
    ix = labs == lab
    r = refs[ix]
    a[ix] = (raw[ix]-np.nanmedian(np.nanmean(r,axis=0)))/np.nanmedian(np.nanstd(r,axis=0))
w = np.r_[gt['theta'],gt['vp_theta']]
c, rate = m._design(*data,gt['taus'])
e = gt['te_true_rate0']-rate[n:]@w
for j in np.flatnonzero(np.isnan(a[n:]).any(1)):
    i=n+j; missing=np.isnan(a[i]); cell=(labs==labs[i])&(strata==strata[i]); feat=w[3:7][missing]; old=data[2][i,missing]
    candidates={'global balanced':old, 'own cell':np.nanmean(a[cell],axis=0)[missing], 'own lab':np.nanmean(a[labs==labs[i]],axis=0)[missing], 'own stratum':np.nanmean(a[strata==strata[i]],axis=0)[missing], 'zero':np.zeros(missing.sum())}
    print('row',j,'lab/stratum',labs[i],strata[i],'missing',np.flatnonzero(missing),'inferred weighted fill',round(old@feat+e[j],6),'candidate weighted fills', {k:round(v@feat,6) for k,v in candidates.items()})
print('complete feature correlation',np.round(np.corrcoef(a[np.isfinite(a).all(1)].T),3))
print('own-cell feature means',[(int(l),int(s),np.round(np.nanmean(a[(labs==l)&(strata==s)],axis=0),3).tolist()) for l,s in np.unique(np.column_stack((labs,strata)),axis=0)])
