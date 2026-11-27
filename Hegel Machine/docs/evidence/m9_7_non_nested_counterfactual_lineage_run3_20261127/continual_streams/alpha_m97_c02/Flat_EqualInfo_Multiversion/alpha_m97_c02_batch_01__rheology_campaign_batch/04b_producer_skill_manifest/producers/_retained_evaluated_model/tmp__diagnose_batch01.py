exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
y = train['strain_curves_obs'] - train['strain_0'][:, None]
r = train['instant_rate_obs']
w = np.r_[gt['theta'], gt['vp_theta']]
c, a = m._design(*data, gt['taus'])
for name, split, sl, obs, rate in [('train',train,slice(0,n),y,r),('reference',test,slice(n,None),gt['te_clean']-test['strain_0'][:,None],gt['te_true_rate0'])]:
    p = data[3][sl]; miss = np.isnan(split['coupon_features_raw']).any(1)
    ec = obs - c[sl] @ w; er = rate - a[sl] @ w
    phi = np.where(p[:,None],np.log1p(data[-1]),data[-1]); u=ec/phi
    drift = np.median(u,axis=1)
    irregular = np.max(np.abs(u-drift[:,None]),axis=1)
    q=lambda v: np.round(np.quantile(v,[0,.1,.5,.9,.95,.99,1]),6)
    for gate in [False,True]:
        ix = (p == gate) & ~miss
        print(name,gate,'drift',q(drift[ix]),'rate',q(er[ix]),'rate minus drift',q((er-drift)[ix]),'irregular',q(irregular[ix]))
        for lab in range(4):
            aidx = ix & (split['lab_id']==lab)
            print('lab',lab,'drift quantiles',q(drift[aidx]),'rmse curve/rate',np.sqrt(np.mean(ec[aidx]**2)), np.sqrt(np.mean(er[aidx]**2)))
    z=np.c_[np.ones(len(p)),data[2][sl],np.eye(4)[split['lab_id']][:,1:]]
    ix=(~miss)&(irregular < np.median(irregular)*3)
    if ix.sum()>8:
        for v in (drift,er):
            b=np.linalg.lstsq(z[ix],v[ix],rcond=None)[0]; print(name,'drift regression',np.round(b,6),'residual',q((v-z@b)[ix]))
for maskname, mask in [('complete',~np.isnan(train['coupon_features_raw']).any(1)), ('all',np.ones(n,bool))]:
    a,wfit=m._fit(tuple(a[:n][mask] for a in data[:5])+data[5:],y[mask],r[mask]); print(maskname,'taus/weights',a,wfit)
