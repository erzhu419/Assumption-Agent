exec(open('/tmp/alpha_cv.py').read().split('for layout')[0])
lab, strata, raw, ref = (np.concatenate([d[s][k] for s in ('train','test')]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons'))
_, cell, sizes = np.unique(np.column_stack((lab,strata)),axis=0,return_inverse=True,return_counts=True)
for scaling in ('scalar_panel_columns','interleaved','feature_major'):
    x = np.empty_like(raw)
    for j in range(4):
        ix = lab==j
        a = ref[ix]
        if scaling == 'scalar_panel_columns':
            loc = np.nanmedian(np.nanmean(a,axis=0)); sc = np.nanmedian(np.nanstd(a,axis=0))
        else:
            a = a.reshape(-1,8,4) if scaling=='interleaved' else a.reshape(-1,4,8).transpose(0,2,1)
            loc = np.nanmedian(np.nanmean(a,axis=1),axis=0); sc = np.nanmedian(np.nanstd(a,axis=1),axis=0)
        x[ix] = (raw[ix]-loc)/sc
    for fill in ('cell','balanced_rows','balanced_observed_cells'):
        cm = np.array([np.nanmean(x[cell==i],axis=0) for i in range(len(sizes))])
        w = (1/sizes[cell])[:,None]
        wr = np.nansum(w*x,axis=0)/np.sum(w*np.isfinite(x),axis=0)
        aligned = np.where(np.isfinite(x),x,{'cell':cm[cell],'balanced_rows':wr,'balanced_observed_cells':np.nanmean(cm,axis=0)}[fill])
        inp = m._inputs(tr,aligned[:n],cal); metrics=[]
        for j in range(4):
            fit=labs!=j; val=~fit
            law=lambda mask:m._Law(tuple(k[mask] for k in inp),dt,t)
            p=m._fit(law(fit),y[fit],rate[fit]); c,r=law(val).evaluate(p)
            ec=abs(c-y[val]); er=abs(r-rate[val]); missing=~np.isfinite(raw[:n][val]).all(1)
            metrics.append([np.median(ec),np.median(ec,axis=0).max(),np.quantile(er,.9),np.median(er[missing]) if missing.any() else 0, p[0],p[1]])
        print(scaling,fill,'worst [curve median,horizon median,rate p90,missing rate median]', np.round(np.max(metrics,axis=0)[:4],6),'tau spans',np.round(np.ptp(np.array(metrics)[:,4:],axis=0),5))
