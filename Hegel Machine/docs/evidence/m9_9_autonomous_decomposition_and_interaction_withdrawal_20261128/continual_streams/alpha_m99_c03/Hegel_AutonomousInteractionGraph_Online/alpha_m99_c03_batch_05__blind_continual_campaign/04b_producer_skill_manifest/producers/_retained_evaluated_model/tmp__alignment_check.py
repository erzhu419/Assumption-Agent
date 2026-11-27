exec(open('/tmp/probe.py').read().split("print('ranges'")[0])
raw=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw'])); lab=np.r_[tr['lab_id'],te['lab_id']]; ref=np.vstack((tr['ref_coupons'],te['ref_coupons'])); raw=raw.copy(); n=len(ytr)
for g in np.unique(lab):
 m=lab==g; raw[m]=(raw[m]-np.median(ref[m].mean(0)))/np.median(ref[m].std(0))
missing=~np.isfinite(raw[:n]).all(1)
for label,feat in [('joint-balanced',x),('joint-plain',np.where(np.isfinite(raw),raw,np.nanmean(raw,axis=0))),('separate-balanced',np.vstack((p._features(tr,tr)[:n],p._features(te,te)[:len(te['lab_id'])])))]:
 a,b=p._design(tau,(train[0],train[1],np.c_[np.ones(n),feat[:n]]),dt,t)
 print(label,'train missing residual MAD strain/rate',*[float(np.median(np.abs(e[missing]))) for e in (a@th-ytr,b@th-rtr)],'deployment maximum feature delta',np.max(np.abs(feat[n:]-x[n:])))
