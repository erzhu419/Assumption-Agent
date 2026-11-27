exec(open('/tmp/chbasis.py').read().split('C,E=profile(tau)')[0])
P=basis(tau); Q=Y-hd[:,None]*(P[:,:2]@beta[:2]); raw=di['coupon_features_raw']; obs=np.isfinite(raw); good=obs.all(1); xn=raw.copy(); lab=di['lab_id']; strat=di['stratum']; ref=di['ref_coupons'];
for l in np.unique(lab):
 k=lab==l; xn[k]=(xn[k]-np.median(ref[k].mean(0)))/np.median(ref[k].std(0))
_,xs=m._features(di,di); r=beta[2]+xs@beta[3:]; a=np.c_[np.ones(len(r)),r]; b=np.linalg.lstsq(a[good],Q[good],rcond=None)[0]; b0=(r[good]@Q[good])/(r[good]@r[good]); print('modes: affine, origin',np.linalg.lstsq(P,b.T,rcond=None)[0].T,np.linalg.lstsq(P,b0,rcond=None)[0]);
for label,fill in [('cell-balanced',xs),('zero',np.nan_to_num(xn)),('pooled',np.where(obs,xn,np.nanmean(xn,0))),('median',np.where(obs,xn,np.nanmedian(xn,0)))]:
 rr=beta[2]+fill@beta[3:]; e=Q-np.c_[np.ones(len(rr)),rr]@b; print(label,'missing drift, residuals',rr[~good].round(5),e[~good][:,[-2,-1]].round(5));
shift=(Q-np.c_[np.ones(len(r)),r]@b)@b[1]/(b[1]@b[1]); print('implied centers',[(j,k,round(xs[j,k]+shift[j]/beta[3+k],5)) for j,k in zip(*np.where(~obs))]); print('actual centers',xs[~obs]); print('zero-intercept complete max',np.max(abs((Q-r[:,None]*b0)[good]),axis=0)[[0,-2,-1]]); print('train summary by missingness'); at,ar=m._design(tau,x,p,h,dt,t); et=tr['strain_curves_obs']-tr['strain_0'][:,None]-at@beta; er=tr['instant_rate_obs']-ar@beta; mask=np.isfinite(tr['coupon_features_raw']).all(1);
for key,k in [('complete',mask),('missing',~mask)]: print(key,np.quantile(abs(et[k]),[.5,.9,.95,1]),np.quantile(abs(er[k]),[.5,.9,.95,1]));
print('test drift range',np.quantile(beta[2]+xt@beta[3:],[0,.1,.5,.9,1])); print('diagnostic drift range',np.min(r),np.max(r))
