exec(open('/tmp/chbasis.py').read().split('C,E=profile(tau)')[0])
raw=di['coupon_features_raw']; obs=np.isfinite(raw); xn=raw.copy(); labs=di['lab_id']; ref=di['ref_coupons'];
for l in np.unique(labs):
 k=labs==l; xn[k]=(xn[k]-np.median(ref[k].mean(0)))/np.median(ref[k].std(0))
P=basis(tau); base=hd[:,None]*(P[:,:2]@beta[:2]); Q=Y-base
variants={'pooled':np.where(obs,xn,np.nanmean(xn,0)),'cell':m._features(di,di)[1]}
for key,xx in variants.items():
 r=beta[2]+xx@beta[3:]
 for affine in [False,True]:
  S=np.c_[np.ones(len(r)),r] if affine else r[:,None]
  co=np.linalg.lstsq(S,Q,rcond=None)[0]@np.linalg.pinv(P.T)
  e=np.empty_like(Q)
  for l in np.unique(labs):
   k=labs==l
   cf=np.linalg.lstsq(S[~k],Q[~k],rcond=None)[0]@np.linalg.pinv(P.T)
   e[k]=Q[k]-S[k]@cf@P.T
  print(key,affine,'coef',co.tolist(),'RMS',np.sqrt((e**2).mean(0))[[0,5,-2,-1]],'max curve/rate',np.max(abs(e[:,:-1])),np.max(abs(e[:,-1])), 'missing max',np.max(abs(e[~obs.all(1)])))
print('candidate drift test extrema, contract vs separate',[(np.min(beta[2]+z@beta[3:]),np.max(beta[2]+z@beta[3:])) for z in [xt,m._features(te,te)[0]]])
a,c=m._design(tau,x,p,h,dt,t); et=tr['strain_curves_obs']-tr['strain_0'][:,None]-a@beta; er=tr['instant_rate_obs']-c@beta
for l in np.unique(tr['lab_id']):
 k=tr['lab_id']==l
 print('train lab',l,'n',k.sum(),'rate median',np.median(er[k]),'late median',np.median(et[k,-1]))
