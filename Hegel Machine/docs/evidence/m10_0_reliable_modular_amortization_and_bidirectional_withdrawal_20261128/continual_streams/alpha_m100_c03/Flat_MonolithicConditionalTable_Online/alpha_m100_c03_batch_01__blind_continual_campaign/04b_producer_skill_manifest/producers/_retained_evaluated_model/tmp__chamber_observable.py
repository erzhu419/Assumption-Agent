exec(open('/tmp/chamber_law.py').read().split('for label,F in')[0])
g=theta[2]+xc@theta[3:]; F=np.c_[np.ones(n),hc,g]; miss=np.isnan(c['coupon_features_raw']).any(axis=1); lab=c['lab_id']; indices=np.arange(n)
for label,A in [('affine',F),('quadratic',np.c_[F,g*g,hc*g,hc*hc]),('history',np.c_[F,pc]),('intrinsic slope',F[:,[0,2]])]:
 def fit(ix):
  offset=hc*(theta[:2]/tau).sum() if label=='intrinsic slope' else np.zeros(n)
  q=np.linalg.lstsq(A[ix],(v-offset)[ix],rcond=None)[0]
  p=least_squares(lambda p:A[ix]@p+offset[ix]-v[ix],q,loss='cauchy',f_scale=.006).x
  return A@p+offset-v,p
 e=np.empty(n)
 for ell in np.unique(lab):
  mask=lab==ell; e[mask]=fit(indices[~mask])[0][mask]
 print('rate',label,'rmse/max',np.mean(e**2)**.5,np.max(abs(e)),'complete',np.mean(e[~miss]**2)**.5,'coef',fit(indices)[1].round(7),'worst indices',np.argsort(abs(e))[-4:])
print('missing covariates',np.c_[hc,g,np.isnan(c['coupon_features_raw'])][miss].round(5))
