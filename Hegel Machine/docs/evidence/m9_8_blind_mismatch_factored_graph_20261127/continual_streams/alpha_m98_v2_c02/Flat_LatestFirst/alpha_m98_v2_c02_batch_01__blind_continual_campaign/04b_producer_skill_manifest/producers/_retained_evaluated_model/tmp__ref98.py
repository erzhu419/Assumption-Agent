exec(open('/tmp/chamber98.py').read().split('np.set_printoptions')[0])
np.set_printoptions(precision=5,suppress=True,linewidth=150)
lab=ck['lab_id']; st=ck['stratum']; raw=ck['coupon_features_raw']; refs=ck['ref_coupons']; valid=np.isfinite(raw).all(1); regular=lab!=3
_,idx,cnt=np.unique(np.c_[lab,st],axis=0,return_inverse=True,return_counts=True); weights=1/cnt[idx]
for poolname,pool in [('diag',ck),('joint',allsite),('test',te),('train',tr)]:
 for convention in ['median','flat','rms']:
  xx=raw.copy()
  for l in np.unique(lab):
   p=pool['ref_coupons'][pool['lab_id']==l]; avg=np.mean(p,0); std=np.std(p,0); off=np.median(avg) if convention=='median' else p.mean(); scale=np.median(std) if convention=='median' else p.std() if convention=='flat' else np.sqrt(np.mean(std**2)); xx[lab==l]=(xx[lab==l]-off)/(scale if scale>1e-8 else 1.)
  xx=np.where(np.isfinite(xx),xx,np.nansum(weights[:,None]*xx,0)/np.sum(weights[:,None]*np.isfinite(xx),0)); rr=np.c_[np.ones(len(xx)),xx]@w[2:]
  delta=Y-H[:,None]*(B@w[:2]); Q=rr[:,None,None]*B[None,:,:]; k=np.linalg.lstsq(Q[regular].reshape(-1,2),delta[regular].ravel(),rcond=None)[0]; err=Q@k-delta
  print(poolname,convention,'gain',k,'RMSE curve/rate reg clean',np.sqrt(np.mean(err[valid&regular,:-1]**2)),np.sqrt(np.mean(err[valid&regular,-1]**2)),'row errors',np.round(np.sqrt(np.mean(err**2,1)),4))
  if poolname=='diag' and convention=='median':
   q=B@k; observed=delta@q/(q@q); print('lab, stratum, missing, raw features, mapped drift, inferred drift',np.c_[lab,st,np.isnan(raw).sum(1),raw,rr,observed]); print('diagnostic counts',cnt)
