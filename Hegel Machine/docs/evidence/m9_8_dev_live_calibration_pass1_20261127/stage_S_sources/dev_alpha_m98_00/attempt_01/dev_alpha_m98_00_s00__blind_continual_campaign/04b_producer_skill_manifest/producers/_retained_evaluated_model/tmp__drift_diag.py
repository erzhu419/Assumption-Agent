import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/pilot_diag.py').read())
raw=pi['coupon_features_raw']; lab=pi['lab_id']; ref=pi['ref_coupons']; print('raw missing',np.where(np.isnan(raw))); print('pilot values'); print(np.c_[slope,R,raw,ref.mean(1),ref.std(1)].round(4))
for label,blocks in [('shared',(tr,te)),('pilot',(pi,)),('pilot+test',(pi,te)),('all',(tr,te,pi)),('test',(te,)),('train',(tr,))]:
 labs=np.concatenate([z['lab_id'] for z in blocks]); refs=np.concatenate([z['ref_coupons'] for z in blocks]); X=raw.copy()
 for l in np.unique(lab):
  q=refs[labs==l]; X[lab==l]=(X[lab==l]-np.median(q.mean(0)))/np.median(q.std(0))
 X=np.where(np.isfinite(X),X,np.nanmean(X,axis=0)); dr=theta[2]+X@theta[3:]; err=slope+drift-dr
 print(label,'offset nrm quantiles',np.quantile(abs(err),[.5,.9,1]).round(4),'err',err.round(4),'drift corr',np.corrcoef(dr,slope+drift)[0,1].round(4))
