exec(open('/tmp/explore_chamber.py').read().split('for tt in [taus]:')[0])
h=cq[1]; B=basis(taus)[:,:2]; amps=np.linalg.lstsq(B,y.T,rcond=None)[0].T; delta=amps-h[:,None]*th[:2]; raw=q['coupon_features_raw']; qlab=q['lab_id']; te=d['test']; ok=np.isfinite(raw).all(1)
print('singular values of two amplitude corrections',np.linalg.svd(delta,compute_uv=False)); print('through-zero fast/slow correction ratio',np.linalg.lstsq(delta[:,1:],delta[:,0],rcond=None)[0]); print('observed correction at each coupon',np.round(delta[:,1],4))
for name,parts in [('joint',[tr,te]),('diagnostic',[q]),('joint+diag',[tr,te,q]),('train',[tr]),('test',[te]),('raw',[])]:
 X=raw.copy();
 if parts:
  labs=np.concatenate([r['lab_id'] for r in parts]); refs=np.concatenate([r['ref_coupons'] for r in parts]); allx=np.concatenate([r['coupon_features_raw'] for r in parts]).copy(); st=np.concatenate([r['stratum'] for r in parts]);
  for g in np.unique(labs):
   r=refs[labs==g]; center=np.median(r.mean(0)); scale=np.median(r.std(0)); X[qlab==g]=(X[qlab==g]-center)/scale; allx[labs==g]=(allx[labs==g]-center)/scale; print(name,g,round(center,3),round(scale,3),end='; ')
  _,ix,ct=np.unique(np.c_[labs,st],axis=0,return_counts=True,return_inverse=True); w=(1/ct[ix])[:,None]; fill=np.nansum(w*allx,axis=0)/np.sum(w*np.isfinite(allx),axis=0); X=np.where(np.isnan(X),fill,X)
 else: X=np.where(np.isnan(X),np.nanmean(X,0),X)
 dr=np.c_[np.ones(len(X)),X]@th[2:]; V=np.c_[np.ones(len(X)),dr]; k=np.linalg.lstsq(V[ok],delta[ok],rcond=None)[0]; err=delta-V@k; print('\n',name,'intercept/drift',np.round(k,4).tolist(),'full/complete rmse',np.sqrt(np.mean(err**2,0)),np.sqrt(np.mean(err[ok]**2,0))); print('slow residuals',np.round(err[:,1],3));
 if name=='diagnostic': print('diagnostic standardized features, dr',np.round(np.c_[X,dr],4));
print('diagnostic ref center/dispersion by coupon',np.round(np.c_[q['ref_coupons'].mean(1),q['ref_coupons'].std(1)],4)); print('diagnostic lab and strata',np.c_[qlab,q['stratum']].T)
