exec(open('/tmp/law98.py').read().split('for mode in range')[0])
ids=np.arange(len(h)); missing=np.isnan(a['coupon_features_raw']); complete=~missing.any(1)
A0=h[:,None]*(b[:2]@T); print('diagnostic missing rows',np.where(~complete)[0]); print('stress-only rate residual',np.c_[h,q,Y[:,-1]/.7-A0[:,-1]/.7,missing.astype(int)]); print('test missing counts',np.isnan(d['test']['coupon_features_raw']).sum(0))
def fit3(ii,kind,ix=False):
 ss=np.sum(s,axis=1); dif=np.diff(s,axis=1); zf=s@W[:,0]; zs=s@W[:,1]
 X={'zero':np.empty((len(h),0)),'mean':ss[:,None], 'moments':np.c_[ss,zf,zs], 'history':s, 'squared':np.c_[ss,np.mean(s*s,axis=1),zf,zs], 'contrasts':np.c_[ss,zf,zs,dif[:,-1],dif[:,-2]], 'hold':h[:,None]}[kind]
 # Missing diagnostic covariates can be profiled as unknown drift, then omitted at prediction.
 F=np.c_[q, np.eye(len(h))[:,~complete]] if ix else q[:,None]
 def calc(p,ret=False):
  Z=np.concatenate([F[:,None,:]*expb(p[0])[None,:,None],X[:,None,:]*(T[0]+p[1]*T[1])[None,:,None]],axis=2)
  co=np.linalg.lstsq(Z[ii].reshape(-1,Z.shape[2]),(Y-A0)[ii].ravel(),rcond=None)[0]; e=A0+Z@co-Y
  return (e,co) if ret else e[ii].ravel()
 p=least_squares(calc,[2.1,5.5],bounds=([.3,-30],[20,30]),max_nfev=70).x
 return (*calc(p,True),p)
for kind in ['zero','mean','moments','history','squared','contrasts','hold']:
 for ix in [False,True]:
  e,c,p=fit3(ids,kind,ix); loo=np.stack([fit3(ids[ids!=i],kind,ix)[0][i] for i in ids]);print(kind,ix,'p',p,'C',c,'fit',rm(e)/[1,1,1,.7],'loo',rm(loo)/[1,1,1,.7],'complete loo',rm(loo[complete])/[1,1,1,.7]);
print('zero history, per-coupon drift amplitude'); H0=expb(2.1); print(np.c_[q,(Y-A0)@H0/(H0@H0),np.sqrt(np.mean(((Y-A0)-((Y-A0)@H0/(H0@H0))[:,None]*H0)**2,axis=1))]);
