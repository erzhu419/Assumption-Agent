exec(open('/tmp/chamber_amplitude.py').read().split('zsets=')[0].replace('print(np.c_[C,H,G,D,U].round(4))','pass'))
xraw=c['coupon_features_raw'].copy()
for l in np.unique(lab):
 s=lab==l; ref=c['ref_coupons'][s]; xraw[s]=(xraw[s]-np.median(ref.mean(0)))/np.median(ref.std(0))
_,ix,n=np.unique(np.c_[lab,c['stratum']],axis=0,return_inverse=True,return_counts=True)
_,xt=m._features(c,c)
for i,j in zip(*np.where(np.isnan(xraw))):
 v=xraw[:,j]; finite=np.isfinite(v); w=1/n[ix]; wc=1/np.bincount(ix[finite],minlength=len(n)).clip(1)[ix]
 print('missing',i,j,'unweighted/median/balanced/observed-balanced/lab/stratum/joint',np.round([np.nanmean(v),np.nanmedian(v),np.average(v[finite],weights=w[finite]),np.average(v[finite],weights=wc[finite]),np.nanmean(v[lab==lab[i]]),np.nanmean(v[c['stratum']==c['stratum'][i]]),Q[i,j]],6))
