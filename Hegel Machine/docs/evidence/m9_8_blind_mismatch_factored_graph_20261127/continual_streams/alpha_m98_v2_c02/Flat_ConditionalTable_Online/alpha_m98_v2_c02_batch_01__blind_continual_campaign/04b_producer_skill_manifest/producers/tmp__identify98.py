exec(open('/tmp/law98.py').read().split('for mode in range')[0])
from scipy.optimize import least_squares
ids=np.arange(len(h)); lab=a['lab_id']; complete=np.isfinite(a['coupon_features_raw']).all(1)
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; cf=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],cal['true_channel_loads'],rcond=None)[0]; S=a['past_channel_raw']@cf[:4]+cf[4]; H=a['hold_channel_raw']@cf[:4]+cf[4]
A0=h[:,None]*(b[:2]@T)
# Use time shape to distinguish features added to drift from features added to memory.
for key,X in [('q',q[:,None]),('affine q',np.c_[np.ones(len(h)),q]),('std',f),('raw',np.c_[np.ones(len(h)),np.nan_to_num(a['coupon_features_raw'])]),('q+history',np.c_[np.ones(len(h)),q,s]),('q+load',np.c_[np.ones(len(h)),q,h]),('q+geometry',np.c_[np.ones(len(h)),q,s.mean(1),s[:,-1],(s*s).mean(1)]),('q+nonlinear',np.c_[np.ones(len(h)),q,q*q,q**3]),('q+interactions',np.c_[np.ones(len(h)),q,h,q*h]),('std+contrast',np.c_[f,s.mean(1),s[:,-1]])]:
 def fit(ii):
  def calc(p,ret=False):
   Z=X[:,None,:]*expb(p[0])[None,:,None]; eg=A0+p[1]*(base@T-A0)
   co=lstsq(Z[ii].reshape(-1,X.shape[1]),(Y-eg)[ii].ravel())[0]; E=eg+Z@co-Y
   return (E,co) if ret else E[ii].ravel()
  p=least_squares(calc,[2.1,0],bounds=([.35,-2],[12,3])).x; return (*calc(p,True),p)
 E,C,p=fit(ids); loo=np.stack([fit(ids[ids!=i])[0][i] for i in ids]); lolo=np.stack([fit(ids[lab!=lab[i]])[0][i] for i in ids]);print(key,p,C,'fit',rm(E)/[1,1,1,.7],'loo',rm(loo)/[1,1,1,.7],'lolo',rm(lolo)/[1,1,1,.7]);
 if key in ('q','std'):
  res=E@expb(p[0]); D=np.c_[h,s,S.reshape(len(h),-1),H,np.mean(S*S,(1,2)),s.mean(1),s.std(1),a['strain_0'],lab,a['stratum'],np.nan_to_num(a['ref_coupons'])]; cr=np.corrcoef(np.c_[res,D].T)[0,1:]; print('top residual covariates',sorted(enumerate(cr),key=lambda ij:-abs(ij[1]))[:12]); print('residual per row',np.c_[lab,complete,q,res])
