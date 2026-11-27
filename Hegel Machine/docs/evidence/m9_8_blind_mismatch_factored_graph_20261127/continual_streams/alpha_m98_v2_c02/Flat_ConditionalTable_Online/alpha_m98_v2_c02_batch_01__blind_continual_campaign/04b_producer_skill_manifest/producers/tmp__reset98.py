exec(open('/tmp/law98.py').read().split('for mode in range')[0])
from scipy.optimize import minimize_scalar
n=len(h); ids=np.arange(n); lab=a['lab_id']; complete=np.isfinite(a['coupon_features_raw']).all(1)
qd=b[2]+m['_standardize'](a,a)[0]@b[3:]; G=h[:,None]*(b[:2]@T)
for featname,dr in [('standalone',qd),('pooled',q)]:
 for kind in ['reset','scale','history']:
  def fit(ii):
   def err(v,full=False):
    Z=dr[:,None,None]*expb(v)[None,:,None]
    if kind=='scale': Z=np.concatenate([Z,G[:,:,None]],2)
    if kind=='history': Z=np.concatenate([Z,(base@T-G)[:,:,None]],2)
    c=np.linalg.lstsq(Z[ii].reshape(-1,Z.shape[-1]),(Y-G)[ii].ravel(),rcond=None)[0]; E=G+Z@c-Y
    return (E,c) if full else np.sum(E[ii]**2)
   v=minimize_scalar(err,bounds=(.3,10),method='bounded').x
   return *err(v,True),v
  E,c,v=fit(ids[complete]); loo=np.stack([fit(ids[complete & (ids!=i)])[0][i] for i in ids]); print(featname,kind,'params',v,c,'complete loo',rm(loo[complete])/[1,1,1,.7],'missing loo',rm(loo[~complete])/[1,1,1,.7]);
  if featname=='standalone' and kind=='reset':
   for g in np.unique(lab):
    ii=ids[lab==g]; el=fit(ids[complete & (lab!=g)])[0][ii]; print('lab',g,rm(el)/[1,1,1,.7]);
   print('row rate errors',E[:,-1]/.7)
   print('fixed unit gain drift timescale',minimize_scalar(lambda v: np.sum((G+qd[:,None]*v*expb(v)-Y)[complete]**2),bounds=(.3,10),method='bounded').x)
