exec(open('/tmp/chamber_probe.py').read().split('for name,v')[0])
from scipy.optimize import least_squares
Y=np.c_[y,yr]; raw=g['coupon_features_raw']; L=g['lab_id']; complete=np.isfinite(raw).all(1)
def split_features(s):
 q=s['coupon_features_raw'].copy(); l=s['lab_id']; ref=s['ref_coupons']
 for i in np.unique(l):
  z=l==i; q[z]=(q[z]-np.median(ref[z].mean(0)))/np.median(ref[z].std(0))
 return q
q=split_features(g); X=np.c_[np.ones(len(q)),q]; H=hg
F=np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1]])
D=X@theta[2:]; B=H[:,None]*(F[:,:2]@theta[:2]); V=np.c_[H,D]; C=np.linalg.lstsq(V[complete],Y[complete],rcond=None)[0]; print('curve signatures'); print('hold fit', C[0]); print('hold prior',B[0]/H[0]); print('drift fit',C[1]); print('projection',C@np.linalg.pinv(F.T),'rmse',np.sqrt(np.mean((C-C@np.linalg.pinv(F.T)@F.T)**2,axis=1)))
K=np.linalg.lstsq(D[complete,None],(Y-B)[complete],rcond=None)[0][0]; proj=(Y-B)@K/(K@K); print('exact hold plus free drift shape',K, 'projection',np.linalg.lstsq(F,K,rcond=None)[0], 'rmse',np.sqrt(np.mean((Y[complete]-B[complete]-D[complete,None]*K)**2)))
for i in np.flatnonzero(~complete):
 j=np.flatnonzero(np.isnan(q[i])); print('missing',i,'lab',L[i], 'cols',j,'implied', (proj[i]-np.nansum(X[i]*theta[2:]))/theta[3:][j], 'rank residual',np.sqrt(np.mean((Y[i]-B[i]-proj[i]*K)**2)))
for name,s in [('train',tr),('test',te),('diag',g)]:
 print(name,list(s),'missing',np.isnan(s['coupon_features_raw']).sum(0),'ref range',np.min(s['ref_coupons']),np.max(s['ref_coupons']))
print('dataset fields',list(d)); print('archived fit implementation'); print(open('/tmp/chamber_base.py').read().split('def _fit')[1])
