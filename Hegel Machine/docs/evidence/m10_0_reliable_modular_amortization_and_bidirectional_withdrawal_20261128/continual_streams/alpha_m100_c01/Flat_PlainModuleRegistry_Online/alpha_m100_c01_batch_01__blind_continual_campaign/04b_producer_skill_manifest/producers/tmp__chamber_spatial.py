exec(open('/tmp/chamber_probe.py').read().split('for name,v')[0])
from scipy.optimize import least_squares
Y=np.c_[y,yr]; n=len(Y); stress=np.c_[hg,pg]; X=fg[:,1:]; miss=np.isnan(g['coupon_features_raw']); labs=g['lab_id']; dr=fg@theta[2:]
print('theta',theta,'drift range',dr.min(),dr.max()); print('raw diag features, target rate, hold, strain0'); print(np.c_[X,yr,hg,g['strain_0']]); print('ref moments by split')
for split in [tr,te,g]:
 print([(l,np.median(split['ref_coupons'][split['lab_id']==l].mean(0)),np.median(split['ref_coupons'][split['lab_id']==l].std(0))) for l in range(4)])
def score(name,B):
 W=np.linalg.lstsq(B,Y,rcond=None)[0]; R=Y-B@W
 pred=np.zeros_like(Y)
 for j in np.unique(labs):
  z=labs==j; pred[z]=B[z]@np.linalg.lstsq(B[~z],Y[~z],rcond=None)[0]
 print(name,'train',np.sqrt(np.mean(R**2,0))[[0,4,11,12]],'lolo',np.sqrt(np.mean((pred-Y)**2,0))[[0,4,11,12]],'row rms', np.sqrt(np.mean(R**2,1)))
 return W,R
one=np.ones(n); two=np.c_[one,hg,dr]; B=np.c_[one,hg,X]; W,R=score('hold/drift',two); print('time signatures',W); W,R=score('hold/features',B); print('time signature singular values',np.linalg.svd(W,compute_uv=False)); print('time signatures',W)
score('hold/features/history',np.c_[B,pg]); score('+missing',np.c_[B,miss[:,1:]]); score('+lab',np.c_[B,np.eye(4)[labs,1:]]); score('+strain0',np.c_[B,g['strain_0']]); score('+interaction',np.c_[B,hg*dr,dr**2,hg**2]); score('hold/drift/strain0',np.c_[two,g['strain_0']]);
# Test whether all coefficient curves share the documented branch-and-drift basis.
for ta in [taus,[0.493,2.8]]:
 ta=np.array(ta); T=np.vstack([np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]]); co=W@np.linalg.pinv(T.T); print('coefficient projection',ta,co,'error',np.linalg.norm(W-co@T.T,axis=1))
