import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from transfer import *
from base import _features
v=_features(ch,{k:ch[k][:0] for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}); r=theta[2]+v@theta[3:];m=np.c_[np.ones(len(r)),r]; a=lstsq(m,lstsq(bb,y.T)[0].T)[0];print('local LOO',evaluate(m)[0]);print('coeff',a);print('score range train-test vs diag',[(rr.min(),rr.max()) for rr in [r,theta[2]+pooled[len(tr['lab_id']):]@theta[3:]]]);
for j in np.unique(ch['lab_id']):
 f=ch['lab_id']==j; e=y[f]-m[f]@lstsq(m[~f],y[~f])[0];print('LOLO',j,rms(e[:,:-1]),max(rms(e[:,:-1],0)),rms(e[:,-1]),max(abs(e[:,-1])))
for j in np.unique(ch['stratum']):
 f=ch['stratum']==j; e=y[f]-m[f]@lstsq(m[~f],y[~f])[0];print('LOSO',j,rms(e[:,:-1]),rms(e[:,-1]))
for g in [tr,te,ch]:
 empty={k:g[k][:0] for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; print('panel score deviation',rms((_features(g,empty)- (pooled[:len(tr['lab_id'])] if g is tr else pooled[len(tr['lab_id']):] if g is te else v))@theta[3:]));
print('incomplete residual', (y-m@lstsq(m,y)[0])[~np.isfinite(ch['coupon_features_raw']).all(1),-1]);print('test raw missing',np.isnan(te['coupon_features_raw']).sum(0))
