import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from explore import *
from base import _features
from itertools import product
bb=np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1]])
y=np.c_[qc,rc]-hc[:,None]*(bb[:,:2]@theta[:2]);
def aligned(ref_parts, missing):
 vals=ch['coupon_features_raw'].copy()
 for lab in np.unique(ch['lab_id']):
  panel=np.concatenate([g['ref_coupons'][g['lab_id']==lab] for g in ref_parts]); m=ch['lab_id']==lab
  vals[m]=(vals[m]-np.median(panel.mean(0)))/np.median(panel.std(0))
 vals=np.where(np.isfinite(vals),vals,missing)
 return vals
pooled=_features(tr,te); impute=np.array([pooled[np.flatnonzero(np.isnan(np.r_[tr['coupon_features_raw'],te['coupon_features_raw']][:,j]))[0],j] for j in range(4)])
print('diag fields',list(ch),'raw feature missing',np.isnan(ch['coupon_features_raw']).sum(0),'deployment impute',impute)
for g in [tr,te,ch]:
 a=g.get('aux_chamber_telemetry');print('aux',None if a is None else (a.shape,a[:2],a.std(0)))

def evaluate(m,y=y):
 q=np.linalg.qr(m,mode='reduced')[0]; z=lstsq(m,y)[0]; er=(y-m@z)/(1-np.sum(q*q,axis=1))[:,None]
 return np.array([rms(er[:,:-1]),max(rms(er[:,:-1],0)),rms(er[:,-1]),max(abs(er[:,-1]))]),z
for name,v in [('tr+ch',xc),('tr+te',aligned([tr,te],impute)),('tr',aligned([tr],impute)),('ch',aligned([ch],impute)),('all',aligned([tr,te,ch],impute))]:
 rr=theta[2]+v@theta[3:];print(name,np.max(abs(v-xc)))
 for model,m in [('linear',np.c_[np.ones(len(rr)),rr]),('quadratic',np.c_[np.ones(len(rr)),rr,rr*rr]),('four features',np.c_[np.ones(len(rr)),v])]:
  e,b=evaluate(m);print(model,e,'coef basis',lstsq(bb,b.T)[0].T.round(5))
 if name=='tr+te':
  ee=y-np.c_[np.ones(len(rr)),rr]@lstsq(np.c_[np.ones(len(rr)),rr],y)[0];print('diagnostic residual by row',np.c_[ch['lab_id'],ch['stratum'],np.isnan(ch['coupon_features_raw']).sum(1),rr,ee[:,0],ee[:,-2],ee[:,-1]])
