exec(open('/tmp/analyze_chamber.py').read().split('for s,xx in ')[0])
from downstream_solver import _chamber_response
_, own = _features(dg, dg)
raw = dg['coupon_features_raw']; missing = ~np.isfinite(raw)
# Check whether the discrepancy on calibration coupons with missing entries
# is explained by a different imputation pool, without changing deployment preprocessing.
raw_aligned = np.where(missing, np.nan, own)
y = _response(dg)
for label, v in [('diagnostic balanced', own), ('diagnostic unweighted', np.where(missing, np.nanmean(raw_aligned, axis=0), own)), ('deployment pool', np.where(missing, xd, own)), ('train plus diagnostic', np.where(missing, _features(tr, dg)[1], own))]:
    s,r = _chamber_response(dg,v,t,tau,th)
    e = np.column_stack((s,r))-y
    print(label, 'imputations', v[missing].round(4), 'missing p90 strain/rate', np.quantile(abs(e[missing.any(1)]),.9,axis=0)[[len(t)-1,len(t)]].round(5))
print('sample counts train/test/diagnostic', *(len(p['lab_id']) for p in (tr,te,dg)))
