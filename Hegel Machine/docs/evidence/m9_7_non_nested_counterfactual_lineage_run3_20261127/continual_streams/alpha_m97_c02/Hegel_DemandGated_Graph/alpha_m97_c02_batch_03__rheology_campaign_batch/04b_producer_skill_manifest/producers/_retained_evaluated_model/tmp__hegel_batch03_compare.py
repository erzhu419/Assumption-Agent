exec(open('/tmp/hegel_batch03_cv.py').read().split('for l in [-1,0,1,2,3]:')[0])
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for name, usable in [('robust_all', np.ones(len(lab), bool)), ('robust_complete', ~missing)]:
    ey, er = np.empty_like(y), np.empty_like(r)
    coeff = []
    for l in range(4):
        f, q = (lab != l) & usable, lab == l
        taus, w = m._fit(p[f], h[f], x[f], dt, times, y[f], r[f])
        coeff.append(np.r_[taus,w]); a,b = m._design(p[q],h[q],x[q],dt,times,taus)
        ey[q], er[q] = a@w-y[q], b@w-r[q]
    print(name, 'fold parameter range', np.ptp(coeff,axis=0).round(5))
    for label,q in [('complete',~missing),('missing',missing),('high_stress',abs(h) >= np.quantile(abs(h),.9)), *[(f'lab{i}',lab==i) for i in range(4)]]:
        yy,rr=abs(ey[q]),abs(er[q]); print(label, 'strain median/p90, rate median/p90/RMSE', np.r_[np.quantile(yy,[.5,.9]),np.quantile(rr,[.5,.9]),rmse(rr)].round(5))
    print('curve residuals > 0.5, complete/missing:', np.count_nonzero(abs(ey[~missing])>.5), np.count_nonzero(abs(ey[missing])>.5), 'out of', ey.size)
    print('high residual counts by time',np.sum(abs(ey)>.5,axis=0))
