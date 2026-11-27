exec(open('/tmp/hegel_batch03_cv.py').read().split('for l in [-1,0,1,2,3]:')[0])
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
ey, er = np.empty_like(y), np.empty_like(r)
for l in np.unique(lab):
    f = lab != l
    taus, w = m._fit(p[f], h[f], x[f], dt, times, y[f], r[f])
    a, b = m._design(p[~f], h[~f], x[~f], dt, times, taus)
    ey[~f], er[~f] = a @ w - y[~f], b @ w - r[~f]
print('regime index:', d['shear_banding_index'])
print('complete coupons: signed held-out median residual, by lab and stress sign')
for l in np.unique(lab):
    for sign in (-1, 1):
        q = (~missing) & (lab == l) & (np.sign(h) == sign)
        if q.any():
            print(l, sign, int(q.sum()), 'strain', round(float(np.median(ey[q])), 5), 'rate', round(float(np.median(er[q])), 5))
print('complete-coupon held-out strain median absolute error by horizon:', np.round(np.median(abs(ey[~missing]), axis=0), 5))
sweep = d['calibration_load_cells']
for l in range(4):
    cx, cy = m._points(sweep, l)
    g, b = cal[l]
    deviation = [m._affine(np.delete(cx,i), np.delete(cy,i)) - cal[l] for i in range(len(cx))]
    print('calibration lab', l, 'max stress residual', float(np.max(abs(cy-g*cx-b))), 'max single-point gain/bias influence', np.max(abs(np.asarray(deviation)), axis=0))
