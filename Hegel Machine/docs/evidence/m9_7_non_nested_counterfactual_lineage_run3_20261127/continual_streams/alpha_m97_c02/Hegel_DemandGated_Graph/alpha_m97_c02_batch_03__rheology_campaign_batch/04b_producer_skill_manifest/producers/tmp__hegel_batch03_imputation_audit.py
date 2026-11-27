exec(open('/tmp/hegel_batch03_cv.py').read().split('for l in [-1,0,1,2,3]:')[0])
raw = np.concatenate([tr['coupon_features_raw'], te['coupon_features_raw']])
labs = np.concatenate([lab, te['lab_id']])
strata = np.concatenate([tr['stratum'], te['stratum']])
missing = np.isnan(raw[:len(lab)])
scores = {k: [] for k in ('cell_balanced', 'row_weighted')}
for l in np.unique(lab):
    f = lab != l
    q = (lab == l) & missing.any(axis=1)
    if not q.any():
        continue
    taus, w = m._fit(p[f], h[f], x[f], dt, times, y[f], r[f])
    observed = np.where(np.isfinite(raw), m._align(tr, te), np.nan)
    row_average = np.nanmean(observed, axis=0)
    for name, xx in [('cell_balanced', x[q]), ('row_weighted', np.where(missing[q], row_average, x[q]))]:
        a, b = m._design(p[q], h[q], xx, dt, times, taus)
        dy, dr = a @ w-y[q], b @ w-r[q]
        scores[name].append((dy, dr))
        print('lab', l, name, 'n', q.sum(), 'strain median AE', round(float(np.median(abs(dy))), 5), 'rate RMSE', round(rmse(dr), 5), 'signed rate median', round(float(np.median(dr)), 5))
print('Missing-feature donors per column by lab and stratum:')
for l, s in np.unique(np.column_stack((labs, strata)), axis=0):
    donor = np.isfinite(raw[(labs == l) & (strata == s)])
    print(l, s, len(donor), donor.sum(axis=0).tolist())
for name, pairs in scores.items():
    dy, dr = [np.concatenate(v) for v in zip(*pairs)]
    print(name, 'OOF missing-coupon strain median AE', round(float(np.median(abs(dy))), 5), 'rate RMSE', round(rmse(dr), 5))
