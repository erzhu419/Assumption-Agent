exec(open('/tmp/m99_joint.py').read().split('models={')[0])
D = np.c_[one, m._features(di)] @ th[2:]
complete = np.isfinite(di['coupon_features_raw']).all(axis=1)
rows = np.arange(N)
for name, A, eligible in [('saved structure', F(one,h,D), rows), ('complete-only calibration', F(one,h,D), rows[complete]), ('complete-only without offset', F(h,D), rows[complete])]:
    residual = np.empty_like(Y)
    for g in groups:
        residual[g] = fit((A,)*3, np.setdiff1d(eligible,g))[g]
    print(name, 'complete', fmt(residual[complete]), 'imputed', fmt(residual[~complete]), 'worst lab', np.max([stats(residual[g]) for g in groups],axis=0).round(5), 'late strain RMSE', round(np.sqrt(np.mean(residual[:,-4:-1]**2)),5))
    fit((A,)*3, eligible, print_co=True)
