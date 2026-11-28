exec(open('/tmp/chamber_cv.py').read().split("print('leave coupon out:") [0])
from scipy.optimize import minimize_scalar
past, _ = m._loads(d['calibration_load_cells'], g)
drift = x @ theta[2:]
lab = g['lab_id']
# Compare the baseline with hold/history contrasts, drift-dependent saturation,
# and lab-specific residuals. All extra parameter fitting is inside each fold.
for kind in ('compact','contrast','rational','lab_center'):
    err_y = np.empty_like(y)
    err_r = np.empty_like(r)
    for l in np.unique(lab):
        te = lab == l
        tr = ~te
        sig = observed_drift.copy()
        def reg(a):
            z = sig[:,None] / (1 + a*sig[:,None]**2)
            if kind == 'contrast':
                # Latest stress versus previous pulse and hold versus latest stress.
                z = np.c_[z, past[:,-1]-past[:,-2], h-past[:,-1]]
            return (z[:,None,:,None]*B[None,:,None,:]).reshape(len(h),len(t),-1)
        dy = y - anchor - sig[:,None]*t
        def fit(a):
            z = reg(a)[tr].reshape(tr.sum()*len(t),-1)
            p = np.linalg.lstsq(z,dy[tr].ravel(),rcond=None)[0]
            return np.mean((z@p-dy[tr].ravel())**2), p
        a = minimize_scalar(lambda a: fit(a)[0],bounds=(0,4),method='bounded').x if kind=='rational' else 0.
        _, p = fit(a)
        sig[te] = drift[te]
        if kind == 'lab_center':
            # Out-of-lab global offset only; unseen lab labels get no fitted offsets.
            sig[te] += np.median(observed_drift[tr]-drift[tr])
        err_y[te] = anchor[te] + sig[te,None]*t + reg(a)[te]@p - y[te]
        err_r[te] = sig[te]-observed_drift[te]
    rms=lambda q: np.sqrt(np.mean(q*q))
    print(kind,'pooled',rms(err_y),rms(err_r),'worst lab',max(rms(err_y[lab==l]) for l in np.unique(lab)),max(rms(err_r[lab==l]) for l in np.unique(lab)), 'worst horizon', max(rms(col) for col in err_y.T))
# Compare bridge polynomial terms on held-out calibration observations,
# evaluating the stress sum rather than the nuisance per-channel prediction.
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; tru=cal['true_channel_loads']; target=tru.sum(1)
z=raw-raw.mean(0); lin=np.c_[np.ones(len(z)),z]; quad=np.c_[lin,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]]
for name,A in [('affine',lin),('quadratic',quad)]:
    slices=[]
    for fold in range(5):
        test=np.arange(len(z))%5==fold
        e=A[test]@np.linalg.lstsq(A[~test],target[~test],rcond=None)[0]-target[test]
        slices.append(np.sqrt(np.mean(e**2)))
    print('calibration',name,'worst fold RMSE',max(slices))
