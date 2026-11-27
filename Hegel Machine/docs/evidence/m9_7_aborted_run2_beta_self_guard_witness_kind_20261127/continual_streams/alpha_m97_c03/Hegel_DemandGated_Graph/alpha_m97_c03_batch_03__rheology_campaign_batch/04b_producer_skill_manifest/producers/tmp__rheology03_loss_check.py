from pathlib import Path
exec(Path('/tmp/rheology03_robust.py').read_text().split('for k in (None,')[0])

def cauchy_refine(keep, tau, w, scales):
    p0 = np.r_[np.log(tau), w]
    sc, sr = scales
    def residual(p):
        c, rr = model(np.sort(np.exp(p[:2])))
        return np.r_[((c @ p[2:] - y)[keep]/sc).ravel(), (rr @ p[2:] - r)[keep]/sr]
    lo = [np.log(min(t)/20)]*2 + [-np.inf]*9
    hi = [np.log(max(sum(dt),max(t))*30)]*2 + [np.inf]*9
    out = least_squares(residual, p0, bounds=(lo,hi), loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100)
    return np.sort(np.exp(out.x[:2])), out.x[2:]

results = {name:(np.empty_like(y),np.empty_like(r),[]) for name in ('soft_l1','cauchy')}
for lab in range(4):
    val = labs == lab
    tau, w, scales = fit_robust(~val, 1.5)
    for name, (a,b) in zip(results, [(tau,w),cauchy_refine(~val,tau,w,scales)]):
        ec, er, folds = results[name]
        c, rr = model(a)
        ec[val], er[val] = (c @ b-y)[val], (rr @ b-r)[val]
        folds.append(np.r_[a,b])
for name,(ec,er,folds) in results.items():
    print(name,'fold estimates',np.round(folds,5).tolist())
    for label, values in [('curve',ec),('rate',er)]:
        print(label,'worst slice RMSE',max((rmse(values[m]),k) for k,m in slices.items() if m.any()),'worst slice MedAE',max((np.median(np.abs(values[m])),k) for k,m in slices.items() if m.any()),'complete q50/q90',np.quantile(np.abs(values[~missing]),[.5,.9]))
    print('worst horizon MedAE', np.max(np.median(np.abs(ec),axis=0)))
