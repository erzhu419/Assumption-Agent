from pathlib import Path
exec(Path('/tmp/rheology03_cv.py').read_text().split('for weight in')[0])
from scipy.optimize import least_squares

def mad(a):
    return 1.4826*np.median(np.abs(a-np.median(a)))

def fit_robust(keep, k):
    tau, w = s._fit(model, y, r, dt, t, row_mask=keep)
    def errors(p):
        c, rr = model(np.sort(np.exp(p[:2])))
        return (c @ p[2:] - y)[keep], (rr @ p[2:] - r)[keep]
    p0 = np.r_[np.log(tau), w]
    for step in range(3):
        e, f = errors(p0)
        sc, sr = max(mad(e), 1e-6), max(mad(f), 1e-6)
        def residual(p):
            a,b=errors(p)
            return np.r_[a.ravel()/sc, b/sr]
        lo = [np.log(min(t)/20)]*2+[-np.inf]*9
        hi = [np.log(max(sum(dt),max(t))*30)]*2+[np.inf]*9
        out=least_squares(residual, p0, bounds=(lo,hi), loss='soft_l1', f_scale=k, x_scale='jac', max_nfev=160)
        p0=out.x
    return np.sort(np.exp(p0[:2])),p0[2:],(sc,sr)

def metrics(ec,er):
    for a in (ec,er):
        print('absolute error q50/90/95/98/99/100:',np.round(np.quantile(np.abs(a),[.5,.9,.95,.98,.99,1.]),5))
    for name, mask in slices.items():
        if mask.any():
            print(name,'RMSE',np.round([rmse(ec[mask]),rmse(er[mask])],6),'MAE',np.round([np.mean(np.abs(ec[mask])),np.mean(np.abs(er[mask]))],6))
    print('horizon MAE',np.round(np.mean(np.abs(ec),axis=0),6))

for k in (None, 1.5, 3.0):
    print('robust cutoff:',k)
    ec, er = np.empty_like(y), np.empty_like(r)
    for lab in range(4):
        val = labs == lab
        if k is None:
            tau,w=s._fit(model,y,r,dt,t,row_mask=~val)
            scales=()
        else:
            tau,w,scales=fit_robust(~val,k)
        c,rr=model(tau)
        ec[val],er[val]=(c@w-y)[val],(rr@w-r)[val]
        print('fold',lab,'tau',np.round(tau,5),'weights',np.round(w,5),'scales',scales)
    metrics(ec,er)
    print('largest curve errors (row, horizon), signed residual:',[(tuple(i),round(ec[tuple(i)],4)) for i in np.argwhere(np.abs(ec)>3)])
