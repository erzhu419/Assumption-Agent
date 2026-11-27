import contextlib, io
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/chamber_cv.py').read().split('for name,x in stds.items():')[0])
x = stds['diag']
drift = th[2] + x @ th[3:]
basis = np.vstack([B, R])
yobs = np.c_[y, v]
base = (hold[:, None] * th[:2]) @ basis[:, :2].T
w = np.r_[np.ones(len(t)), .65]
contrast = z[:, 0] - z[:, 1]
q = drift / np.std(drift)
models = {'fixed': None, 'linear': drift[:, None], 'contrast': np.c_[drift, contrast], 'pair_product': np.c_[drift, x[:, 0]*x[:, 1], x[:, 2]*x[:, 3]], 'triple_product': np.c_[drift, x[:, 0]*x[:, 1]*x[:, 2]], 'rational': np.c_[drift, drift/(1+q*q)], 'lab_shift_control': np.c_[drift, np.eye(len(np.unique(labs)))[labs]]}
for name, f in models.items():
    if f is not None:
        design = (basis[None, :, :, None] * f[:, None, None, :]).reshape(len(y), len(w), -1)
    e = np.zeros_like(yobs)
    for lab in np.unique(labs):
        a = labs != lab
        if f is None:
            e[~a] = yobs[~a] - base[~a] - (drift[~a,None]*[.25,1.8,.03]) @ basis.T
        else:
            coef = np.linalg.lstsq((design[a]*w[None,:,None]).reshape(-1, design.shape[-1]), ((yobs-base)[a]*w).ravel(), rcond=None)[0]
            e[~a] = yobs[~a] - base[~a] - design[~a] @ coef
    def score(a):
        return [np.sqrt(np.mean(e[a,:-1]**2)), np.sqrt(np.mean(e[a,-1]**2))]
    miss = np.isnan(g['coupon_features_raw']).any(axis=1)
    print(name, 'worst lab', np.max([score(labs==l) for l in np.unique(labs)],0).round(6), 'complete/missing', np.round([score(~miss), score(miss)],6).tolist(), 'worst lab absolute rate bias', round(max(abs(e[labs==l,-1].mean()) for l in np.unique(labs)),6))
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; true=cal['true_channel_loads'].sum(1)
X=np.c_[np.ones(len(raw)),raw]; Q=np.c_[X, *(raw[:,j]*raw[:,k] for j in range(4) for k in range(j,4))]
for label, a in [('affine',X),('quadratic',Q)]:
    e=[]
    for part in np.array_split(np.argsort(true),5):
        mask=np.ones(len(raw),bool); mask[part]=False
        er=a[part]@np.linalg.lstsq(a[mask],true[mask],rcond=None)[0]-true[part]
        e.append(np.sqrt(np.mean(er**2)))
    print('bridge',label,'held-out stress quintile RMSE',np.round(e,8).tolist())
