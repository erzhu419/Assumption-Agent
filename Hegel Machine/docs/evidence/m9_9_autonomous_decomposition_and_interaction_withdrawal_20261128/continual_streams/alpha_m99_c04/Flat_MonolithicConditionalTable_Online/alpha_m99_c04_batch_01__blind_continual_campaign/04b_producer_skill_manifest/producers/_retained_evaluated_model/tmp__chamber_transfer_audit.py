import contextlib, io
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/chamber_cv.py').read().split('for name,x in stds.items():')[0])
tr_y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
tr_v = tr['instant_rate_obs']
complete = np.isfinite(g['coupon_features_raw']).all(axis=1)
print('diagnostic complete/missing counts:', [(int(l),int(sum((labs==l)&complete)),int(sum((labs==l)&~complete))) for l in np.unique(labs)])
errors = {key: np.zeros_like(ys) for key in ('current','trimmed','horizon_scaled')}
for l in np.unique(labs):
    a = tr['lab_id'] != l
    b = labs == l
    ta, tt = m._fit({k:z[a] for k,z in tr.items()},xt[a],st,dt,t)
    def res(p):
        C,R = m._design(tr,xt,st,dt,t,np.exp(p[:2]))
        return (C@p[2:]-tr_y)[a], (R@p[2:]-tr_v)[a]
    p0=np.r_[np.log(ta),tt]; ec,er=res(p0)
    sc=m._mad(ec); sr=m._mad(er); keep=np.abs(ec-np.median(ec))<4*sc
    horizon_sc=np.clip(np.array([m._mad(z) for z in ec.T]),.75*sc,1.5*sc)
    for key in errors:
        pp=p0
        if key!='current':
            def objective(p):
                c,r=res(p)
                return np.r_[c[keep]/sc if key=='trimmed' else (c/horizon_sc).ravel(),r/sr]
            pp=least_squares(objective,p0,bounds=(np.r_[np.log([.05000001]*2),[-np.inf]*7],np.r_[np.log([24.999999]*2),[np.inf]*7]),loss='linear' if key=='trimmed' else 'cauchy',f_scale=2.5,x_scale='jac').x
        taus=np.exp(pp[:2]); theta=pp[2:]; drift=theta[2]+stds['diag'][b]@theta[3:]; amp=hold[b,None]*theta[:2]+drift[:,None]*[.25,1.8]; tail=.03*drift
        errors[key][b] = np.c_[y[b]-amp@(-np.expm1(-t[:,None]/taus)).T-tail[:,None]*t,v[b]-amp@(1/taus)-tail]
for key,e in errors.items():
    def score(a):
        return [np.sqrt(np.mean(e[a,:len(t)//2]**2)),np.sqrt(np.mean(e[a,len(t)//2:-1]**2)),np.sqrt(np.mean(e[a,-1]**2))]
    print(key,'worst lab early/late/rate',np.max([score(labs==l) for l in np.unique(labs)],0).round(6),'complete/missing',np.round([score(complete),score(~complete)],6).tolist())
