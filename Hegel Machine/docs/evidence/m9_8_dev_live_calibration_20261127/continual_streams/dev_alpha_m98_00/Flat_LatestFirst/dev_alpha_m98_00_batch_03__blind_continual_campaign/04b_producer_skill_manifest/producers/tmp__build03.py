from pathlib import Path
exec(Path('/tmp/probe03.py').read_text().split("err('intrinsic-time")[0])
keys=('lab_id','stratum','ref_coupons','coupon_features_raw')
empty={k:a[k][:0] for k in keys}
x_diag=m._aligned_features(a,empty)
v=x_diag@b[2:]
co=(ps,h,x_diag)
D,R=m._design(tau,dt,t,co)
base=np.c_[D@b,R@b]
growth=-np.expm1(-t[:,None]/tau)
# The forcing terms are integrated rate perturbations with zero initial value.
W=v[:,None,None]*(growth-t[:,None]/tau)[None,:,:]
q=y[:,:-1]-base[:,:-1]
def fit(keep):
 return np.linalg.lstsq(W[keep].reshape(-1,2),q[keep].ravel(),rcond=None)[0]
forcing=fit(np.ones(len(y),bool))
print('calibrated forcing:',forcing)
for j in np.unique(labs):
 keep=labs!=j;c=fit(keep)
 print('held-out diagnostic lab',j,'forcing',c,'strain/rate RMSE',np.sqrt(np.mean((W[~keep]@c-q[~keep])**2)),np.sqrt(np.mean((base[~keep,-1]-y[~keep,-1])**2)))
source=Path('/tmp/hegel_continual_archive/dev_alpha_m98_00_batch_02_solver.py').read_text()
source='"""Estimate the material spectrum and predict retained-memory chamber response."""\n\n'+source[source.index('import numpy as np'):]
start=source.index('_APPARATUS_LOG')
end=source.index('def solve_batch(',start)
source=source[:start]+f'''# Chamber forcing estimated on the public verification coupons, with diagnostic-local
# reference alignment. Intrinsic parameters and bridge are fitted from each dataset.
_CHAMBER_FORCING = np.array({forcing.tolist()!r})


def _deployment(taus, theta, dt, times, coupon):
    curve, rate = _design(taus, dt, times, coupon)
    drift = coupon[2] @ theta[2:]
    growth = -np.expm1(-times[:, None] / taus)
    correction = (growth - times[:, None] / taus) @ _CHAMBER_FORCING
    return curve @ theta + drift[:, None] * correction, rate @ theta


'''+source[end:]
Path('/holdout_task/batch_solver.py').write_text(source)
