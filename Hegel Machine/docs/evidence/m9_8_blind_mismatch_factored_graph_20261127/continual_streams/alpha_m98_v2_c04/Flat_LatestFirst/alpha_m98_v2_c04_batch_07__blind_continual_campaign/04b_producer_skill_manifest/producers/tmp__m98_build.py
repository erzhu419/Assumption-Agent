from m98_chamber import *
from pathlib import Path

# The diagnostic panel is a separate coupon cohort. The material fit and
# production preprocessing use the prescribed joint train/deployment panel.
X = m._features(tr, te)
tau, th = m._fit(X[:n], *m._loads(tr, B), dt, t,
                 tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs'])
xdiag = m._features(q, q)[:len(h)]
ddiag = th[2] + xdiag @ th[3:]

def design(logtau):
    tt = np.exp(logtau)
    g0, g1 = h-z(tt[0]), h-z(tt[1])
    b0 = np.r_[-np.expm1(-t/tt[0]), 1/tt[0]]
    b1 = np.r_[-np.expm1(-t/tt[1]), 1/tt[1]]
    bt = np.r_[t, 1.]
    return np.concatenate([g0[:,None,None]*b0[None,:,None],
                           np.c_[g1,ddiag][:,None,:]*b1[None,:,None],
                           np.c_[one,h,ddiag][:,None,:]*bt[None,:,None]], axis=2)

def fit(take):
    p = np.log([.43,1.1]); a = design(p)[take].reshape(-1,6)
    coef = np.linalg.lstsq(a,y[take].ravel(),rcond=None)[0]
    fit = least_squares(lambda v: ((design(v[:2])@v[2:]-y)[take]).ravel(),
                        np.r_[p,coef],loss='soft_l1',f_scale=.012,
                        bounds=(np.r_[np.log([.25,.65]),np.full(6,-np.inf)],
                                np.r_[np.log([.65,8.]),np.full(6,np.inf)]))
    return fit.x

def errors(err):
    return np.round(np.r_[np.median(abs(err[:,:-1])), np.median(abs(err[:,-1])),np.quantile(abs(err[:,:-1]),.9),np.quantile(abs(err[:,-1]),.9)],5)

for key in ['lab_id','stratum']:
    cv = np.zeros_like(y)
    for val in np.unique(q[key]):
        out = q[key]==val
        v=fit(~out); cv[out]=(design(v[:2])@v[2:]-y)[out]
        print(key,val,errors(cv[out]))
    print(key,'OOF',errors(cv))
par = fit(np.ones(len(h),bool))
scales = np.exp(par[:2])/tau
amp_scales = par[2:4]/th[:2]
other = par[4:]
print('intrinsic:',tau,th,'chamber scales:',scales,amp_scales,other)
print('fit:',errors(design(par[:2])@par[2:]-y))
for label, xx in [('train',X[:n]),('deployment',X[n:]),('diagnostic',xdiag)]:
    print(label,'drift quantiles',np.round(np.quantile(th[2]+xx@th[3:],[0,.1,.5,.9,1]),4))
base = Path('/tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_06_solver.py').read_text().split('\ndef _chamber(')[0].split('\ndef solve_batch(')[0]
for name in ['_features','_fit_bridge','_loads','_design','_fit']:
    assert ('def '+name+'(') in base
body = '''

# Fitted from chamber verification coupons. Memory and hold response in
# each branch share a chamber-adjusted time constant.
_CHAMBER_TAU_SCALE = np.array({tau_scale})
_CHAMBER_WEIGHT_SCALE = np.array({amp_scale})
_SLOW_DRIFT_COUPLING, _DRIFT_OFFSET, _HOLD_DRIFT, _MATERIAL_DRIFT = {other}


def _predict_chamber(x, past, hold, dt, times, tau, theta):
    effective = tau * _CHAMBER_TAU_SCALE
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = -np.expm1(-dt[:, None] / effective) * np.exp(-remaining[:, None] / effective)
    gaps = hold[:, None] - past @ memory
    drift = theta[2] + x @ theta[3:]
    amplitudes = gaps * (theta[:2] * _CHAMBER_WEIGHT_SCALE)
    amplitudes[:, 1] += _SLOW_DRIFT_COUPLING * drift
    steady = _DRIFT_OFFSET + _HOLD_DRIFT * hold + _MATERIAL_DRIFT * drift
    growth = -np.expm1(-times[:, None] / effective)
    return amplitudes @ growth.T + steady[:, None] * times, amplitudes @ (1 / effective) + steady


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x = _features(train, test)
    n_train = len(train['lab_id'])
    bridge = _fit_bridge(dataset['calibration_load_cells'])
    dt = np.asarray(dataset['past_dt'])
    times = np.asarray(dataset['eval_times'])
    past, hold = _loads(train, bridge)
    measured = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, theta = _fit(x[:n_train], past, hold, dt, times, measured, train['instant_rate_obs'])
    past, hold = _loads(test, bridge)
    change, rate = _predict_chamber(x[n_train:], past, hold, dt, times, tau, theta)
    return {{'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + change,
            'initial_rate_pred': rate}}
'''
Path('/holdout_task/batch_solver.py').write_text(base.rstrip()+body.format(tau_scale=scales.tolist(),amp_scale=amp_scales.tolist(),other=tuple(other.tolist())),encoding='utf-8')
