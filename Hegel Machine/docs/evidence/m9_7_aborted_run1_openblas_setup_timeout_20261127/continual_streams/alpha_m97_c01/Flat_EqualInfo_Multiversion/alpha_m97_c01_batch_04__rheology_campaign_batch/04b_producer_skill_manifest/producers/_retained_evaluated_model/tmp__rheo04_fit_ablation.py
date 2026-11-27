import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p+'alpha_m97_c01_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(p+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]['dataset']
tr = d['train']
lab = tr['lab_id']
n = len(lab)
s = tuple(v[:n] for v in m._prepare(d))
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)

def fit_and_predict(fit, test):
    tau, w = m._fit(tuple(z[fit] for z in s), dt, t, y[fit], r[fit])
    a, b = m._design(tau, tuple(z[test] for z in s), dt, t)
    return tau, a@w-y[test], b@w-r[test]

for complete_fit in (False, True):
    e, f = np.empty_like(y), np.empty_like(r)
    taus = []
    for c in np.unique(lab):
        test = lab == c
        fit = ~test & (complete if complete_fit else True)
        tau, e[test], f[test] = fit_and_predict(fit, test)
        taus.append(tau)
    print('complete-only fit' if complete_fit else 'current all-row fit')
    for label, select in [('complete', complete), ('missing', ~complete), ('zero damage', s[4]==0), ('active', s[4]>0)] + [(f'complete lab {c}', complete & (lab==c)) for c in np.unique(lab)]:
        print(label, int(select.sum()), 'late-horizon median-abs max / rate RMSE', round(np.nanmedian(abs(e[select,-3:]), axis=0).max(),6), round(np.sqrt(np.nanmean(f[select]**2)),6))
    print('fold taus', np.round(taus,6).tolist())
