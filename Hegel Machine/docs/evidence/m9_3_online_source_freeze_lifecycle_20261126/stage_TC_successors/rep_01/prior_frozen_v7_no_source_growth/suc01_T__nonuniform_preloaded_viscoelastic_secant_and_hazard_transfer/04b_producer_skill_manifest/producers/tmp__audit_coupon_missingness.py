import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_suc01_T_datasets.pkl', 'rb'))[0]['dataset']
a, t, dt = d['train'], d['eval_times'], d['past_dt']
labs, strata = a['lab_id'], a['stratum']
missing = ~np.isfinite(a['coupon_features_raw']).all(axis=1)
errors = []
for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0):
    hold = (labs == lab) & (strata == stratum)
    fit = {k: v[~hold] for k, v in a.items()}
    test = {k: v[hold] for k, v in a.items()}
    tau, amp, drift = m._fit_physics(fit, dt, t)
    _, fit_rate = m._branch(tau, fit, dt, t)
    branch, test_rate = m._branch(tau, test, dt, t)
    train_rate = fit['instant_rate_obs'] - fit_rate @ amp
    rate = test['instant_rate_obs'] - test_rate @ amp
    fused = m._predict_drift(fit, test, drift, train_rate, rate)
    # The prediction is affine in the measured rate; recover the coupon-only endpoint.
    p0 = m._predict_drift(fit, test, drift, train_rate, np.zeros_like(rate))
    p1 = m._predict_drift(fit, test, drift, train_rate, np.ones_like(rate))
    prior = p0 / (1. - (p1 - p0))
    # An independent held-out drift proxy uses only curves, not the initial-rate observation.
    observed = test['strain_curves_obs'] - test['strain_0'][:, None] - branch @ amp
    reference = np.median(observed[:, -5:] / t[-5:], axis=1)
    errors.append(np.column_stack((np.full(sum(hold), lab), np.full(sum(hold), stratum), missing[hold], np.abs(reference[:, None] - np.column_stack((fused, rate, prior))))))
e = np.vstack(errors)
print('Held-cell drift MAE: fused / rate-only / coupon-only')
for flag in (0, 1):
    subset = e[e[:, 2] == flag]
    print('missing', flag, 'n', len(subset), 'MAE', subset[:, 3:].mean(0))
    for lab in np.unique(subset[:, 0]):
        rows = subset[subset[:, 0] == lab, 3:]
        print('  lab', int(lab), 'n', len(rows), 'MAE', rows.mean(0))
