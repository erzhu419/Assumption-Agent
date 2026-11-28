from pathlib import Path
import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = item['dataset']
q = item['campaign_diagnostics']
spec = importlib.util.spec_from_file_location('base', '/tmp/m100_base.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
a = b.solve_batch(d)
tau, theta = a['estimated_taus'], a['hereditary_weights']
t = d['eval_times']
bridge = b._bridge(d['calibration_load_cells'])
_, hold = b._loads(q, bridge)
score = theta[2] + b._features(q, q)[:len(hold)] @ theta[3:]
observed = np.column_stack((q['instant_rate_obs'], q['strain_curves_obs'] - q['strain_0'][:, None]))
kernels = np.column_stack((1 / tau, -np.expm1(-t / tau[:, None])))
hold_term = hold[:, None] * (theta[:2] @ kernels)
complete = np.isfinite(q['coupon_features_raw']).all(axis=1)

def fit(mask):
    # Fix the shared branch times and constrain the feature kernel's initial slope to one.
    design = (score[:, None, None] * np.stack((tau[0] * kernels[0] - tau[1] * kernels[1], np.r_[1., t] - tau[1] * kernels[1]), axis=-1)[None, :, :])
    offset = hold_term + score[:, None] * tau[1] * kernels[1]
    weights = np.linalg.lstsq(design[mask].reshape(-1, 2), (observed - offset)[mask].ravel(), rcond=None)[0]
    return weights, offset + design @ weights

fold_metrics = []
for lab in np.unique(q['lab_id']):
    test = q['lab_id'] == lab
    _, pred = fit(complete & ~test)
    err = pred[test] - observed[test]
    fold_metrics.append([np.sqrt(np.mean(err[:, 0]**2)), np.sqrt(np.mean(err[:, 1:]**2)), np.max(np.sqrt(np.mean(err[:, 1:]**2, axis=0)))])
w, pred = fit(complete)
err = pred - observed
print('worst held-out lab rate / curve / single-time RMSE', np.max(fold_metrics, axis=0))
print('complete / imputed rate and curve RMSE', [(np.sqrt(np.mean(err[m, 0]**2)), np.sqrt(np.mean(err[m, 1:]**2))) for m in (complete, ~complete)])
print('feature kernel slope fractions (fast, slow, linear)', w[0], 1-w.sum(), w[1])
assert np.all(np.max(fold_metrics, axis=0) < [.015, .02, .025])
source = Path('/tmp/m100_base.py').read_text().split('def solve_batch', 1)[0]
source = source.replace('Estimate a two-branch hereditary response from the benchtop coupons.', 'Fit the intrinsic spectrum on benchtop coupons and predict the calibrated chamber response.')
path = Path('/holdout_task/batch_solver.py')
path.write_text(source + f'''# Initial-slope fractions identified on complete public chamber coupons.
_CHAMBER_FAST = {w[0]:.10f}
_CHAMBER_LINEAR = {w[1]:.10f}


def _chamber(taus, theta, hold, features, times):
    slow = 1.0 - _CHAMBER_FAST - _CHAMBER_LINEAR
    fractions = np.array([_CHAMBER_FAST, slow])
    kernel = -np.expm1(-times[None, :] / taus[:, None])
    score = theta[2] + features @ theta[3:]
    stress_profile = theta[:2] @ kernel
    feature_profile = (fractions * taus) @ kernel + _CHAMBER_LINEAR * times
    increments = hold[:, None] * stress_profile + score[:, None] * feature_profile
    rate = hold * (theta[:2] @ (1.0 / taus)) + score
    return increments, rate


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    features = _features(train, test)
    bridge = _bridge(dataset['calibration_load_cells'])
    dt, times = (np.asarray(dataset[key]) for key in ('past_dt', 'eval_times'))
    taus, theta = _fit(*_loads(train, bridge), features[:n], dt, times,
                       np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None],
                       np.asarray(train['instant_rate_obs']))
    _, hold = _loads(test, bridge)
    # The chamber uses its own reference panel, independently of the intrinsic fit.
    chamber_features = _features(test, test)[:len(hold)]
    increments, rate = _chamber(taus, theta, hold, chamber_features, times)
    return {{'estimated_taus': taus, 'hereditary_weights': theta,
             'predicted_strain': np.asarray(test['strain_0'])[:, None] + increments,
             'initial_rate_pred': rate}}
''')
