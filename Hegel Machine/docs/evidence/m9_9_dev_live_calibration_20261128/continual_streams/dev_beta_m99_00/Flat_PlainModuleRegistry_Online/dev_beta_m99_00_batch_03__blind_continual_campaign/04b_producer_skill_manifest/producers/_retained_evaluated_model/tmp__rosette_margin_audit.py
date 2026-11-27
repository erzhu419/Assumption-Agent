import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    study = item['study']
    sweeps = study['sensor_calibration_sweeps']
    x, y = (sweeps[k] for k in ('ref_rosette', 'raw_rosette'))
    full = up._fit(x, y)
    clean = np.linalg.norm(up._basis(x) @ full - y, axis=1) < 0.01
    bank = sweeps['single_bank_sweeps']
    ids = np.asarray(bank['bank_id'])
    bx = np.eye(4)[ids] * np.asarray(bank['ref_signal'])[:, None]
    predicted = (up._basis(bx) @ full)[np.arange(len(bx)), ids]
    err = abs(predicted - bank['raw_readout'])
    maxima = [float(err[(ids == i) & (err < .01)].max()) for i in range(4)]
    print('Independent clean single-bank max error by channel:', maxima)
    assert max(maxima) < 2e-4
    # Compare the actual delivered inverse with the same fit after removing the observed EMI rows.
    clean_fit = up._fit(x[clean], y[clean])
    obs = study['y_obs_raw']
    shift = float(np.max(abs(up._invert(obs, full) - up._invert(obs, clean_fit))))
    print('Observation coordinate shift after removing measured EMI rows:', shift)
    assert shift < 2e-5
    # Signed channel bias and curvature omission are evaluated on the same held-out rows.
    rows = np.flatnonzero(clean)
    folds = np.array_split(np.random.default_rng(18).permutation(rows), 5)
    errors, quad_errors = [], []
    a = up._basis(x)
    for test in folds:
        train = np.setdiff1d(rows, test)
        coef = up._fit(x[train], y[train])
        errors.append(up._invert(y[test], coef) - x[test])
        q = np.linalg.lstsq(a[train, :15], y[train], rcond=None)[0]
        quad_errors.extend(np.linalg.norm(a[test, :15] @ q - y[test], axis=1))
    e = np.concatenate(errors)
    worst = float(np.linalg.norm(e, axis=1).max())
    q90 = float(np.percentile(quad_errors, 90))
    print('ng_t04 comparison: quadratic p90 forward error:', q90, 'cubic worst inverse error:', worst, 'mean signed error by channel:', e.mean(axis=0).tolist())
    assert worst < 2e-4 and q90 > 10 * np.percentile(np.linalg.norm(e, axis=1), 90)
    out = solve_batch(study)
    cert = out['kkt_certificate']
    gap = abs(cert['primal_objective'] - cert['dual_lower_bound'])
    assert np.isfinite(out['terminal_field']).all() and gap < 1e-9
    print('Isolated composed solver KKT gap:', gap)
