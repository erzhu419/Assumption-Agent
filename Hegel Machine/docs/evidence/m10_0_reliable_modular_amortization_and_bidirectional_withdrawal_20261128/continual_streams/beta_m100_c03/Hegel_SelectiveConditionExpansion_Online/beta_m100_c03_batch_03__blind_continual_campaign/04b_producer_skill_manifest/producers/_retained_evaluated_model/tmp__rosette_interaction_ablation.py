import pickle
import sys

import numpy as np

sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from upstream_transducer import _design, _fit, _invert, _response

for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    study = item['study']
    sweeps = study['sensor_calibration_sweeps']
    x, y = (np.asarray(sweeps[k]) for k in ('ref_rosette', 'raw_rosette'))
    design = _design(x)
    mixed = np.flatnonzero(np.count_nonzero(x, axis=1) > 1)
    # Score the same held-out rows for all three models. Screen gross EMI
    # using the training response only, independently of the ablations.
    mixed = mixed[np.argsort(np.linalg.norm(x[mixed], axis=1))]
    peaks = np.zeros(3)
    inverse_peak = 0.0
    counts = []
    for test in np.array_split(mixed, 4):
        tr = np.setdiff1d(np.arange(len(x)), test)
        full, good = _response(design[tr], y[tr])
        tr_clean = tr[good]
        err_train = np.linalg.norm(design[tr_clean] @ full - y[tr_clean], axis=1)
        gate = max(0.01, 20 * np.median(err_train))
        test = test[np.linalg.norm(design[test] @ full - y[test], axis=1) < gate]
        counts.append(len(test))
        assert len(test) >= 12, 'too few accepted held-out mixed sweeps'
        candidate = _fit(x[tr], y[tr])
        diagonal_only = np.r_[0:5, 5, 9, 12, 14, 15:19]
        preds = []
        for terms in (np.arange(5), diagonal_only):
            coef = np.linalg.lstsq(design[tr_clean][:, terms], y[tr_clean], rcond=None)[0]
            preds.append(design[test][:, terms] @ coef)
        preds.append(design[test] @ candidate)
        errors = np.array([np.max(np.abs(p - y[test])) for p in preds])
        peaks = np.maximum(peaks, errors)
        inverse_peak = max(inverse_peak, np.max(np.abs(_invert(y[test], candidate) - x[test])))
    print('accepted mixed rows by amplitude band:', counts)
    print('worst forward absolute error [affine, no cross-products, saved candidate]:', peaks)
    print('worst saved-candidate inverse absolute error:', inverse_peak)
    assert peaks[-1] < 1e-4 and inverse_peak < 1e-4, 'saved candidate regressed on amplitude-block holdout'
    assert peaks[-1] < 0.01 * min(peaks[:2]), 'interaction ablation did not establish a material gain'
    result = solve_batch(study)
    assert np.isfinite(result['terminal_field']).all()
