import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _basis, _fit, _invert, _TERMS, calibrate_upstream
from _verified_downstream import solve_downstream

with open('/holdout_task/public_batch.pkl', 'rb') as stream:
    studies = [item['study'] for item in pickle.load(stream)]
keep = np.array([True] + [len(set(t)) == 1 for t in _TERMS])
for st in studies:
    cal = st['sensor_calibration_sweeps']
    x, y = cal['ref_rosette'], cal['raw_rosette']
    a = _basis(x)
    blocks = np.arange(len(x)) // 10
    baseline = calibrate_upstream(st)
    w0 = solve_downstream(baseline)['weights']
    measurements = []
    for fold in np.unique(blocks):
        train = blocks != fold
        mixed = _fit(x[train], y[train])
        # Gross EMI is separated by orders of magnitude; use the same rows for both models.
        clean = np.linalg.norm(a @ mixed - y, axis=1) < .01
        test = ~train & clean
        fit_rows = train & clean
        additive = np.linalg.lstsq(a[fit_rows][:, keep], y[fit_rows], rcond=None)[0]
        rms = lambda z: np.sqrt(np.mean(z*z))
        study = dict(baseline, y_obs_raw=_invert(st['y_obs_raw'], mixed))
        weights = solve_downstream(study)['weights']
        row = [rms(a[test][:, keep] @ additive - y[test]), rms(a[test] @ mixed - y[test]), rms(_invert(y[test], mixed) - x[test]), np.max(np.abs(weights-w0))]
        measurements.append(row)
        print('block', fold, 'EMI rows', int(np.count_nonzero(~train & ~clean)), 'additive forward / mixed forward / mixed inverse / weight drift:', *row)
    worst = np.max(measurements, axis=0)
    print('worst across blocks:', *worst)
    assert worst[1] < 1e-4 and worst[2] < 1e-4 and worst[0] > 100 * worst[1]
    b = cal['single_bank_sweeps']
    j = np.asarray(b['bank_id'], dtype=int)
    amp = b['ref_signal']
    axes = np.eye(4)[j] * amp[:, None]
    single_errors = []
    for bank in range(4):
        # Hold out all corresponding axis-only rosette rows before checking the independent bank sweep.
        other = np.arange(4) != bank
        train = np.any(x[:, other] != 0, axis=1)
        c = _fit(x[train], y[train])
        rows = j == bank
        err = (_basis(axes[rows]) @ c)[:, bank] - b['raw_readout'][rows]
        clean = abs(err) < .01
        single_errors.extend(err[clean])
        print('independent bank', bank, 'retained', clean.sum(), '/', len(err), 'RMSE', rms(err[clean]))
    print('independent bank sweep RMSE:', rms(np.asarray(single_errors)))
