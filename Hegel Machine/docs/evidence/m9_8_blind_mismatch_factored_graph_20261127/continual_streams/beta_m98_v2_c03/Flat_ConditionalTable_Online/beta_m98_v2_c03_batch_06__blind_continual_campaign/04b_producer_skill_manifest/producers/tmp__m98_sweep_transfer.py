import importlib.util
import pickle
from unittest.mock import patch
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for item in pickle.load(open(root + 'public_batch.pkl', 'rb')):
    s = item['study']
    cal = s['sensor_calibration_sweeps']
    x, y = cal['ref_rosette'], cal['raw_rosette']
    clean = np.max(abs(y), axis=1) < 7
    amp = np.max(abs(x), axis=1)
    high = clean & (amp >= np.quantile(amp[clean], .8))
    sb = cal['single_bank_sweeps']
    banks, refs, readings = (np.asarray(sb[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    solo = np.zeros((len(refs), 4)); solo[np.arange(len(refs)), banks.astype(int)] = refs
    def fits(train, augmented):
        coefficients = []
        for b in range(4):
            A = m._features(x[train], b); t = y[train, b]
            if augmented:
                keep = (banks == b) & (abs(readings) < 7)
                A = np.vstack((A, m._features(solo[keep], b))); t = np.r_[t, readings[keep]]
            coefficients.append(np.linalg.lstsq(A, t, rcond=None)[0])
        return np.asarray(coefficients)
    S, _ = m._forward(s)
    baseline = m.solve_batch(s)['weights']
    for augmented in (False, True):
        outer = fits(clean & ~high, augmented)
        held = np.column_stack([m._features(x[high], b) @ outer[b] for b in range(4)]) - y[high]
        coef = fits(clean, augmented)
        pulse_errors = []
        with patch.object(m, '_rosette_coefficients', return_value=coef):
            for pulse in item.get('campaign_diagnostics') or []:
                e = m._calibrate(dict(s, y_obs_raw=pulse['y_pulse_raw'])) - S @ pulse['w_pulse']
                pulse_errors.append(float(np.sqrt(np.mean(e.reshape(len(s['mu_shifts']), -1, 4)**2, axis=1)).max()))
            w = m.solve_batch(s)['weights']
        print('augmented', augmented, 'high-amplitude held-out worst-channel RMSE', float(np.sqrt(np.mean(held**2, axis=0)).max()), 'unprojected worst shift-bank RMSE per pulse', pulse_errors, 'max weight change', float(np.max(abs(w-baseline))))
