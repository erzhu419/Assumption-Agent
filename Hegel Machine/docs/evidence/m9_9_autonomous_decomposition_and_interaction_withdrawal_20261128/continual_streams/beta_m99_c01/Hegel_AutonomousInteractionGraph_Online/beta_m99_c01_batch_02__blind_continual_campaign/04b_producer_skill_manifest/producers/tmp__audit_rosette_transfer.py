import json
import pickle
import sys
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
sys.path.insert(0, str(root))
from _verified_upstream import calibrate_upstream
from downstream_solver import _forward

with (root / 'public_batch.pkl').open('rb') as stream:
    items = pickle.load(stream)
payload = json.loads((root / '_upstream_param_payload.json').read_text())
fold_errors, frozen_errors = [], []
for item in items:
    study = item['study']
    sweeps = study['sensor_calibration_sweeps']
    n = len(sweeps['ref_rosette'])
    nshift = len(study['mu_shifts'])
    s, _ = _forward(study)
    for pulse in item['campaign_diagnostics'] or []:
        probe = {**study, 'y_obs_raw': pulse['y_pulse_raw']}
        expected = s @ pulse['w_pulse']
        def rmse(x):
            return np.sqrt(np.mean((x-expected).reshape(nshift, -1)**2, axis=1))
        frozen_errors.extend(rmse(calibrate_upstream(probe, payload)['y_obs_raw']))
        for fold in range(4):
            keep = np.arange(n) % 4 != fold
            train = {**sweeps, 'ref_rosette': np.asarray(sweeps['ref_rosette'])[keep], 'raw_rosette': np.asarray(sweeps['raw_rosette'])[keep]}
            observed = calibrate_upstream({**probe, 'sensor_calibration_sweeps': train}, None)['y_obs_raw']
            fold_errors.extend(rmse(observed))
assert fold_errors and frozen_errors
worst = max(fold_errors)
print(f'Sweep subsampling: {len(fold_errors)} pulse-by-shift calibration-fold slices, worst RMSE={worst:.6e}; frozen payload worst RMSE={max(frozen_errors):.6e}')
assert worst < 2e-6, 'Sweep-fit calibration exceeds the established pulse residual margin when calibration rows are withheld'
