import pickle
import sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from upstream_transducer import _fit_forward, _inverse, calibrate_upstream
from _verified_downstream import solve_downstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
sweeps = study['sensor_calibration_sweeps']
x, y = sweeps['ref_rosette'], sweeps['raw_rosette']
rows = np.arange(len(x))
scorable = np.max(abs(y), axis=1) < 8  # visibly separated EMI, used only for scoring
reference = calibrate_upstream(study)
weights = solve_downstream(reference)['weights']
worst = np.zeros(4)
obs_shift = weight_shift = 0.
for fold in range(5):
    test = rows % 5 == fold
    coef = _fit_forward(x[~test], y[~test])
    err = abs(_inverse(y[test & scorable], coef) - x[test & scorable])
    worst = np.maximum(worst, err.max(axis=0))
    obs = _inverse(study['y_obs_raw'], coef)
    obs_shift = max(obs_shift, float(np.max(abs(obs - reference['y_obs_raw']))))
    w = solve_downstream(dict(reference, y_obs_raw=obs))['weights']
    weight_shift = max(weight_shift, float(np.max(abs(w - weights))))
print('Worst held-out inverse error by channel:', worst)
print('Largest observation change across refits:', obs_shift)
print('Largest downstream weight change across refits:', weight_shift)
print('Minimum downstream box slack:', min(weights.min(), study['u_box'] - weights.max()))
assert max(worst) < 1e-4 and obs_shift < 1e-4 and weight_shift < 1e-4
