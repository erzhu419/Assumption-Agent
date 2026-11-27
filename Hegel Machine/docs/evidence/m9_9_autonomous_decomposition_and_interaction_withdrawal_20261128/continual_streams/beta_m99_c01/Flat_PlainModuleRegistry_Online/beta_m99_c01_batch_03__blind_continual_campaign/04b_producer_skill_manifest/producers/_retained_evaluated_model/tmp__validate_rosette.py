import sys, pickle
sys.path.insert(0, '/holdout_task')
import numpy as np
from upstream_transducer import _fit_forward, _inverse, calibrate_upstream
from _verified_downstream import solve_downstream
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
sweep = study['sensor_calibration_sweeps']
x, y = sweep['ref_rosette'], sweep['raw_rosette']
clean = np.max(abs(y), axis=1) < 8.  # Label the visibly separated spikes for scoring only.
axis = np.count_nonzero(x, axis=1) == 1
radius = np.max(abs(x), axis=1)
worst = 0.
for name, mask in [('axis', axis), ('outer', radius >= np.quantile(radius, .8)), ('central', radius <= np.quantile(radius, .2))] + [(f'rows{i}', np.arange(len(x)) % 5 == i) for i in range(5)]:
    coef = _fit_forward(x[~mask], y[~mask])
    test = mask & clean
    err = abs(_inverse(y[test], coef) - x[test])
    worst = max(worst, err.max())
    print(name, test.sum(), 'max', err.max(), 'rms', np.sqrt(np.mean(err**2)))
assert worst < 1e-4
coef = _fit_forward(x, y)
print('fit coefficients:', np.round(coef, 6))
base = calibrate_upstream(study)
assert base['y_obs_raw'].shape == study['y_obs_raw'].shape
assert np.array_equal(base['sensor_calibration_sweeps']['raw_rosette'], x)
assert np.array_equal(base['sensor_calibration_sweeps']['single_bank_sweeps']['raw_readout'], sweep['single_bank_sweeps']['ref_signal'])
assert not np.shares_memory(base['sensor_calibration_sweeps']['raw_rosette'], x)
assert np.max(abs(y)) > 10.
assert calibrate_upstream(dict(study, reference_tracer_pulses=None))['reference_tracer_pulses'] is None
pulse = y[clean][:12].reshape(2, 6, 4)
with_pulse = calibrate_upstream(dict(study, reference_tracer_pulses=[{'y_pulse_raw': pulse, 'marker': 3}, {'y_pulse_raw': pulse.ravel()}]))
for item in with_pulse['reference_tracer_pulses']:
    assert np.max(abs(item['y_pulse_raw'].reshape(-1, 4) - x[clean][:12])) < 1e-4
assert with_pulse['reference_tracer_pulses'][0]['marker'] == 3
before = solve_batch(study)
assert np.max(abs(before['weights'] - solve_downstream(base)['weights'])) == 0.
assert abs(before['weights'].sum() - study['total_budget']) < 1e-10
assert before['weights'].min() >= -1e-10 and before['weights'].max() <= study['u_box'] + 1e-10
print('weights:', before['weights'], 'certificate:', before['kkt_certificate'])
print('worst held-out component error:', worst)
