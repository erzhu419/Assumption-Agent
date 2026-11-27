import sys
from itertools import product
import numpy as np
sys.path[:0] = ['/holdout_task', '/tmp']
from batch_solver import solve_batch
from downstream_solver import _design, solve_downstream
from _verified_upstream import calibrate_upstream
from chamber_probe import s, W, pulses, pay
from wall_probe import predict

cal = calibrate_upstream(s)
S, _ = _design(cal)
Y = np.column_stack([calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']))['y_obs_raw'] for p in pulses])
Z = cal['Z_nuisance']
periodic = ('periodic', 'periodic')
kinds = ('reflect','robin','double','half','dirichlet','periodic','wraplocal','wraponly')
residuals = {p: predict(*p) - Y for p in product(kinds, repeat=2)}
np.testing.assert_allclose(residuals[periodic], S @ W - Y, rtol=0, atol=1e-12)
sensors = np.tile(np.arange(len(s['sensor_ij'])), len(s['mu_shifts']))
worst = float('-inf')
for sensor in np.unique(sensors):
    test = sensors == sensor
    def adjusted(r):
        return r - Z @ np.linalg.lstsq(Z[~test], r[~test], rcond=None)[0]
    selected = min(residuals, key=lambda p: np.mean(adjusted(residuals[p])[~test]**2))
    assert selected == periodic
    err = np.sqrt(np.mean(adjusted(residuals[selected])[test]**2, axis=0)).max()
    worst = max(worst, err)
assert worst < 1e-5
print('Boundary: periodic on both axes in every leave-one-sensor-out fold; worst held-out pulse RMSE', worst)

sweeps = s['sensor_calibration_sweeps']
raw = sweeps['raw_rosette']
ref = sweeps['ref_rosette']
valid = np.all(abs(raw) <= 8.5, axis=1)
fold = np.arange(len(raw)) % 5
rmse = []
for k in range(5):
    train, test = fold != k, (fold == k) & valid
    study = dict(s, sensor_calibration_sweeps=dict(sweeps, raw_rosette=raw[train], ref_rosette=ref[train]), y_obs_raw=raw[test].ravel())
    dynamic = calibrate_upstream(study)['y_obs_raw'].reshape(-1, 4)
    fixed = calibrate_upstream(study, pay)['y_obs_raw'].reshape(-1, 4)
    rmse.append([np.sqrt(np.mean((p-ref[test])**2, axis=0)).max() for p in (dynamic, fixed)])
worst_dynamic, worst_fixed = np.max(rmse, axis=0)
assert worst_dynamic < 1e-4
print('Calibration: worst held-out real sweep channel RMSE, dynamic and fixed', worst_dynamic, worst_fixed)

base = solve_batch(s)
np.testing.assert_array_equal(base['weights'], solve_downstream(cal)['weights'])
a = np.array([1.05,.95,1.08,.92]); b = np.array([.03,-.04,.05,-.02])
shifted = np.where(valid[:,None], raw*a+b, raw)
assert np.array_equal(valid, np.all(abs(shifted)<=8.5, axis=1))
study = dict(s, sensor_calibration_sweeps=dict(sweeps, raw_rosette=shifted), y_obs_raw=(s['y_obs_raw'].reshape(-1,4)*a+b).ravel())
trial = solve_batch(study)
fixed = [solve_downstream(calibrate_upstream(t,pay)) for t in (s,study)]
dynamic_delta = np.max(abs(base['weights']-trial['weights']))
fixed_delta = np.max(abs(fixed[0]['weights']-fixed[1]['weights']))
assert dynamic_delta < 1e-10 and fixed_delta > .1
print('Gain/offset transfer: maximum weight drift, dynamic and fixed',dynamic_delta,fixed_delta)
