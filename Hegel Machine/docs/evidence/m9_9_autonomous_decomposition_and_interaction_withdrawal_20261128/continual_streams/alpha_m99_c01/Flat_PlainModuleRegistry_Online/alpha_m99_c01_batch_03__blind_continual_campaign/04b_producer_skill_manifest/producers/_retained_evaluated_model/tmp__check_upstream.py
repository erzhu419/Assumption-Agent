import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_bridge, _basis, _invert, calibrate_upstream
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as stream:
    ds = pickle.load(stream)[0]['dataset']
cal = ds['calibration_load_cells']
x, y = (cal[k] for k in ('true_channel_loads', 'bridge_raw'))
# Gross spikes are omitted from scoring only; all other rows, including
# the spikes, remain available to the robust fit when in the training fold.
clean = np.max(np.abs(y), axis=1) < 10
sweep = np.count_nonzero(x, axis=1) == 1
axis = np.argmax(np.abs(x), axis=1)
errors = np.full_like(x, np.nan)
for i in np.flatnonzero(clean):
    train = np.arange(len(x)) != i
    coef = _fit_bridge(x[train], y[train])
    errors[i] = _invert(y[i], coef) - x[i]
for name, m in [('single-axis LOO', sweep), ('mixed LOO', ~sweep),
                ('large-amplitude LOO', np.max(np.abs(x), axis=1) >= 1.4)]:
    worst = np.nanmax(np.abs(errors[m]), axis=0)
    print(name, 'worst absolute load error', worst)
    assert worst.max() < .001
for j in range(4):
    leave = sweep & (axis == j)
    coef = _fit_bridge(x[~leave], y[~leave])
    worst = np.max(np.abs(_invert(y[leave & clean], coef) - x[leave & clean]))
    print('axis block withheld', j, 'worst absolute load error', worst)
    assert worst < .001

raw_cal = y.copy()
raw_cmd = cal['single_channel_sweeps']['cmd_stress'].copy()
raw_parts = {(part, field): ds[part][field].copy() for part in ('train', 'test')
             for field in ('past_channel_raw', 'hold_channel_raw')}
coef = _fit_bridge(x, y)
canonical = calibrate_upstream(ds)
for (part, field), raw in raw_parts.items():
    loads = canonical[part][field]
    worst = np.max(np.abs(_basis(loads) @ coef - raw))
    print(part, field, 'worst round-trip bridge error', worst)
    assert loads.shape == raw.shape and worst < 1e-8
    np.testing.assert_array_equal(ds[part][field], raw)
np.testing.assert_array_equal(cal['bridge_raw'], raw_cal)
np.testing.assert_array_equal(cal['single_channel_sweeps']['cmd_stress'], raw_cmd)
c = canonical['calibration_load_cells']
np.testing.assert_array_equal(c['bridge_raw'], x)
assert not np.shares_memory(c['bridge_raw'], x)
s = c['single_channel_sweeps']
np.testing.assert_array_equal(s['cmd_stress'], s['true_stress'])
assert not np.shares_memory(s['cmd_stress'], s['true_stress'])
trial = dict(past_channel_raw=ds['test']['past_channel_raw'][:3], hold_channel_raw=ds['test']['hold_channel_raw'][:3])
result = calibrate_upstream(dict(ds, chamber_verification_trials=trial))
for field in trial:
    np.testing.assert_allclose(result['chamber_verification_trials'][field], canonical['test'][field][:3], atol=1e-10, rtol=0)
assert calibrate_upstream(dict(ds, chamber_verification_trials=None))['chamber_verification_trials'] is None
# Recalibration of the normalized dataset must act as identity.
for k, v in solve_batch(ds).items():
    np.testing.assert_allclose(solve_batch(canonical)[k], v, atol=1e-9, rtol=1e-9)
print('isolated upstream and composition checks passed')
