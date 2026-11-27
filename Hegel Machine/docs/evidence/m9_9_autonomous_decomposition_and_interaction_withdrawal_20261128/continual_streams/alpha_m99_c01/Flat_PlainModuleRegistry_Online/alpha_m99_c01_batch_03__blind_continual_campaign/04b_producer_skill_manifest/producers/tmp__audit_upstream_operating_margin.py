import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _basis, _fit_bridge, _invert

with open('/holdout_task/public_batch.pkl', 'rb') as stream:
    ds = pickle.load(stream)[0]['dataset']
cal = ds['calibration_load_cells']
x, y = cal['true_channel_loads'], cal['bridge_raw']
base = _fit_bridge(x, y)
cols = np.arange(4)
axis_only = np.count_nonzero(x, axis=1) == 1
axis_id = np.argmax(np.abs(x), axis=1)
clean = np.max(np.abs(y), axis=1) < 10
cases = [(name, field, part['lab_id'], part[field], _invert(part[field], base))
         for name in ('train', 'test') for part in (ds[name],)
         for field in ('past_channel_raw', 'hold_channel_raw')]
worst = (0., None)
for j in cols:
    withheld = axis_only & (axis_id == j)
    fit = _fit_bridge(x[~withheld], y[~withheld])
    score = withheld & clean
    e = _invert(y[score], fit) - x[score]
    summed = np.max(np.abs(e.sum(axis=-1)))
    print(f'withheld axis {j}: worst measured summed-load error {summed:.8g}')
    assert summed < .001, 'Withholding one axis compromises summed-load accuracy'
    for name, field, labs, raw, ref in cases:
        est = _invert(raw, fit)
        assert np.max(np.abs(_basis(est) @ fit - raw)) < 1e-8
        drift = np.abs((est - ref).sum(axis=-1))
        for lab in np.unique(labs):
            peak = float(drift[labs == lab].max())
            if peak > worst[0]:
                worst = (peak, (int(j), name, field, int(lab)))
print('Worst operating summed-load sensitivity to a withheld axis sweep:', worst)
assert worst[0] < .001, 'A measured operating lab slice is unstable under calibration withdrawal'
print('Verified withheld-axis operating stability on all measured lab/load slices')
