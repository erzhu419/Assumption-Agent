import pickle
import sys

import numpy as np

sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
import batch_solver as batch
import _verified_downstream as dn


def compare(actual, expected, path='result'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k in expected:
            compare(actual[k], expected[k], f'{path}.{k}')
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{path}[{i}]')
    elif np.asarray(expected).dtype.kind in 'iufc':
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8, err_msg=path)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=path)


with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
rng = np.random.default_rng(42)
worst = dict(central=0., edge=0., observations=0., pulses=0., synthetic_emi=0.)
pulse_count = 0
for item in items:
    st = item['study']; sw = st['sensor_calibration_sweeps']
    x = sw['ref_rosette']; y = sw['raw_rosette']
    clean = np.linalg.norm(y, axis=1) < 8
    radius = np.max(abs(x), axis=1)
    edge = radius >= np.quantile(radius, .8)
    for hold in np.array_split(rng.permutation(len(x)), 5):
        train = np.ones(len(x), dtype=bool); train[hold] = False
        coef = up._fit(x[train], y[train])
        for name, mask in [('central', ~edge), ('edge', edge)]:
            idx = hold[(mask & clean)[hold]]
            err = np.linalg.norm(up._inverse(y[idx], coef) - x[idx], axis=1).max(initial=0.)
            worst[name] = max(worst[name], err)
    out = up.calibrate_upstream(st)
    coef = up._fit(x, y)
    readings = [('observations', st['y_obs_raw'], out['y_obs_raw'])]
    for a, b in zip(st.get('reference_tracer_pulses') or [], out.get('reference_tracer_pulses') or []):
        if a is not None and a.get('y_pulse_raw') is not None:
            readings.append(('pulses', a['y_pulse_raw'], b['y_pulse_raw']))
            pulse_count += 1
    for name, a, b in readings:
        assert np.shape(a) == np.shape(b) and np.isfinite(b).all(), name
        residual = up._design(b.reshape(-1, 4)) @ coef - np.asarray(a).reshape(-1, 4)
        worst[name] = max(worst[name], np.max(abs(residual), initial=0.))
    cal = out['sensor_calibration_sweeps']
    np.testing.assert_array_equal(cal['raw_rosette'], x)
    np.testing.assert_array_equal(cal['single_bank_sweeps']['raw_readout'], sw['single_bank_sweeps']['ref_signal'])
    c = coef.copy(); c[0] = [.08, -.043, .021, -.094]
    c[1:5] = np.roll(coef[1:5], (1, 1), axis=(0, 1)); c[5:11] *= .5; c[11:] *= .8
    def encode(a):
        a = np.asarray(a)
        return (up._design(a.reshape(-1, 4)) @ c).reshape(a.shape)
    z = encode(x)
    ids = rng.choice(len(z), max(1, len(z)//10), replace=False)
    z[ids] += 20 * rng.choice([-1, 1], (len(ids), 4))
    synth = dict(out, sensor_calibration_sweeps=dict(sw, raw_rosette=z), y_obs_raw=encode(out['y_obs_raw']), reference_tracer_pulses=[None, {'y_pulse_raw': None}, {'y_pulse_raw': encode(x[edge])}])
    got = up.calibrate_upstream(synth)
    assert got['reference_tracer_pulses'][:2] == synth['reference_tracer_pulses'][:2]
    assert up.calibrate_upstream(dict(synth, reference_tracer_pulses=None))['reference_tracer_pulses'] is None
    worst['synthetic_emi'] = max(worst['synthetic_emi'], np.max(abs(got['y_obs_raw'] - out['y_obs_raw'])), np.max(abs(got['reference_tracer_pulses'][2]['y_pulse_raw'] - x[edge])))
    print('worst so far:', worst, flush=True)
    compare(batch.solve_batch(synth), dn.solve_downstream(out))
print(f'{len(items)} public studies; {pulse_count} non-null public pulses; all downstream outputs match')
assert max(worst['edge'], worst['central']) < 1e-4, worst
assert max(worst['synthetic_emi'], worst['observations'], worst['pulses']) < 1e-8, worst
