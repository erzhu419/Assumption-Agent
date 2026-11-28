import copy
import pickle
import sys
from pathlib import Path
import numpy as np
p = Path('/holdout_task')
sys.path.insert(0, str(p))
from upstream_transducer import _fit, _forward, _inverse, calibrate_upstream
from batch_solver import solve_batch
batch = pickle.loads((p/'public_batch.pkl').read_bytes())
for item in batch:
    st = item['study']; c = st['sensor_calibration_sweeps']; x,y = (c[k] for k in ('ref_rosette','raw_rosette'))
    folds = np.random.default_rng(8).permutation(len(x)) % 5
    clean = np.max(np.abs(y), axis=1) < 5  # Scoring only; all training EMI rows remain available to the robust fit.
    inv = np.full_like(x, np.nan); fwd = np.full_like(y, np.nan)
    for k in range(5):
        tr = folds != k; te = ~tr & clean
        m = _fit(dict(c, ref_rosette=x[tr], raw_rosette=y[tr]))
        fwd[te] = _forward(x[te], *m) - y[te]
        inv[te] = _inverse(y[te], *m) - x[te]
    rms = lambda e: np.sqrt(np.mean(e**2, axis=0))
    worst_fwd = max(np.max(rms(fwd[clean & (folds == k)])) for k in range(5))
    worst_inv = max(np.max(rms(inv[clean & (folds == k)])) for k in range(5))
    edge = clean & (np.max(abs(x), axis=1) > 1.4)
    print('worst fold/bank forward and inverse RMSE', worst_fwd, worst_inv,
          'worst bank edge inverse RMSE', max(rms(inv[edge])), flush=True)
    assert worst_fwd < 5e-5 and worst_inv < 1e-4
    assert max(rms(inv[edge])) < 1e-4
    model = _fit(c)
    shifted = copy.deepcopy(st)
    pulse_ref = x[clean][:6].reshape(2, 3, 4)
    pulse_raw = _forward(pulse_ref, *model)
    shifted['reference_tracer_pulses'] = [dict(y_pulse_raw=pulse_raw, tag='tracer'), {'y_pulse_raw': None}, None]
    before = pickle.dumps(shifted)
    out = calibrate_upstream(shifted)
    assert pickle.dumps(shifted) == before
    assert out['reference_tracer_pulses'][0]['tag'] == 'tracer'
    np.testing.assert_allclose(out['reference_tracer_pulses'][0]['y_pulse_raw'], pulse_ref, atol=1e-9)
    assert out['reference_tracer_pulses'][1:] == [{'y_pulse_raw': None}, None]
    obs = out['y_obs_raw']; assert obs.shape == st['y_obs_raw'].shape
    np.testing.assert_allclose(_forward(obs.reshape(-1,4), *model).reshape(obs.shape), st['y_obs_raw'], atol=1e-9)
    identity = out['sensor_calibration_sweeps']; scalar = identity['single_bank_sweeps']
    for raw, ref in ((identity['raw_rosette'], identity['ref_rosette']), (scalar['raw_readout'], scalar['ref_signal'])):
        np.testing.assert_array_equal(raw, ref)
        assert not np.shares_memory(raw, ref)
    assert isinstance(solve_batch(st), dict)
print('Isolated-process inverse, pulse, identity, and composition checks passed.')
