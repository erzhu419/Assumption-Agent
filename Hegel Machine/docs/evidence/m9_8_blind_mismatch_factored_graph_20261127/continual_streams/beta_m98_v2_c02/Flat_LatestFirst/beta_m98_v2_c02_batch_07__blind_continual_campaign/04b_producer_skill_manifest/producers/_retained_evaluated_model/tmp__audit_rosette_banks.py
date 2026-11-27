import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_batch.pkl').open('rb') as f:
    items = pickle.load(f)
worst = (0.0, None)
for i, item in enumerate(items):
    s = item['study']
    pulses = item['campaign_diagnostics']
    S, _ = m._forward(s)
    sw = s['sensor_calibration_sweeps']
    ref, raw = sw['ref_rosette'], sw['raw_rosette']
    z = s['Z_nuisance']
    nshift = len(s['mu_shifts'])
    for fold, held in enumerate(np.array_split(np.random.default_rng(0).permutation(len(ref)), 5)):
        keep = np.ones(len(ref), dtype=bool)
        keep[held] = False
        sub = {**sw, 'ref_rosette': ref[keep], 'raw_rosette': raw[keep]}
        for j, pulse in enumerate(pulses):
            y = m._calibrate({**s, 'sensor_calibration_sweeps': sub, 'y_obs_raw': pulse['y_pulse_raw']})
            r = y - S @ pulse['w_pulse']
            r_orth = r - z @ np.linalg.lstsq(z, r, rcond=None)[0]
            for name, rr in [('raw', r), ('nuisance-orthogonal', r_orth)]:
                bank_rms = np.sqrt(np.mean(rr.reshape(nshift, -1, 4)**2, axis=1))
                shift, bank = np.unravel_index(bank_rms.argmax(), bank_rms.shape)
                value = float(bank_rms[shift, bank])
                assert np.isfinite(value)
                if value > worst[0]:
                    worst = (value, (i, fold, j, int(shift), int(bank), name))
print('Worst actual candidate inversion RMS, with location (item, fold, pulse, shift, bank, projection):', worst)
assert worst[1] is not None and worst[0] < 5e-6
