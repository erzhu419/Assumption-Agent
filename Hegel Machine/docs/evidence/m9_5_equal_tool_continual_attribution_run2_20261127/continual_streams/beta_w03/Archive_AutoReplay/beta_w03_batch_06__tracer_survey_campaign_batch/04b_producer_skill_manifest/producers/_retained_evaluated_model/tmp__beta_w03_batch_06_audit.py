import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'beta_w03_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
hist = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
worst = 0.0
for idx, record in enumerate(hist):
    if not all(len(x) >= 4 for x, _ in m._sweeps(record)):
        continue
    other = [r for j, r in enumerate(hist) if idx != j]
    m.json = SimpleNamespace(loads=lambda _: other)
    bank = np.asarray(record['bank_id'], dtype=int)
    ref = np.asarray(record['ref_signal'])
    raw = np.asarray(record['raw_readout'])
    per_bank = np.zeros(4)
    for b in range(4):
        ix = np.flatnonzero(bank == b)
        for keep in ([ix[np.argmin(ref[ix])]], [ix[np.argmax(ref[ix])]], ix[[np.argmin(ref[ix]), np.argmax(ref[ix])]]):
            study = {'sensor_calibration_sweeps': {'bank_id': bank[keep], 'ref_signal': ref[keep], 'raw_readout': raw[keep]}, 'row_bank_id': bank, 'y_obs_raw': raw}
            err = np.abs(m._calibrate(study) - ref)
            held_out = np.ones(len(ref), dtype=bool)
            held_out[keep] = False
            per_bank = np.maximum(per_bank, [np.max(err[held_out & (bank == k)]) for k in range(4)])
    print(record['episode_short_name'], 'held-out bank residual Linf', per_bank.tolist())
    worst = max(worst, float(per_bank.max()))
m.json = json
s, g = [(item['study'], item['gt']) for item in pickle.loads((root / 'public_beta_w03_batch_06.pkl').read_bytes())][0]
R = m._responses(s)
i, j = s['sensor_ij'].T
D = R[:, :, i, j].transpose(0, 2, 1).reshape(-1, R.shape[1])
P = m._profile(np.c_[D, m._calibrate(s)], s['Z_nuisance'], s['nuisance_ridge'])
print('public worst profiled design error by shift', np.max(np.abs(P[:, :-1] - g['S_tilde']).reshape(len(R), -1), axis=1).tolist())
print('public profiled observation error by wing', [float(np.max(np.abs(P[:, -1] - g['r_y'])[np.isin(s['row_bank_id'], wing)])) for wing in ((0, 1), (2, 3))])
assert np.max(np.abs(P[:, :-1] - g['S_tilde'])) < 1e-12
assert worst < 1e-5, worst
