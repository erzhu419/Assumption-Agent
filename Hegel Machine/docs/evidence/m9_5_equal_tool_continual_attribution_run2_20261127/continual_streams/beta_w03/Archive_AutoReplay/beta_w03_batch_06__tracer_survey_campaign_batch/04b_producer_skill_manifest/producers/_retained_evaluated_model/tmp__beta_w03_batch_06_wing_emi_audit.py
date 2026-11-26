import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/beta_w03_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
hist = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
worst = (0.0, None)
for idx, record in enumerate(hist):
    if not all(len(x) >= 4 for x, _ in m._sweeps(record)):
        continue
    bank = np.asarray(record['bank_id'], dtype=int)
    ref = np.asarray(record['ref_signal'], dtype=float)
    raw = np.asarray(record['raw_readout'], dtype=float)
    other = [r for j, r in enumerate(hist) if j != idx]
    m.json = SimpleNamespace(loads=lambda _: other)
    errors = []
    for b in range(4):
        ix = np.flatnonzero(bank == b)
        keep = ix[[np.argmin(ref[ix]), np.argmax(ref[ix])]]
        unseen = np.ones(len(ref), dtype=bool)
        unseen[keep] = False
        for pos in (0, 1):
            for delta in (-0.25, 0.25):
                sweep_raw = raw[keep].copy()
                sweep_raw[pos] += delta
                study = {'sensor_calibration_sweeps': {'bank_id': bank[keep], 'ref_signal': ref[keep], 'raw_readout': sweep_raw}, 'row_bank_id': bank, 'y_obs_raw': raw}
                err = np.abs(m._calibrate(study) - ref)
                for target_wing in (0, 1):
                    value = float(err[unseen & (bank // 2 == target_wing)].max())
                    errors.append(value)
                    if value > worst[0]:
                        worst = value, (record['episode_short_name'], b, pos, delta, target_wing)
    print(record['episode_short_name'], 'worst held-out wing error with one corrupted endpoint:', max(errors))
print('worst error and slice:', worst)
assert worst[0] < 1e-5, worst
