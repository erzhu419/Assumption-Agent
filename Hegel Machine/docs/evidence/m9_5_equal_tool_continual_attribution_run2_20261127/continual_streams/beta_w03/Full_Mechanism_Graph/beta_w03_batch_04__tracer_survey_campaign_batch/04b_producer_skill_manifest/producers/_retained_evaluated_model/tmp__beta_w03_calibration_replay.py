import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_w03_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
history = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
keys = ('bank_id', 'ref_signal', 'raw_readout')
for episode in history:
    sweep = {k: np.asarray(episode[k]) for k in keys}
    b, x, y = (sweep[k] for k in keys)
    rows = [np.flatnonzero(b == bank) for bank in range(4)]
    fits = [m._fit_bank(x[r], y[r], (1., 0.)) for r in rows]
    full_error = np.max(np.abs(m._calibrate(dict(sensor_calibration_sweeps=sweep, row_bank_id=b, y_obs_raw=y)) - x))
    print(episode['episode_short_name'], 'bank sizes', [len(r) for r in rows], 'fits', np.round(fits, 8).tolist(), 'worst full-sweep error', full_error)
    for wing in (0, 1):
        local = np.flatnonzero(b // 2 == wing)
        other = np.flatnonzero(b // 2 != wing)
        first, second = rows[2*wing:2*wing+2]
        for label, choices in [('one spot on wing', ((i,) for i in local)), ('one spot per bank', itertools.product(first, second))]:
            worst = (-1., ())
            count = material = 0
            for idx in choices:
                keep = np.r_[other, np.asarray(idx, dtype=int)]
                hold = local[~np.isin(local, idx)]
                if not hold.size:
                    continue
                predicted = m._calibrate(dict(sensor_calibration_sweeps={k: v[keep] for k, v in sweep.items()}, row_bank_id=b[hold], y_obs_raw=y[hold]))
                error = float(np.max(np.abs(predicted-x[hold])))
                worst = max(worst, (error, tuple(int(i) for i in idx)))
                count += 1
                material += error > max(1e-5, 5*full_error)
            print('  wing', wing, label, 'trials', count, 'worst held-out error and kept spots', worst, 'material failures', material)
