import importlib.util, itertools, json, pickle
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with (root/'public_alpha_m97_c01_batch_01.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    history = json.load(f)
c = d['calibration_load_cells']; labs = c['lab_id']; x = c['cmd_stress']; y = c['true_stress']
ref = np.array([np.polyfit(x[labs == j], y[labs == j], 1) for j in range(4)])
np.testing.assert_allclose(m._calibrate(d), ref, atol=1e-12)
worst = {2: 0., 3: 0.}; count = 0
for j in range(4):
    other = np.flatnonzero(labs != j); rows = np.flatnonzero(labs == j)
    sweep = next(h for h in reversed(history) if np.count_nonzero(np.asarray(h['lab_id']) == j) >= 2)
    q = np.asarray(sweep['lab_id']) == j
    fallback = np.polyfit(np.asarray(sweep['cmd_stress'])[q], np.asarray(sweep['true_stress'])[q], 1)
    test = d['test']; q = test['lab_id'] == j
    vals = np.r_[test['past_stress_cmd'][q].ravel(), test['stress_hold_cmd'][q]]
    for size in (2, 3):
        for combo in itertools.combinations(rows, size):
            keep = np.r_[other, combo]; clean = {k: np.asarray(v)[keep].copy() for k, v in c.items()}
            np.testing.assert_allclose(m._calibrate(dict(d, calibration_load_cells=clean))[j], np.polyfit(x[list(combo)], y[list(combo)], 1), atol=1e-10)
            for k, bad in enumerate(combo):
                good = [i for i in combo if i != bad]
                expected = fallback if size == 2 else np.polyfit(x[good], y[good], 1)
                for delta in (-4., -3.0001, 3.0001, 4.):
                    corrupted = dict(clean, true_stress=clean['true_stress'].copy())
                    corrupted['true_stress'][len(other)+k] += delta
                    actual = m._calibrate(dict(d, calibration_load_cells=corrupted))
                    np.testing.assert_allclose(actual[j], expected, atol=1e-10)
                    np.testing.assert_allclose(actual[np.arange(4) != j], ref[np.arange(4) != j], atol=1e-12)
                    worst[size] = max(worst[size], np.max(abs(np.polyval(actual[j]-ref[j], vals))))
                    count += 1
print('Sparse EMI cases passed:', count, 'worst stress shift by sweep size:', worst)
