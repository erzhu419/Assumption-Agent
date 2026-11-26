import importlib.util
import pickle
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
name = 'beta_w02_batch_07_solver.py'

def load(path):
    spec = importlib.util.spec_from_file_location('candidate', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old_path, = Path('/tmp/.hegel_candidate_archive/v01_02f0fb27').rglob(name)
old, new = load(old_path), load(root / name)
with (root / 'public_beta_w02_batch_07.pkl').open('rb') as f:
    case, = pickle.load(f)
s, ref = case['study'], case['gt']
sw = s['sensor_calibration_sweeps']
rows = s['row_bank_id']
S, y = ref['S_tilde'], ref['r_y']
cap, budget, ridge = (s[k] for k in ('u_box', 'total_budget', 'source_ridge'))

def metrics(out):
    w = out['weights']
    nu = out['kkt_certificate']['nu_sum']
    g = S.T @ (S @ w - y) + ridge * w
    gap = g @ w - nu * budget - cap * np.minimum(g-nu, 0).sum()
    rel = lambda u, v: np.linalg.norm(u-v) / np.linalg.norm(v)
    return np.array([rel(w, ref['w_opt']), rel(out['terminal_field'], ref['terminal_field_A']), gap])

A, _ = new._forward(s)
worst = np.zeros(3)
worst_bank = np.zeros(4)
for j in np.flatnonzero(sw['bank_id'] >= 2):
    keep = np.arange(len(sw['bank_id'])) != j
    v = dict(s, sensor_calibration_sweeps={k: a[keep] for k, a in sw.items()})
    before = metrics(old.solve_beta_w02_batch_07(v))
    after = metrics(new.solve_beta_w02_batch_07(v))
    _, y_after = new._profile(v, A, new._calibrate(v))
    bank_err = np.array([np.max(np.abs(y_after[rows == b]-y[rows == b])) for b in range(4)])
    worst = np.maximum(worst, after)
    worst_bank = np.maximum(worst_bank, bank_err)
    if j == 24:
        print('Observed spot 24 removed: old -> final [relative weights, relative field, independent KKT gap]', before.tolist(), after.tolist())
        assert before[0] > 1.
    assert np.all(after < 5e-6) and after[2] >= -1e-12, (j, after)
    assert np.all(bank_err < 1e-6), (j, bank_err)
print('Worst over six observed Wing B spot deletions:', worst.tolist(), '; worst bankwise profiled observation errors:', worst_bank.tolist())
print(f'Worst KKT gap / public validator tolerance: {worst[2]/2e-4:.3e}')
