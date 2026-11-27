import importlib.util, pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with (root/'public_alpha_m97_c01_batch_01.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; p = d['test']
for lab in range(4):
    ids = np.flatnonzero(c['lab_id']==lab)
    vals = np.r_[p['past_stress_cmd'][p['lab_id']==lab].ravel(), p['stress_hold_cmd'][p['lab_id']==lab]]
    baseline = m._calibrate(d)[lab]
    for size in (2, 3, 5):
        worst = 0.
        for j in range(size):
            for sign in (-1, 1):
                keep = np.r_[np.flatnonzero(c['lab_id']!=lab), ids[:size]]
                sweeps = {k: np.asarray(v)[keep].copy() for k,v in c.items()}
                sweeps['true_stress'][-size+j] += sign*4
                est = m._calibrate(dict(d, calibration_load_cells=sweeps))[lab]
                worst = max(worst, np.max(np.abs(np.polyval(est-baseline, vals))))
        print('channel, rows including EMI, worst physical stress shift:', lab, size, round(worst, 6))
