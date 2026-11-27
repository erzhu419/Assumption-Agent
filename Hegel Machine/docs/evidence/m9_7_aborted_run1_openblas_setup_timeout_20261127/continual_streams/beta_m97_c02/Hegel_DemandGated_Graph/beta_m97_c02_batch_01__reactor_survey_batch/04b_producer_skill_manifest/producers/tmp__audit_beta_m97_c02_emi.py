import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('batch_solver', root/'beta_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_beta_m97_c02_batch_01.pkl').open('rb') as f:
    items = pickle.load(f)

worst = (0., None)
failures = 0
for case, item in enumerate(items):
    s = item['study']
    b, x, y = (np.asarray(s['sensor_calibration_sweeps'][k]) for k in ('bank_id','ref_signal','raw_readout'))
    for bank in range(4):
        ids = np.flatnonzero(b == bank)
        ids = ids[np.argsort(x[ids])]
        selected = ids[np.linspace(0, len(ids)-1, 5, dtype=int)]
        for spike in selected:
            clean = selected[selected != spike]
            base = np.flatnonzero((b != bank) | np.isin(np.arange(len(b)), clean))
            contaminated = np.sort(np.append(base, spike))
            def study(rows, raw):
                return dict(s, sensor_calibration_sweeps=dict(bank_id=b[rows], ref_signal=x[rows], raw_readout=raw[rows]))
            expected = m._calibration(study(base, y))
            for emi in (-3.1, 3.1):
                raw = y.copy()
                raw[spike] += emi
                found = m._calibration(study(contaminated, raw))
                error = float(np.max(np.abs(found-expected)))
                label = (case, bank, float(x[spike]), emi)
                failures += error > 1e-10
                if error > worst[0]:
                    worst = (error, label)
print('Four clean sweeps plus one signed EMI spike:', 'failed slices', failures, 'worst calibrated-readout error and slice', worst)
if failures:
    import inspect
    print(inspect.getsource(m._calibration))
    raise SystemExit(1)
