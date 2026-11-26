import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'beta_ep04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d = ')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (root / 'public_beta_ep04.pkl').open('rb') as f:
    cases = pickle.load(f)
worst_loo = worst_weight_delta = worst_field_delta = worst_gap = 0.0
for item in cases:
    s, gt = item['study'], item['gt']
    out = m.solve_beta_ep04(s)
    w, field = out['weights'], out['terminal_field']
    check = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
    print('certificate against public reference:', check)
    assert check['valid'], check
    worst_gap = max(worst_gap, check['gap'])
    sweep = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
    banks, x, raw = (sweep[k] for k in ('bank_id', 'ref_signal', 'raw_readout'))
    X = np.column_stack((x, np.ones_like(x)))
    pooled = np.linalg.lstsq(X, raw, rcond=None)[0]
    for b in np.unique(banks):
        idx = np.flatnonzero(banks == b)
        bank_loo = 0.0
        for i in idx:
            training = idx[idx != i]
            beta = np.linalg.lstsq(X[training], raw[training], rcond=None)[0]
            bank_loo = max(bank_loo, abs(X[i] @ beta - raw[i]))
            keep = np.arange(len(banks)) != i
            perturbed = dict(s, sensor_calibration_sweeps={k: v[keep] for k, v in sweep.items()})
            alt = m.solve_beta_ep04(perturbed)
            worst_weight_delta = max(worst_weight_delta, np.max(abs(alt['weights'] - w)))
            worst_field_delta = max(worst_field_delta, np.max(abs(alt['terminal_field'] - field)))
        worst_loo = max(worst_loo, bank_loo)
        print('bank', b, 'held-out raw max error', bank_loo, 'pooled raw max error', np.max(abs(X[idx] @ pooled - raw[idx])))
print('worst held-out raw error', worst_loo, 'weight delta', worst_weight_delta, 'terminal delta', worst_field_delta, 'public-reference KKT gap', worst_gap)
assert worst_weight_delta < 1e-6 and worst_field_delta < 1e-6
