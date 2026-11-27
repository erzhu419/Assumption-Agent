import importlib.util
import pickle
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'beta_m96_w03_s00_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (root / 'public_beta_m96_w03_s00.pkl').open('rb') as f:
    cases = pickle.load(f)
for i, case in enumerate(cases):
    s, gt = case['study'], case['gt']
    sweep = s['sensor_calibration_sweeps']
    banks = np.asarray(sweep['bank_id'])
    x, y = map(np.asarray, (sweep['ref_signal'], sweep['raw_readout']))
    rows = np.asarray(s['row_bank_id'])
    Z = s['Z_nuisance']
    projection = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    error = []
    metrics = []
    for j in range(len(banks)):
        keep = np.arange(len(banks)) != j
        bank_keep = keep & (banks == banks[j])
        gain, offset = solver._fit_bank(x[bank_keep], y[bank_keep])
        error.append(abs(gain*x[j] + offset - y[j]))
        st = dict(s, sensor_calibration_sweeps={k: np.asarray(v)[keep] for k, v in sweep.items()})
        obs = solver._observations(st)
        residual = obs - Z @ (projection @ obs) - gt['r_y']
        out = solver.solve_beta_m96_w03_s00(st)
        proof = verify(gt['S_tilde'], gt['r_y'], st['source_ridge'], st['u_box'], st['total_budget'], out['kkt_certificate'], gap_tol=1e-7)
        assert proof['valid'], (i, j, proof)
        werr = np.max(abs(out['weights'] - gt['w_opt']))
        ferr = np.linalg.norm(out['terminal_field'] - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A'])
        bankerr = max(np.max(abs(residual[rows == b])) for b in np.unique(rows))
        metrics.append((werr, ferr, abs(proof['gap']), bankerr))
    worst = np.max(metrics, axis=0)
    print(f'public case {i}: {len(banks)} held-out calibration points; held-out raw error {max(error):.3e}; worst weight max abs / field rel L2 / KKT gap / bankwise profiled obs max abs: {worst}', flush=True)
    assert max(error) < 1e-6 and worst[0] < 1e-6 and worst[1] < 1e-6 and worst[3] < 1e-6
