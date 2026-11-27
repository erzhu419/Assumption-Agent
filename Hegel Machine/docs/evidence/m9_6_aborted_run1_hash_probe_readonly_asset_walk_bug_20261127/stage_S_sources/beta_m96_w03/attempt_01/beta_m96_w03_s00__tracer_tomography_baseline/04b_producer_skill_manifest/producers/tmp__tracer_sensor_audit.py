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
for case_id, case in enumerate(cases):
    s, gt = case['study'], case['gt']
    Z = s['Z_nuisance']
    rho = s['nuisance_ridge']
    # Recover the reference pre-profile arrays, then re-profile inside each fold.
    prof = np.column_stack((gt['S_tilde'], gt['r_y']))
    raw = prof + Z @ (Z.T @ prof) / rho
    n_sensors = len(s['sensor_ij'])
    sensor = np.tile(np.arange(n_sensors), len(s['mu_shifts']))
    worst_cert_gap = worst_prediction_rmse = worst_weight_shift = 0.
    for j in range(n_sensors):
        keep = sensor != j
        z = Z[keep]
        normal = z.T @ z + rho * np.eye(z.shape[1])
        reference = raw[keep] - z @ np.linalg.solve(normal, z.T @ raw[keep])
        st = dict(s, sensor_ij=s['sensor_ij'][np.arange(n_sensors) != j], Z_nuisance=z,
                  row_bank_id=s['row_bank_id'][keep], y_obs_raw=s['y_obs_raw'][keep])
        out = solver.solve_beta_m96_w03_s00(st)
        proof = verify(reference[:, :-1], reference[:, -1], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'], gap_tol=1e-7)
        assert proof['valid'], (case_id, j, proof)
        w = out['weights']
        res = raw[:, -1] - raw[:, :-1] @ w
        nuisance = np.linalg.solve(normal, z.T @ res[keep])
        worst_prediction_rmse = max(worst_prediction_rmse, np.linalg.norm(res[~keep] - Z[~keep] @ nuisance) / np.sqrt((~keep).sum()))
        worst_weight_shift = max(worst_weight_shift, np.max(abs(w - gt['w_opt'])))
        worst_cert_gap = max(worst_cert_gap, abs(proof['gap']))
    print(f'case={case_id} sensor_folds={n_sensors} max_cert_gap={worst_cert_gap:.3e} max_heldout_prediction_rmse={worst_prediction_rmse:.3e} max_weight_shift_from_full_fit={worst_weight_shift:.3e}')
