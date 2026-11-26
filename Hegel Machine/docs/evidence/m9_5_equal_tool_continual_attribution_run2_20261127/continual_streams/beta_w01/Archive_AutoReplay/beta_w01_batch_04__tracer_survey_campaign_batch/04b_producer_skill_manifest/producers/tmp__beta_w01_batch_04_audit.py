import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'beta_w01_batch_04_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
hist = json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
for item in pickle.loads((root / 'public_beta_w01_batch_04.pkl').read_bytes()):
    study, gt = item['study'], item['gt']
    sw = study['sensor_calibration_sweeps']
    ids, x, y = solver._sweep_columns(sw)
    for entry in hist:
        b, u, v = solver._sweep_columns(entry)
        fit = np.array([solver._fit_affine(u[b == j], v[b == j]) for j in range(4)])
        pred = fit[ids, 0] * x + fit[ids, 1]
        print('held-out public sweep max residual by bank', entry['episode_short_name'], [float(np.max(np.abs(pred[ids == j] - y[ids == j]))) for j in range(4)])
    a, fields = solver._responses(study)
    z = study['Z_nuisance']
    prof = a - z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ a)
    design_err = np.max(np.abs(prof - gt['S_tilde']))
    field_err = np.linalg.norm(fields @ gt['w_opt'] - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A'])
    print('independent profiled design max error', design_err, 'reference-weight field relative error', field_err)
    assert design_err < 1e-12 and field_err < 1e-12
    args = (gt['S_tilde'], gt['r_y'], study['source_ridge'], study['u_box'], study['total_budget'])
    ix = [np.flatnonzero(ids == j) for j in range(4)]
    for wing in ((0, 1), (2, 3), (0, 1, 2, 3)):
        keep = ~np.isin(ids, wing)
        rows = []
        for k in range(-1, min(len(ix[j]) for j in wing)):
            mask = keep.copy()
            if k >= 0:
                mask[[ix[j][k] for j in wing]] = True
            out = solver.solve_beta_w01_batch_04(dict(study, sensor_calibration_sweeps={key: val[mask] for key, val in sw.items()}))
            cert = verify(*args, out['kkt_certificate'])
            w = np.max(np.abs(out['weights'] - gt['w_opt']))
            f = np.linalg.norm(out['terminal_field'] - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A'])
            rows.append((cert['valid'], cert['gap'], w, f, cert['reason']))
        print('simultaneous sparse banks', wing, 'valid', sum(r[0] for r in rows), '/', len(rows), 'worst gap/weight/field', np.max([r[1:4] for r in rows], axis=0), 'failure reasons', sorted({r[4] for r in rows if not r[0]}))
