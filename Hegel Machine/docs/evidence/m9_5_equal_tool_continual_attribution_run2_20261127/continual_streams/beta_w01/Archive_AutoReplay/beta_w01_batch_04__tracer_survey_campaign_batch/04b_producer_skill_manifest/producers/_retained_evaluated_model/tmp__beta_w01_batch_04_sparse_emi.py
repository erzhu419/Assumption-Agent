import importlib.util
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
results = {}
for item in pickle.loads((root / 'public_beta_w01_batch_04.pkl').read_bytes()):
    study, gt = item['study'], item['gt']
    sweep = study['sensor_calibration_sweeps']
    ids, x, y = solver._sweep_columns(sweep)
    reference = solver._calibration(study)
    args = (gt['S_tilde'], gt['r_y'], study['source_ridge'], study['u_box'], study['total_budget'])
    for wing in ((0, 1), (2, 3)):
        for count in (2, 3, 4):
            order = []
            for bank in wing:
                ix = np.flatnonzero(ids == bank)
                ix = ix[np.argsort(x[ix])]
                order.append(ix[np.linspace(0, len(ix) - 1, count).round().astype(int)])
            keep = ~np.isin(ids, wing) | np.isin(np.arange(len(ids)), np.concatenate(order))
            for label, targets in [('clean', []), ('low-end EMI', [seq[0] for seq in order]), ('high-end EMI', [seq[-1] for seq in order])]:
                raw = y.copy()
                raw[np.asarray(targets, dtype=int)] += -1. if label == 'high-end EMI' else 1.
                local = dict(bank_id=ids[keep], ref_signal=x[keep], raw_readout=raw[keep])
                test_study = dict(study, sensor_calibration_sweeps=local)
                out = solver.solve_beta_w01_batch_04(test_study)
                cert = verify(*args, out['kkt_certificate'])
                werr = np.max(np.abs(out['weights'] - gt['w_opt']))
                ferr = np.linalg.norm(out['terminal_field'] - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A'])
                cerr = np.max(np.abs(solver._calibration(test_study) - reference))
                results.setdefault((count, label), []).append((cert['valid'], werr, ferr, cerr, cert['reason']))
for (count, label), rows in sorted(results.items()):
    print(f'{count} spots, {label}: {sum(row[0] for row in rows)}/{len(rows)} valid; worst weight/field/calibration errors', *(f'{v:.3e}' for v in np.max([r[1:4] for r in rows], axis=0)), 'failures', sorted({r[4] for r in rows if not r[0]}))
