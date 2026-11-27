import importlib.util
import itertools
import pickle
from pathlib import Path
from unittest.mock import patch
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'beta_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
for case, item in enumerate(pickle.loads((root / 'public_beta_m97_c02_batch_01.pkl').read_bytes())):
    s, gt = item['study'], item['gt']
    q = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
    design = m._forward(s)
    with patch.object(m, '_forward', return_value=design):
        for bank in range(4):
            ids = np.flatnonzero(q['bank_id'] == bank)
            x = q['ref_signal']
            groups = list(itertools.combinations(ids, 4)) if len(ids) >= 4 else []
            slices = [('leave_one_out', np.delete(ids, j)) for j in range(len(ids)) if len(ids) >= 5]
            if groups:
                for label, pick in [('tight_four', min), ('wide_four', max)]:
                    retained = np.asarray(pick(groups, key=lambda idx: np.var(x[list(idx)])))
                    slices.append((label, retained))
            results = {}
            for label, retained in slices:
                keep = (q['bank_id'] != bank) | np.isin(np.arange(len(x)), retained)
                reduced = dict(s, sensor_calibration_sweeps={k: v[keep] for k, v in q.items()})
                out = m.solve_beta_m97_c02_batch_01(reduced)
                c = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'])
                errors = [np.linalg.norm(out[key] - gt[ref]) / np.linalg.norm(gt[ref]) for key, ref in [('weights', 'w_opt'), ('terminal_field', 'terminal_field_A')]]
                results.setdefault(label, []).append((c['valid'], c['gap'], c['reason'], *errors))
            for label, rows in results.items():
                reasons = sorted({r[2] for r in rows if not r[0]})
                print(f'case={case} bank={bank} slice={label} passed={sum(r[0] for r in rows)}/{len(rows)} gap_max={max(r[1] for r in rows):.3e} weight_rel_max={max(r[3] for r in rows):.3e} field_rel_max={max(r[4] for r in rows):.3e} failures={reasons}')
