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
worst_gap = (-1., None)
worst_delta = (-1., None)
with (root / 'public_beta_ep04.pkl').open('rb') as f:
    for case, item in enumerate(pickle.load(f)):
        s = item['study']
        baseline = m.solve_beta_ep04(s)['weights']
        ij = s['sensor_ij']
        S, _ = m._transport(s)
        y = m._calibrate(s)
        for sensor in range(len(ij)):
            keep = np.arange(len(ij)) != sensor
            rows = np.tile(keep, len(s['mu_shifts']))
            fold = dict(s, sensor_ij=ij[keep], row_bank_id=s['row_bank_id'][rows], y_obs_raw=s['y_obs_raw'][rows], Z_nuisance=s['Z_nuisance'][rows])
            out = m.solve_beta_ep04(fold)
            Z = fold['Z_nuisance']
            joined = np.column_stack((S[rows], y[rows]))
            projected = joined - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ joined)
            check = verify(projected[:, :-1], projected[:, -1], s['source_ridge'], s['u_box'], s['total_budget'], out['kkt_certificate'], gap_tol=1e-8)
            assert check['valid'], (case, sensor, check)
            assert np.allclose(out['weights'], out['kkt_certificate']['weights'], rtol=0, atol=1e-12)
            delta = np.max(np.abs(out['weights'] - baseline))
            if check['gap'] > worst_gap[0]:
                worst_gap = (check['gap'], (case, sensor))
            if delta > worst_delta[0]:
                worst_delta = (delta, (case, sensor))
print('Worst leave-one-sensor-out KKT gap and slice:', worst_gap)
print('Largest weight change and slice (sensor leverage, not solver error):', worst_delta)
