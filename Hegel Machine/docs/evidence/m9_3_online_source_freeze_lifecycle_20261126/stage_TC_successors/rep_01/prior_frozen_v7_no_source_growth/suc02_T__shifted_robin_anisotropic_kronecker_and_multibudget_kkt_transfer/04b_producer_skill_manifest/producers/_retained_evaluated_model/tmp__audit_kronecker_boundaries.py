import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.linalg import solve_sylvester

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'multibudget_kronecker_kkt_transfer_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
worst_sensor = worst_field = worst_gap = 0.
slices = 0
with (root / 'public_suc02_T_studies.pkl').open('rb') as f:
    studies = pickle.load(f)
for item in studies:
    s, gt = item['study'], item['gt']
    for axis in 'xy':
        t = dict(s)
        velocity = np.asarray(s['vel_' + axis])
        signs = np.ones(velocity.size)
        signs[::2] = -1.
        new_velocity = velocity * signs
        robin = np.broadcast_to(s['robin_' + axis], (2,))
        new_robin = robin * [0.45, 1.7]
        t['vel_' + axis], t['robin_' + axis] = new_velocity, new_robin
        h = s['L' + axis] / velocity.size
        key = {'x': 'Ax', 'y': 'By'}[axis]
        mat = gt[key].copy()
        # Changing only flow direction leaves its diagonal contribution intact.
        # Move the upwind contribution from the old neighbor to the new one.
        for row in range(0, velocity.size, 2):
            v = velocity[row]
            if v == 0:
                continue
            old = row - int(np.sign(v))
            new = row + int(np.sign(v))
            if 0 <= old < velocity.size:
                mat[row, old] += abs(v) / h
            if 0 <= new < velocity.size:
                mat[row, new] -= abs(v) / h
        for side, row in enumerate((0, velocity.size - 1)):
            half_cell = h / (2 * s['kappa_' + axis][row])
            for r, sign in ((robin[side], -1), (new_robin[side], 1)):
                if r:
                    mat[row, row] += sign / (h * (half_cell + 1 / r))
        ax, by = (mat, gt['By']) if axis == 'x' else (gt['Ax'], mat)
        t['mu_shifts'] = np.asarray(s['mu_shifts'])[::-1]
        nr = len(s['sensor_ij'])
        order = np.arange(len(s['y_obs'])).reshape(-1, nr)[::-1].ravel()
        z, y = s['Z_nuisance'][order], s['y_obs'][order]
        t['Z_nuisance'], t['y_obs'] = z, y
        i, j = s['sensor_ij'].T
        eye = np.eye(ax.shape[0])
        blocks = []
        for mu in t['mu_shifts']:
            fields = np.stack([solve_sylvester(ax + mu * eye, by.T, q) for q in s['source_modes']])
            blocks.append(fields[:, i, j].T)
        sensor = np.concatenate(blocks)
        sensor_error = np.max(np.abs(solver._transport(t)[0] - sensor))
        assert sensor_error < 1e-10, (axis, sensor_error)
        # Independent SVD form of the ordinary ridge residual map.
        q, d, _ = np.linalg.svd(z, full_matrices=False)
        values = np.column_stack((sensor, y))
        denom = d*d + s['nuisance_ridge']
        shrink = np.divide(d*d, denom, out=np.zeros_like(d), where=denom > 0)
        prof = values - q @ (shrink[:, None] * (q.T @ values))
        out = solver.solve_multibudget_kronecker_transfer(t)
        w = out['weights_by_budget']
        field_error = np.max(np.abs(out['terminal_field_mid_budget'] - np.einsum('m,mij->ij', w[1], fields)))
        assert field_error < 1e-10, (axis, field_error)
        for budget, weights, cert in zip(t['budget_grid'], w, out['kkt_certificates']):
            np.testing.assert_array_equal(weights, cert['weights'])
            check = verify(prof[:, :-1], prof[:, -1], t['source_ridge'], t['u_box'], budget, cert)
            assert check['valid'], (axis, budget, check)
            worst_gap = max(worst_gap, check['gap'])
        worst_sensor = max(worst_sensor, sensor_error)
        worst_field = max(worst_field, field_error)
        slices += 1
print(f'{slices} public-derived mixed-upwind/asymmetric-Robin slices: max sensor error {worst_sensor:.3e}, max terminal-field error {worst_field:.3e}, worst KKT gap {worst_gap:.3e}')
