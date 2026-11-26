import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'multibudget_kronecker_kkt_transfer_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
with (root / 'public_suc02_T_studies.pkl').open('rb') as f:
    studies = pickle.load(f)
worst_gap = worst_weight = worst_field = 0.
rng = np.random.default_rng(917)
for item in studies:
    s, gt = item['study'], item['gt']
    nk = len(s['mu_shifts'])
    nr = len(s['sensor_ij'])
    perm = np.r_[rng.permutation(nk-1), nk-1]
    rows = (perm[:, None] * nr + np.arange(nr)).ravel()
    order = [2, 1, 0]
    z = s['Z_nuisance']
    q, _ = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))
    for axes in ('x', 'y', 'xy'):
        t = dict(s)
        t['sensor_ij'] = np.array(s['sensor_ij'], copy=True)
        field_slice = [slice(None), slice(None)]
        for a in axes:
            d = 'xy'.index(a)
            field_slice[d] = slice(None, None, -1)
            t['kappa_'+a] = s['kappa_'+a][::-1]
            t['vel_'+a] = -s['vel_'+a][::-1]
            t['robin_'+a] = np.broadcast_to(s['robin_'+a], (2,))[::-1]
            t['sensor_ij'][:, d] = s['n'+a] - 1 - s['sensor_ij'][:, d]
        t['source_modes'] = s['source_modes'][(slice(None), *field_slice)]
        t['mu_shifts'] = np.asarray(s['mu_shifts'])[perm]
        t['y_obs'] = np.asarray(s['y_obs'])[rows]
        t['Z_nuisance'] = z[rows] @ q
        t['budget_grid'] = np.asarray(s['budget_grid'])[order]
        out = mod.solve_multibudget_kronecker_transfer(t)
        ew = np.max(np.abs(out['weights_by_budget'] - gt['weights_by_budget'][order]))
        ef = np.max(np.abs(out['terminal_field_mid_budget'] - gt['terminal_field_mid_budget'][tuple(field_slice)]))
        assert ew < 1e-8 and ef < 1e-9, (axes, ew, ef)
        worst_weight, worst_field = max(worst_weight, ew), max(worst_field, ef)
        for b, cert in zip(t['budget_grid'], out['kkt_certificates']):
            ck = verify(gt['S_tilde'][rows], gt['r_y'][rows], t['source_ridge'], t['u_box'], b, cert)
            assert ck['valid'], (axes, b, ck)
            worst_gap = max(worst_gap, ck['gap'])
print(f'{len(studies)} public studies, 3 reflection slices each: worst weight error {worst_weight:.3e}, field error {worst_field:.3e}, KKT gap {worst_gap:.3e}')
