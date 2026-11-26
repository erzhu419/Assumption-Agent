import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'beta_ep01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_beta_ep01.pkl', 'rb') as f:
    case = pickle.load(f)[0]
s, gt = case['study'], case['gt']
sweep = s['sensor_calibration_sweeps']
S, y = gt['S_tilde'], gt['r_y']
ridge, cap = s['source_ridge'], s['u_box']
budgets = s['budget_grid']
worst = {}
for i, bank in enumerate(sweep['bank_id']):
    keep = np.arange(len(sweep['bank_id'])) != i
    reduced = {k: v[keep] for k, v in sweep.items()}
    out = m.solve_beta_ep01(dict(s, sensor_calibration_sweeps=reduced))
    for j, (B, w, cert) in enumerate(zip(budgets, out['weights_by_budget'], out['kkt_certificates'])):
        g = S.T @ (S @ w - y) + ridge*w
        nu = cert['nu_sum']
        gap = g @ w - nu*B - cap*np.minimum(g-nu, 0.).sum()
        error = np.max(np.abs(w - gt['weights_by_budget'][j]))
        assert abs(w.sum()-B) < 1e-7 and w.min() >= 0 and w.max() <= cap
        assert -1e-9 <= gap < 1e-6 and error < 1e-5, (i, bank, B, gap, error)
        for name, value in [('reference KKT gap', gap), ('reference weight error', error)]:
            if name not in worst or value > worst[name][0]:
                worst[name] = (value, i, bank, B)
    field_error = np.max(np.abs(out['terminal_field_mid_budget'] - gt['terminal_field_mid_budget']))
    assert field_error < 1e-6, (i, bank, field_error)
    if 'reference terminal field error' not in worst or field_error > worst['reference terminal field error'][0]:
        worst['reference terminal field error'] = (field_error, i, bank, budgets[1])
for name, (value, i, bank, B) in worst.items():
    print(f'{name}: {value:.3e} (omitted sweep={i}, bank={bank}, budget={B:g})')
print(f'Passed all {len(sweep["bank_id"])} leave-one-calibration-point-out folds.')
