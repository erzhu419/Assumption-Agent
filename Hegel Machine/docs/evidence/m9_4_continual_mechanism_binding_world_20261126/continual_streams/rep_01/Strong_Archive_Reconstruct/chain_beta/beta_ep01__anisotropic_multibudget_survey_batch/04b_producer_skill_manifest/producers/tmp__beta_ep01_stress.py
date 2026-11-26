import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'beta_ep01_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(root + 'public_beta_ep01.pkl', 'rb') as f:
    case = pickle.load(f)[0]
s, gt = case['study'], case['gt']
S, y = gt['S_tilde'], gt['r_y']
cap, ridge = s['u_box'], s['source_ridge']
full = cap * S.shape[1]

# Re-express the measured readouts in different units for each bank.
sweep = s['sensor_calibration_sweeps']
scale = np.array([0.55, 1.8, 0.85, 1.3])
shift = np.array([0.2, -0.35, 0.7, -0.15])
b, br = sweep['bank_id'], s['row_bank_id']
reencoded = dict(s, sensor_calibration_sweeps=dict(sweep, raw_readout=scale[b]*sweep['raw_readout']+shift[b]), y_obs_raw=scale[br]*s['y_obs_raw']+shift[br])
base = mod.solve_beta_ep01(s)
other = mod.solve_beta_ep01(reencoded)
for key in ('weights_by_budget', 'terminal_field_mid_budget'):
    err = np.max(np.abs(base[key]-other[key]))
    print(f'bank unit invariance {key}: {err:.3e}')
    assert err < 1e-10

# Keep the real survey design and observations. Move the budget toward the
# limits to exercise upper constraints not encountered in the public grid.
covered = set()
worst_gap = 0.
for budgets in (s['budget_grid'], [0., cap*1e-6, cap], [full-2*cap, full-cap*1e-6, full]):
    out = mod.solve_beta_ep01(dict(s, budget_grid=budgets))
    for budget, w, cert in zip(budgets, out['weights_by_budget'], out['kkt_certificates']):
        nu = cert['nu_sum']
        lo, hi = cert['lam_lower'], cert['lam_upper']
        r = S @ w - y
        g = S.T @ r + ridge*w
        obj = 0.5*(r@r + ridge*(w@w))
        bound = obj - g@w + nu*budget + cap*np.minimum(g-nu, 0.).sum()
        gap = obj-min(bound, cert['dual_lower_bound'])
        stat = np.max(np.abs(g-nu-lo+hi))
        comp = max(np.max(np.abs(lo*w)), np.max(np.abs(hi*(cap-w))))
        low, high = int(sum(w == 0)), int(sum(w == cap))
        if 0 < budget < full:
            if low: covered.add('lower')
            if high: covered.add('upper')
        print(f'budget={budget:.9g} lower={low} upper={high} gap={gap:.3e} stationarity={stat:.3e} complementarity={comp:.3e}')
        assert abs(w.sum()-budget) < 1e-7 and w.min() >= 0 and w.max() <= cap
        assert np.array_equal(w, cert['weights']) and min(lo.min(), hi.min()) >= 0
        assert abs(cert['primal_objective']-obj) < 1e-6 and cert['dual_lower_bound'] <= bound+1e-6
        assert -1e-9 <= gap < 1e-6 and max(stat, comp) < 1e-6
        worst_gap = max(worst_gap, gap)
assert covered == {'lower', 'upper'}
print(f'worst public-measurement stress gap: {worst_gap:.3e}')
