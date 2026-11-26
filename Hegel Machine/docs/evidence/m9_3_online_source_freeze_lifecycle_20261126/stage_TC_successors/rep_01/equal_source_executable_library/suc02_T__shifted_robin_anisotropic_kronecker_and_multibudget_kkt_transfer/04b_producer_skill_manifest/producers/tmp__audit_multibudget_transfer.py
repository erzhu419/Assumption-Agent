import importlib.util
import pickle
from pathlib import Path

import numpy as np
from scipy.linalg import solve_sylvester

root = Path('/holdout_task')
path = root / 'multibudget_kronecker_kkt_transfer_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with (root / 'public_suc02_T_studies.pkl').open('rb') as f:
    items = pickle.load(f)
worst_gap = worst_readout = worst_stationarity = worst_comp = 0.0
for k, item in enumerate(items):
    st, gt = item['study'], item['gt']
    a, b = gt['Ax'], gt['By']
    si, sj = st['sensor_ij'].T
    identity = np.eye(a.shape[0])
    design, _ = mod._transport(st)
    for t, shift in enumerate(st['mu_shifts']):
        ref = np.column_stack([solve_sylvester(a + shift * identity, b.T, mode)[si, sj] for mode in st['source_modes']])
        err = float(np.max(np.abs(ref - design[t*len(si):(t+1)*len(si)])))
        assert err < 1e-10, (k, t, err)
        worst_readout = max(worst_readout, err)
    out = mod.solve_multibudget_kronecker_transfer(st)
    s, y = gt['S_tilde'], gt['r_y']
    cap, ridge = st['u_box'], st['source_ridge']
    for i, (budget, cert) in enumerate(zip(st['budget_grid'], out['kkt_certificates'])):
        w = out['weights_by_budget'][i]
        assert np.all(w >= -1e-12) and np.all(w <= cap + 1e-12) and abs(w.sum() - budget) < 1e-12
        assert np.array_equal(w, cert['weights'])
        r = s @ w - y
        j = 0.5 * (r @ r + ridge * (w @ w))
        g = s.T @ r + ridge * w
        # Solve the linearized problem directly by filling cheapest coordinates first.
        z = np.zeros_like(w)
        z[np.argsort(g)] = np.clip(budget - cap * np.arange(len(w)), 0, cap)
        optimum_tangent_gap = float(g @ (w - z))
        gap = j - cert['dual_lower_bound']
        assert -1e-12 <= gap <= 1e-6 and abs(gap - optimum_tangent_gap) < 1e-10, (k, i, gap, optimum_tangent_gap)
        lower, upper = cert['lam_lower'], cert['lam_upper']
        assert np.min(lower) >= 0 and np.min(upper) >= 0
        stat = float(np.max(np.abs(g - cert['nu_sum'] - lower + upper)))
        comp = float(lower @ w + upper @ (cap - w))
        assert stat < 1e-10 and abs(comp - gap) < 1e-10, (k, i, stat, comp, gap)
        worst_gap = max(worst_gap, gap)
        worst_stationarity = max(worst_stationarity, stat)
        worst_comp = max(worst_comp, comp)
    print(f'study {k}: {len(st["mu_shifts"])} shifts, {len(st["budget_grid"])} budgets, nuisance shape={st["Z_nuisance"].shape}, batch/cohort keys={[key for key in st if "batch" in key or "cohort" in key]}')
print(f'worst readout error={worst_readout:.3e}; worst KKT gap={worst_gap:.3e}; worst stationarity={worst_stationarity:.3e}; worst complementarity={worst_comp:.3e}')
