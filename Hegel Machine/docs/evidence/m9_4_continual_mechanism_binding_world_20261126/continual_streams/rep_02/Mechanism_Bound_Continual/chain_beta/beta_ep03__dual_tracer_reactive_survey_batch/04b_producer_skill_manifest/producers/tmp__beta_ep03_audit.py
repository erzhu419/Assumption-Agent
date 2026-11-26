import importlib.util
import pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'beta_ep03_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Assemble the stencil by adding individual face and cell contributions.
def axis(s, dim):
    n = int(s['n' + dim]); h = s['L' + dim] / n
    k, v, rv = [s[key + '_' + dim] for key in ('kappa', 'vel', 'robin')]
    rv = np.broadcast_to(rv, (2,))
    a = np.diag(s['reaction_' + dim]).astype(float)
    for p in range(n - 1):
        d = 2 / (1 / k[p] + 1 / k[p + 1]) / h**2
        a[p, p] += d; a[p + 1, p + 1] += d
        a[p, p + 1] -= d; a[p + 1, p] -= d
    for p, vp in enumerate(v):
        q = p - 1 if vp >= 0 else p + 1
        a[p, p] += abs(vp) / h
        if 0 <= q < n:
            a[p, q] -= abs(vp) / h
    for p, r in zip((0, n - 1), rv):
        a[p, p] += r * k[p] / (h * (k[p] + r * h / 2))
    return a

for case in pickle.load(open(root + 'public_beta_ep03.pkl', 'rb')):
    s, gt = case['study'], case['gt']
    design, _ = mod._forward(s)
    out = mod.solve_beta_ep03(s)
    nx, ny = s['nx'], s['ny']
    ax, by = axis(s, 'x'), axis(s, 'y')
    base = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    modes = s['source_modes']; rhs = modes.reshape(len(modes), -1).T
    sensors = np.ravel_multi_index(np.asarray(s['sensor_ij']).T, (nx, ny))
    rows = 0
    for ch in 'ab':
        v = s['V_channel_' + ch]
        interaction = v - v.mean(0) - v.mean(1, keepdims=True) + v.mean()
        worst = 0.
        for mu in s['mu_shifts']:
            op = base + diags(v.ravel() + mu, format='csc')
            ref = splu(op).solve(rhs)[sensors]
            worst = max(worst, np.linalg.norm(design[rows:rows+len(sensors)] - ref) / np.linalg.norm(ref))
            rows += len(sensors)
        field = out['terminal_field_channel_' + ch]
        source = rhs @ out['weights']
        rel_res = np.linalg.norm(op @ field.ravel() - source) / np.linalg.norm(source)
        print(ch, 'interaction ratio', np.linalg.norm(interaction) / np.linalg.norm(v), 'worst shift sensor error', worst, 'terminal equation residual', rel_res)
        assert max(worst, rel_res) < 1e-9
    sw = s['sensor_calibration_sweeps']
    banks, x, y = (np.asarray(sw[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    for b in np.unique(banks):
        xb, yb = x[banks == b], y[banks == b]
        a = np.column_stack((xb, np.ones_like(xb)))
        errors = []
        for j in range(len(xb)):
            train = np.arange(len(xb)) != j
            p = np.linalg.lstsq(a[train], yb[train], rcond=None)[0]
            errors.append(abs(a[j] @ p - yb[j]))
        print('bank', b, 'worst held-out calibration error', max(errors), 'fraction of sweep span', max(errors) / np.ptp(yb))
    design, y = mod._profile(design, mod._calibrate(s), s['Z_nuisance'], s['nuisance_ridge'])
    residual = design @ out['weights'] - y
    reference = gt['S_tilde'] @ gt['w_opt'] - gt['r_y']
    for b in np.unique(s['row_bank_id']):
        mask = s['row_bank_id'] == b
        delta = np.max(np.abs(residual[mask] - reference[mask]))
        print('bank', b, 'profiled residual max deviation from public reference', delta)
        assert delta < 1e-6
