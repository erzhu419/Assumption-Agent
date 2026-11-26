import importlib.util
import itertools
import json
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root/'beta_w04_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ns = {}
exec((root/'validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
item = pickle.load(open(root/'public_beta_w04_batch_01.pkl', 'rb'))[0]
s, gt = item['study'], item['gt']
archives = json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for label, sweeps in [('public', s['sensor_calibration_sweeps'])] + [(a.get('episode_short_name', 'historical'), a) for a in archives]:
    bank, ref, raw = m._sweep_arrays(sweeps)
    for b in np.unique(bank):
        x, y = ref[bank == b], raw[bank == b]
        worst = 0.0
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            if m._resolved_gain(x[keep]):
                g, c = m._fit_bank(x[keep], y[keep])
                worst = max(worst, abs(g*x[i] + c - y[i]))
        print(label, 'bank', b, 'held-out calibration residual', worst)

# Profiled normal equations must equal explicit minimization over nuisance.
D, _ = m._forward(s)
z = s['Z_nuisance']
A = np.column_stack((D, m._calibrate(s)))
ridge = s['nuisance_ridge']
P = m._profile(z, A, ridge)
B = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ A)
E = A - z @ B
err = np.max(np.abs(P.T @ P - (E.T @ E + ridge * (B.T @ B))))
print('explicit-vs-profiled Gram maximum error', err)
assert err < 1e-10
v = s['V_reaction_2d']
print('reaction additive maximum residual', np.max(np.abs(v-v.mean(1)[:,None]-v.mean(0)[None,:]+v.mean())))
cap = s['u_box']; total = cap * D.shape[1]
for budgets in ([0, cap, total], [cap*1e-5, total/2, total-cap*1e-5]):
    out = m.solve_beta_w04_batch_01(dict(s, budget_grid=budgets))
    w = out['weights_by_budget']
    checks = [verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], cap, b, c, gap_tol=1e-6) for b, c in zip(budgets, out['kkt_certificates'])]
    print('boundary budgets', budgets, 'maximum reference gap', max(c['gap'] for c in checks), 'maximum mass error', np.max(np.abs(w.sum(1)-budgets)))
    assert all(c['valid'] for c in checks), checks

# Thin the observed public sweeps and inject one EMI spike in every position.
bank, ref, raw = m._sweep_arrays(s['sensor_calibration_sweeps'])
for n in (4, 6, 12):
    worst = (0.0, None)
    for b in range(4):
        x, y = ref[bank == b], raw[bank == b]
        for take in itertools.combinations(range(len(x)), n):
            xx, yy = x[list(take)], y[list(take)]
            for j in range(n):
                zz = (1.18 * yy + .07).copy(); zz[j] += (-1)**j * .8
                fit = m._fit_bank(xx, zz)
                error = max(abs(fit[0]-1.18), abs(fit[1]-.07))
                if error > worst[0]:
                    worst = (error, (b, take, j, fit))
    print('one-spike stress', n, 'points, worst affine error and witness', worst)
