import importlib.util
import pickle
from pathlib import Path
import numpy as np


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def axis(kap, vel, h, robin):
    n = len(kap)
    mat = np.zeros((n, n))
    for i in range(n):
        if i == 0 or i == n - 1:
            mat[i, i] = kap[i] / h**2 + robin / h
            mat[i, 1 if i == 0 else n - 2] = -kap[i] / h**2
        else:
            km = (kap[i - 1] + kap[i]) / 2
            kp = (kap[i] + kap[i + 1]) / 2
            mat[i, i-1] = -km / h**2 - max(0, vel[i]) / h
            mat[i, i+1] = -kp / h**2 - max(0, -vel[i]) / h
            mat[i, i] = (km + kp) / h**2 + abs(vel[i]) / h
    return mat


p = Path('/holdout_task')
m = load(p / 'batch_solver.py')
archive = Path('/tmp/hegel_continual_archive')
for item in pickle.load((p / 'public_batch.pkl').open('rb')):
    s = item['study']
    nx, ny = s['nx'], s['ny']
    ax, ay = (axis(s['kappa_' + a], s['vel_' + a], s['L' + a] / (n - 1), s['robin_' + a]) for a, n in [('x', nx), ('y', ny)])
    iy, ix = np.eye(ny), np.eye(nx)
    op = np.kron(iy, ax) + np.kron(ay, ix) + np.diag(s['V_reaction_2d'].ravel(order='F'))
    rhs = np.column_stack([u.ravel(order='F') for u in s['source_modes']])
    ij = np.asarray(s['sensor_ij'])
    rows = nx * ij[:, 1] + ij[:, 0]
    ref = [np.linalg.solve(op + mu * np.eye(nx * ny), rhs) for mu in s['mu_shifts']]
    design, _ = m._forward(s)
    expected = np.vstack([u[rows] for u in ref])
    out = m.solve_batch(s)
    term = out['terminal_field'].ravel(order='F')
    terminal_resid = (op + s['mu_shifts'][-1] * np.eye(nx * ny)) @ term - rhs @ out['weights']
    print('max sensor design error', np.max(abs(design - expected)), 'max terminal PDE residual', np.max(abs(terminal_resid)))
    np.testing.assert_allclose(design, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(terminal_resid, 0, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(s['row_bank_id'].reshape(-1, 4), np.broadcast_to(np.arange(4), (len(design) // 4, 4)))
    y, z = m._calibrate(s), s['Z_nuisance']
    zr = s['nuisance_ridge'] * np.eye(z.shape[1])
    shift = np.repeat(np.arange(len(ref)), len(rows))
    for tag, a in [('candidate', design)] + [(tag, load(next(archive.glob('*' + tag + '_solver.py')))._forward(s)[0]) for tag in ('batch_01', 'batch_02', 'batch_03')]:
        errs = []
        for k in range(len(ref)):
            test = shift == k
            b, c, d = a[~test], y[~test], z[~test]
            proj = np.linalg.solve(d.T @ d + zr, d.T)
            at, yt = b - d @ (proj @ b), c - d @ (proj @ c)
            w, _ = m._box_simplex_qp(at.T @ at + s['source_ridge'] * np.eye(at.shape[1]), -at.T @ yt, s['u_box'], s['total_budget'])
            gamma = proj @ (c - b @ w)
            errs.append(np.sqrt(np.mean((a[test] @ w + z[test] @ gamma - y[test])**2)))
        print(tag, 'held-out shift RMSE', np.round(errs, 8), 'worst', max(errs))
