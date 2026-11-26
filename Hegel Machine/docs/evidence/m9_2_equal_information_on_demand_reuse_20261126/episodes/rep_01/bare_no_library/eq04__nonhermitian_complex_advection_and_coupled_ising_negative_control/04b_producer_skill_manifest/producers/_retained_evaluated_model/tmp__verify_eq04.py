import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

path = Path('/holdout_task/nonhermitian_advection_coupled_select.py')
spec = importlib.util.spec_from_file_location('candidate', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference_operator(p):
    nx, ny = p['vx'].shape
    hx, hy = p['Lx'] / (nx + 1), p['Ly'] / (ny + 1)
    a = lil_matrix((nx * ny, nx * ny), dtype=np.complex128)
    for i, j in np.ndindex(nx, ny):
        k = i * ny + j
        a[k, k] = p['sigma'][i, j] + 1j * p['omega']
        for di, dj, h, v in [(1, 0, hx, p['vx'][i, j]), (0, 1, hy, p['vy'][i, j])]:
            a[k, k] += 2 * p['nu'] / h**2 + abs(v) / h
            for sign in (-1, 1):
                ii, jj = i + sign * di, j + sign * dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    a[k, ii * ny + jj] = -p['nu'] / h**2 - max(-sign * v, 0) / h
    return a.tocsr()


def objective(p, g, y, w):
    residual = g @ w - y
    active = abs(w) > 1e-8
    return (0.5 * np.vdot(residual, residual).real + 0.5 * p['lam2'] * np.vdot(w, w).real
            + p['lam0'] * active.sum() + sum(c for u, v, c in p['pair_penalties'] if active[u] and active[v]))


def reference(p, g, y):
    b = g.conj().T @ y
    h = g.conj().T @ g + p['lam2'] * np.eye(11)
    best = (np.inf, None)
    second = np.inf
    for mask in range(2048):
        s = [i for i in range(11) if (mask >> i) & 1]
        w = np.zeros(11, dtype=complex)
        if s:
            w[s] = solve(h[np.ix_(s, s)], b[s], assume_a='pos')
            if np.any(abs(w[s]) <= 1e-8):
                continue
        value = objective(p, g, y, w)
        if value < best[0]:
            second = best[0]
            best = (value, w)
        elif value < second:
            second = value
    return *best, second - best[0]


rng = np.random.default_rng(20504)
max_errors = np.zeros(5)
for case in ('mixed-sign', 'anisotropic', 'low-diffusion', 'duplicate-columns', 'empty', 'pair-coupled', 'pair-free'):
    p = dict(Lx=1.7, Ly=0.8, nu=0.009, omega=7.1, lam2=0.003, lam0=0.001)
    if case == 'anisotropic':
        p.update(Lx=3.1, Ly=0.21, omega=-70, nu=0.03)
    if case == 'low-diffusion':
        p.update(nu=1e-6, omega=-0.3)
    p.update(vx=rng.uniform(-2, 2, (32, 32)), vy=rng.uniform(-1, 1, (32, 32)),
             sigma=rng.uniform(-0.15, 0.4, (32, 32)), basis=rng.normal(size=(11, 32, 32)),
             pair_penalties=[(0, 1, .09), (1, 6, .02), (6, 4, .008), (4, 9, .01)])
    if case == 'duplicate-columns':
        p['lam2'] = 1e-7
        p['basis'][1] = p['basis'][0] + 1e-8 * p['basis'][1]
    if case in ('pair-coupled', 'pair-free'):
        p.update(vx=np.zeros((32, 32)), vy=np.zeros((32, 32)), sigma=np.zeros((32, 32)), lam0=.02, lam2=.01)
        if case == 'pair-free':
            p['pair_penalties'] = []
        else:
            p['pair_penalties'] = [(0, 1, 2.0)]
        idx = np.arange(11) * 76 + 34
    else:
        idx = np.unique(np.r_[0, 31, 992, 1023, rng.choice(1024, size=65, replace=False)])
    p['sensor_ij'] = np.column_stack(np.unravel_index(idx, (32, 32)))
    a = reference_operator(p)
    if case in ('pair-coupled', 'pair-free'):
        # Delta forcing at widely separated sensors makes the singleton gains comparable.
        p['basis'].fill(0)
        p['basis'].reshape(11, -1)[np.arange(11), idx] = abs(a[0, 0])
    rhs = p['basis'].reshape(11, -1).T
    fields = spsolve(a, rhs)
    g = fields[idx]
    w0 = np.zeros(11, dtype=complex)
    w0[[0, 1, 6]] = [1+0.2j, -0.5+0.9j, 0.7-0.1j]
    if case in ('pair-coupled', 'pair-free'):
        w0[[0, 1, 6]] = [1.05+0.2j, -0.5+0.9j, 0]
    y = g @ w0
    if case == 'empty':
        y *= 0
    p.update(y_obs_re=y.real, y_obs_im=y.imag)
    expected, w_ref, gap = reference(p, g, y)
    start = time.perf_counter()
    out = module.solve_nonhermitian_coupled_inversion(p)
    elapsed = time.perf_counter() - start
    assert set(out) == {'weights_real', 'weights_imag', 'objective', 'predicted_field_abs'}
    for name, shape in [('weights_real', (11,)), ('weights_imag', (11,)), ('predicted_field_abs', (32, 32))]:
        assert out[name].shape == shape and out[name].dtype.kind == 'f' and np.isfinite(out[name]).all()
    assert isinstance(out['objective'], float)
    w = out['weights_real'] + 1j * out['weights_imag']
    f_ref = abs((fields @ w_ref).reshape(32, 32))
    fields_candidate = module._basis_responses(p, p['basis'].astype(complex))
    errors = [abs(out['objective'] - expected), abs(out['objective'] - objective(p, g, y, w)),
              np.linalg.norm(out['predicted_field_abs'] - f_ref) / max(np.linalg.norm(f_ref), 1e-15),
              np.linalg.norm(a @ fields_candidate - rhs) / np.linalg.norm(rhs), np.linalg.norm(w - w_ref)]
    max_errors = np.maximum(max_errors, errors)
    support = np.flatnonzero(abs(w) > 1e-8).tolist()
    print(case, 'errors', ['%.2e' % e for e in errors], 'support', support, 'reference gap', '%.2e' % gap, 'seconds', '%.3f' % elapsed)
    assert max(errors[:2]) <= 1e-6 and errors[2] <= 1e-5 and errors[3] < 1e-10 and errors[4] < 1e-5
    if case == 'pair-coupled':
        assert support == [0]
    elif case == 'pair-free':
        assert support == [0, 1]
    elif case == 'empty':
        assert not support
print('worst errors:', max_errors)
