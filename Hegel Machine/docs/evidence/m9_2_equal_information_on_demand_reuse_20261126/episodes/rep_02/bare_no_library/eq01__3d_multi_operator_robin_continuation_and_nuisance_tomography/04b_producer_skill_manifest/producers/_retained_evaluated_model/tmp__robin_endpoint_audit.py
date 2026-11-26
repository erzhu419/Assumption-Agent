import runpy
import time
import numpy as np

ns = runpy.run_path('/tmp/robin_stress.py')
mod = ns['mod']
original = mod.cg

def flux_residual(s, theta, solution, rhs):
    shape = s['kappa_base'].shape
    h = np.array([s['Lx'], s['Ly'], s['Lz']]) / shape
    v = np.prod(h)
    k = s['kappa_base'] + theta * s['kappa_mod1'] + theta**2 * s['kappa_mod2']
    u = solution.reshape(shape)
    out = v * (s['sigma_base'] + theta * s['sigma_mod']) * u
    for axis, spacing in enumerate(h):
        f, g, m = (np.moveaxis(a, axis, 0) for a in (u, k, out))
        flow = v / spacing**2 * (f[:-1] - f[1:]) / (.5/g[:-1] + .5/g[1:])
        m[:-1] += flow
        m[1:] -= flow
        for i in (0, -1):
            m[i] += v / spacing * f[i] / (1/s['gamma_robin'] + .5*spacing/g[i])
    return np.linalg.norm(out.ravel() - rhs) / np.linalg.norm(rhs)

elapsed = 0.
for pattern in ('channels', 'switching_layers'):
    s = ns['study']((28, 28, 28), int(pattern == 'channels'))
    i,j,k = np.indices((28,28,28))
    if pattern == 'channels':
        high = ((i % 7 == 2) & (j % 7 == 3)) | ((k % 9 == 4) & (j % 7 == 3))
        activates = ((i + j) % 11 < 4) & (k > 7)
    else:
        high = ((i + 2*j) // 5) % 4 == 1
        activates = ((j + k) // 5) % 3 == 1
    s['kappa_base'] = np.where(high, 1., 1/3500)
    s['kappa_mod1'] = np.where((j+k) % 9 < 2, .045, 1e-8)
    s['kappa_mod2'] = np.where(activates & ~high, 1.15, 1e-8)
    order = np.sort(s['thetas'])
    solves = []
    def observed(a, b, *args, **kwargs):
        result = original(a, b, *args, **kwargs)
        if 'x0' in kwargs:
            solves.append([result[0], b, None])
        else:
            solves[-1][2] = result[0]
        return result
    mod.cg = observed
    t0 = time.perf_counter()
    out = mod.solve_robin_continuation_tomography(s)
    dt = time.perf_counter() - t0
    elapsed += dt
    residuals = [flux_residual(s, order[idx//3], x if dx is None else x + dx, b)
                 for idx,(x,b,dx) in enumerate(solves)]
    assert len(residuals) == 27
    p, w = ns['check_inverse'](s, out, out['continuation_sensor_predictions'])
    print(pattern, 'solver seconds', round(dt,3), 'worst independent flux residual', max(residuals), 'profile/joint errors', p,w, flush=True)
    assert max(residuals) < 1e-7 and p < 1e-5 and w < .03
print('Audit pair solver seconds', elapsed, flush=True)
assert elapsed <= 25.
