import importlib.util
import time
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve

path = '/holdout_task/robin_continuation_tomography.py'
spec = importlib.util.spec_from_file_location('robin', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def study(shape, variant):
    rng = np.random.default_rng(78 + variant)
    x, y, z = np.indices(shape) / np.array(shape)[:, None, None, None]
    if variant == 1:
        base = np.where(x + .15 * np.sin(10*y) < .55, 1., 1/3500)
        m1 = np.where(y < .48, 1e-7, .025)
        m2 = np.where((z > .40) & (x > .45), .65, 1e-8)
    else:
        base = np.exp(np.log(3500) * rng.uniform(-1, 0, shape))
        m1 = .015 * np.exp(rng.uniform(-5, 0, shape))
        m2 = .08 * np.exp(rng.uniform(-5, 0, shape))
    thetas = np.array([.48, 1.85, .16, 1.6, .94, .15, 1.12, .67, .28])
    sensors = np.column_stack([rng.integers(d, size=55) for d in shape])
    n = len(thetas) * len(sensors)
    Z = rng.normal(size=(n, 7))
    Z[:, -1] = Z[:, 0] - 2 * Z[:, 1] + 1e-7 * Z[:, -1]
    return dict(Lx=1.21, Ly=.77, Lz=1.82, kappa_base=base, kappa_mod1=m1, kappa_mod2=m2,
                sigma_base=(1e-4 if variant == 1 else .001) * (1 + x), sigma_mod=.0003 * (1 + y),
                gamma_robin=.008 if variant == 1 else 1.8, thetas=thetas,
                source_modes=np.array([1 + x, np.sin(6*y) + .1*z, rng.normal(size=shape)]),
                sensor_ijk=sensors, Z_nuisance=Z, Y_obs=rng.normal(size=(n, 5)), nuisance_ridge=1e-5,
                source_ridge=.002)


def independent_matrix(s, theta):
    kap = s['kappa_base'] + theta*s['kappa_mod1'] + theta*theta*s['kappa_mod2']
    shape = kap.shape
    h = np.array([s['Lx'], s['Ly'], s['Lz']]) / shape
    vol = np.prod(h)
    sigma = s['sigma_base'] + theta*s['sigma_mod']
    gamma = s['gamma_robin']
    rows, cols, values = [], [], []
    for ijk in np.ndindex(shape):
        i = np.ravel_multi_index(ijk, shape)
        k = kap[ijk]
        diagonal = sigma[ijk] * vol
        for a in range(3):
            for direction in [-1, 1]:
                neighbor = list(ijk)
                neighbor[a] += direction
                if 0 <= neighbor[a] < shape[a]:
                    neighbor = tuple(neighbor)
                    edge = vol / h[a]**2 / (.5/k + .5/kap[neighbor])
                    rows.append(i); cols.append(np.ravel_multi_index(neighbor, shape)); values.append(-edge)
                    diagonal += edge
                else:
                    diagonal += (vol/h[a]) / (1/gamma + .5*h[a]/k)
        rows.append(i); cols.append(i); values.append(diagonal)
    return sparse.csr_matrix((values, (rows, cols)), shape=(kap.size, kap.size)), vol


def check_inverse(s, output, S):
    Z, Y = s['Z_nuisance'], s['Y_obs']
    n, q = Z.shape
    u, singular, _ = linalg.svd(Z, full_matrices=True)
    lamz = s['nuisance_ridge']
    weights = np.ones(n)
    weights[:q] = np.sqrt(lamz/(singular**2 + lamz))
    design = weights[:, None] * (u.T @ S)
    obs = weights[:, None] * (u.T @ Y)
    amp = (design.T @ obs) / np.sum(design**2, axis=0)[:, None]
    A = np.column_stack((Z, S))
    regularization = np.r_[np.full(q, lamz), np.full(S.shape[1], s['source_ridge'])]
    coef = np.linalg.lstsq(np.vstack((A, np.diag(np.sqrt(regularization)))),
                           np.vstack((Y, np.zeros((A.shape[1], Y.shape[1])))), rcond=None)[0]
    return [np.linalg.norm(output[key]-ref)/np.linalg.norm(ref) for key, ref in
            [('profiled_amplitudes', amp), ('joint_source_weights', coef[q:])]]


def tiny():
    s = study((4, 3, 2), 1)
    out = mod.solve_robin_continuation_tomography(s)
    expected = []
    for theta in s['thetas']:
        a, vol = independent_matrix(s, theta)
        fields = spsolve(a, vol*s['source_modes'].reshape(3, -1).T).T.reshape(s['source_modes'].shape)
        expected.append(fields[(slice(None), *s['sensor_ijk'].T)].T)
    S = np.concatenate(expected)
    e = np.linalg.norm(out['continuation_sensor_predictions']-S)/np.linalg.norm(S)
    t = np.linalg.norm(out['terminal_fields']-fields)/np.linalg.norm(fields)
    amp, joint = check_inverse(s, out, S)
    print('independent small-grid errors sensor terminal profile joint:', e, t, amp, joint, flush=True)
    assert max(e,t,amp,joint) < 1e-8


def full():
    original_cg, original_factor = mod.cg, mod._factor
    counts, residuals, factors = [], [], []
    def watched_cg(a, b, *args, **kwargs):
        count = 0
        def count_step(x):
            nonlocal count
            count += 1
        x, status = original_cg(a,b,*args,callback=count_step,**kwargs)
        if 'x0' in kwargs:
            counts.append(count)
            residuals.append(np.linalg.norm(a@x-b)/np.linalg.norm(b))
        return x, status
    def watched_factor(*args):
        start = time.perf_counter()
        factor = original_factor(*args)
        factors.append(time.perf_counter()-start)
        return factor
    mod.cg, mod._factor = watched_cg, watched_factor
    elapsed = 0
    for variant in (0, 1):
        s = study((28, 28, 28), variant)
        counts.clear(); factors.clear(); residuals.clear()
        start = time.perf_counter()
        out = mod.solve_robin_continuation_tomography(s)
        seconds = time.perf_counter() - start
        elapsed += seconds
        print('full variant', variant, 'seconds', round(seconds,3), 'factors', np.round(factors,3), 'iterations', np.reshape(counts, (-1,3)).tolist(), 'max primary residual', max(residuals), flush=True)
        a, vol = independent_matrix(s, s['thetas'][-1])
        b = vol * s['source_modes'].reshape(3, -1).T
        residual = np.max(np.linalg.norm(a @ out['terminal_fields'].reshape(3,-1).T-b, axis=0) / np.linalg.norm(b,axis=0))
        amp, joint = check_inverse(s, out, out['continuation_sensor_predictions'])
        print('independent terminal residual, profile error, joint error:', residual, amp, joint, flush=True)
        assert residual < 1e-7 and amp < 1e-5 and joint < .03
    print('full total', elapsed, flush=True)

if __name__ == '__main__':
    tiny()
    full()
