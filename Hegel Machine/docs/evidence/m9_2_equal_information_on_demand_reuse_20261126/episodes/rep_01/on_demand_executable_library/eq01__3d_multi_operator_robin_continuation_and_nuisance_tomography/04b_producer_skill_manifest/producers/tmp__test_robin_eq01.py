import importlib.util
from time import perf_counter
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location('robin', '/holdout_task/robin_continuation_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def reference(st, t):
    k = st['kappa_base'] + t * st['kappa_mod1'] + t*t * st['kappa_mod2']
    shape = k.shape
    h = np.array([st['Lx'], st['Ly'], st['Lz']]) / shape
    v = np.prod(h)
    xyz = np.indices(shape).reshape(3, -1)
    k = k.ravel()
    cell = np.arange(k.size)
    diag = v * (st['sigma_base'] + t * st['sigma_mod']).ravel()
    diffs, weights = [], []
    for d in range(3):
        start = cell[xyz[d] < shape[d]-1]
        end = start + np.prod(shape[d+1:], dtype=int)
        diffs.append(sparse.csr_matrix((np.tile([1., -1.], start.size), (np.repeat(np.arange(start.size), 2), np.column_stack((start, end)).ravel())), shape=(start.size, k.size)))
        weights.append(2 * v / h[d]**2 / (1/k[start] + 1/k[end]))
        sides = cell[(xyz[d] == 0) | (xyz[d] == shape[d]-1)]
        diag[sides] += (v / h[d]) / (1/st['gamma_robin'] + h[d]/(2*k[sides]))
    d = sparse.vstack(diffs, format='csr')
    return (d.T @ d.multiply(np.concatenate(weights)[:, None]) + sparse.diags(diag)).tocsr(), v

def make_study(case):
    rng = np.random.default_rng(801+case)
    shape = (28,)*3
    x,y,z = np.indices(shape) / 27
    if case == 0:
        base = np.exp(rng.uniform(0, np.log(3500), shape))
        k1 = 0.02 + base * rng.uniform(0.01, 0.9, shape)
        k2 = 0.02 + base * rng.uniform(0.01, 0.9, shape)
        lengths = (1.2, 0.93, 1.61)
    else:
        base = np.where((x<.18) | ((y>.3)&(y<.5)) | (z>.85), 3500., 1.)
        k1 = .01 + 120*np.exp(-90*(z-.55)**2)
        k2 = np.where((x>.75) | ((z>.3)&(z<.5)), 2800., .001)
        lengths = (.75, 1.15, 1.83)
    t = rng.permutation(np.linspace(.15, 1.85, 9))
    sensors = rng.integers(0, 28, (80,3))
    n = len(sensors)*len(t)
    a = rng.normal(size=(n, 15))
    a[:, -1] = a[:, 0] + 1e-8*a[:, 1]
    return dict(Lx=lengths[0], Ly=lengths[1], Lz=lengths[2], kappa_base=base,
        kappa_mod1=k1, kappa_mod2=k2, sigma_base=.01+.015*x, sigma_mod=.002+.012*y,
        gamma_robin=[2., .18][case], thetas=t,
        source_modes=np.stack((np.ones(shape), np.sin(6.3*x)*np.cos(4*y), np.exp(-20*((x-.3)**2+(y-.2)**2+(z-.7)**2)))),
        sensor_ijk=sensors, Z_nuisance=a, Y_obs=rng.normal(size=(n, 5)), nuisance_ridge=1e-6, source_ridge=2e-6)

def run(st):
    capture = []
    factors = []
    real_state, real_lu = mod._solve_state, mod.splu
    def state(*args, **kwargs):
        sol = real_state(*args, **kwargs)
        capture.append(sol.copy())
        return sol
    def lu(*args, **kwargs):
        factors.append(1)
        return real_lu(*args, **kwargs)
    mod._solve_state, mod.splu = state, lu
    start = perf_counter()
    try:
        out = mod.solve_robin_continuation_tomography(st)
    finally:
        mod._solve_state, mod.splu = real_state, real_lu
    elapsed = perf_counter() - start
    idx = tuple(st['sensor_ijk'].T)
    states = len(st['thetas'])
    truth = np.empty((states, len(idx[0]), 3))
    resid = []
    for j, k in enumerate(np.argsort(st['thetas'])):
        mat, v = reference(st, st['thetas'][k])
        b = v*st['source_modes'].reshape(3, -1).T
        u = capture[j].T
        resid.extend(np.linalg.norm(mat@u-b, axis=0) / np.linalg.norm(b,axis=0))
        exact = splu(mat.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0, options={'SymmetricMode':True}).solve(b)
        shaped = exact.T.reshape(st['source_modes'].shape)
        truth[k] = shaped[(slice(None),)+idx].T
        if k == states-1:
            fields = shaped
    s = truth.reshape(-1, 3)
    z,y = st['Z_nuisance'], st['Y_obs']
    p = z.shape[1]
    y_aug = np.vstack((y, np.zeros((p, y.shape[1]))))
    z_aug = np.vstack((z, np.sqrt(st['nuisance_ridge'])*np.eye(p)))
    s_aug = np.vstack((s, np.zeros((p,3))))
    prof = np.stack([linalg.lstsq(np.column_stack((z_aug, col)), y_aug, lapack_driver='gelsd')[0][-1] for col in s_aug.T])
    block = np.vstack((np.column_stack((z_aug,s_aug)), np.column_stack((np.zeros((3,p)), np.sqrt(st['source_ridge'])*np.eye(3)))))
    joint = linalg.lstsq(block, np.vstack((y_aug, np.zeros((3,y.shape[1])))), lapack_driver='gelsd')[0][-3:]
    expected = dict(terminal_fields=fields, continuation_sensor_predictions=s, profiled_amplitudes=prof, joint_source_weights=joint)
    assert out.keys() == expected.keys()
    err = {}
    for name, a in expected.items():
        actual = out[name]
        assert actual.shape == a.shape and actual.dtype.kind == 'f'
        err[name] = np.linalg.norm(actual-a)/np.linalg.norm(a)
    print(f'solve={elapsed:.3f}s fallback_factors={len(factors)-1} max_residual={max(resid):.3g} errors={err}', flush=True)
    assert max(resid) <= 1e-7
    assert err['profiled_amplitudes'] <= 1e-5 and err['joint_source_weights'] <= .03
    assert max(err['terminal_fields'], err['continuation_sensor_predictions']) <= 1e-7
    return elapsed

if __name__ == '__main__':
    import sys
    weak = '--weak' in sys.argv
    def fixture(i):
        if not weak:
            return make_study(i)
        st = make_study(1)
        st['gamma_robin'] = (.08, .04)[i]
        st['sigma_base'] *= .5
        st['sigma_mod'] *= .5
        if i:
            st['kappa_mod2'] = np.flip(st['kappa_mod2'], axis=0).copy()
        return st
    with threadpool_limits(limits=1, user_api='blas'):
        total = sum(run(fixture(i)) for i in range(2))
    print(f'Two-study solve time: {total:.3f}s')
    assert total <= 25
