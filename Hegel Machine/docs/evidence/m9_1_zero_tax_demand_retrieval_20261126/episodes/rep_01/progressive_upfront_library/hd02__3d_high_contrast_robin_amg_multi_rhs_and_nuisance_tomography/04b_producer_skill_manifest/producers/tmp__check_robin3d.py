import importlib.util
import pickle
import time

import numpy as np

spec = importlib.util.spec_from_file_location('robin', '/holdout_task/robin3d_amg_tomography.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    case = pickle.load(f)[0]


def independent_residual(p, u):
    K = p['kappa']
    g = p['gamma_robin']
    h = np.array([p['Lx'],p['Ly'],p['Lz']]) / K.shape
    out = u * p['sigma']
    for a, step in enumerate(h):
        k = np.moveaxis(K, a, 0)
        v = np.moveaxis(u, a + 1, 1)
        r = np.moveaxis(out, a + 1, 1)
        flux = (v[:, 1:] - v[:, :-1]) / (0.5 * step**2 * (1/k[1:] + 1/k[:-1]))
        r[:, 1:] += flux
        r[:, :-1] -= flux
        for b in (0, -1):
            r[:, b] += v[:, b] / (step * (1/g + step/(2*k[b])))
    src = p['candidate_sources']
    return np.linalg.norm((out - src).reshape(len(src), -1), axis=1) / np.linalg.norm(src.reshape(len(src), -1), axis=1)


def regression_check(p, result):
    sensor = tuple(p['sensor_ijk'].T)
    S = result['basis_fields'][(slice(None),) + sensor].T
    Z, Y = p['nuisance_design'], p['Y_obs']
    A, W = result['single_source_profiled_amplitudes'], result['joint_source_weights']
    def fit(X, ridge):
        B = np.concatenate((Z, X), axis=1)
        D = np.diag(np.sqrt([p['nuisance_ridge']]*Z.shape[1] + [ridge]*X.shape[1]))
        return np.linalg.lstsq(np.vstack((B, D)), np.pad(Y, ((0,len(D)), (0,0))), rcond=None)[0][-X.shape[1]:]
    true_A = np.vstack([fit(S[:, m:m+1], 0.) for m in range(S.shape[1])])
    true_W = fit(S, p['source_ridge'])
    np.testing.assert_allclose(A, true_A, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(W, true_W, atol=1e-10, rtol=1e-9)
    return np.max(abs(A-true_A)), np.max(abs(W-true_W))


def test(label, p):
    start = time.perf_counter()
    result = solver.solve_3d_robin_tomography(p)
    elapsed = time.perf_counter()-start
    residuals = independent_residual(p, result['basis_fields'])
    assert max(residuals) <= 1e-7, residuals
    print(label, 'seconds', round(elapsed,3), 'worst residual', max(residuals), 'regression abs errors', regression_check(p,result), flush=True)
    return result,elapsed

p0 = case['problem']
result,_ = test('public', p0)
for key, truth in case['gt'].items():
    np.testing.assert_allclose(result[key], truth, rtol=1e-7, atol=1e-9)
    print('public reference', key, 'max abs error', np.max(np.abs(result[key]-truth)), flush=True)

idx = (np.arange(28)*20//28)
grid = np.ix_(idx, idx, idx)
p1 = dict(p0, kappa=p0['kappa'][grid], sigma=p0['sigma'][grid],
          candidate_sources=p0['candidate_sources'][(slice(None),)+grid], sensor_ijk=p0['sensor_ijk']*28//20)
rng = np.random.default_rng(321)
p2 = dict(p1, kappa=np.where(rng.random((28,28,28)) < .60, 80., .02),
          Lx=1.5, Ly=0.9, Lz=0.8, gamma_robin=.11, sigma=p1['sigma'] * .15)
_, t1 = test('28 resampled public', p1)
_, t2 = test('28 random high contrast, weak Robin', p2)
print('28 pair total', t1+t2, flush=True)
