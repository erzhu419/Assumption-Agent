import pickle
import runpy
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

module = runpy.run_path('/holdout_task/robin_spd_profiler.py')
solve = module['solve_robin_spd_and_profile_sources']
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as file:
    cases = pickle.load(file)


def reference_system(p):
    k = p['kappa']
    nx, ny = k.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    n = k.size
    diag = hx * hy * p['sigma'].ravel().copy()
    rows, cols, values = [], [], []
    for i, j in np.ndindex(k.shape):
        t = i * ny + j
        for di, dj, h, length in [(1, 0, hx, hy), (0, 1, hy, hx)]:
            u, v = i + di, j + dj
            if u < nx and v < ny:
                rt = u * ny + v
                conductance = length / (h/(2*k[i,j]) + h/(2*k[u,v]))
                diag[t] += conductance
                diag[rt] += conductance
                rows.extend([t, rt])
                cols.extend([rt, t])
                values.extend([-conductance, -conductance])
        gamma = p['gamma_robin']
        if gamma:
            if i in (0, nx-1):
                diag[t] += hy/(1/gamma + hx/(2*k[i,j]))
            if j in (0, ny-1):
                diag[t] += hx/(1/gamma + hy/(2*k[i,j]))
    rows.extend(range(n))
    cols.extend(range(n))
    values.extend(diag)
    return coo_matrix((values, (rows, cols)), shape=(n,n)).tocsc(), hx*hy*p['candidate_sources'].reshape(-1,n).T


def check(label, p, expected=None):
    start = time.perf_counter()
    out = solve(p)
    elapsed = time.perf_counter() - start
    a, b = reference_system(p)
    m = b.shape[1]
    u = out['basis_fields'].reshape(m,-1).T
    norms = np.linalg.norm(b,axis=0)
    residual = np.linalg.norm(a @ u-b,axis=0)
    assert np.all(residual[norms == 0] == 0)
    worst = np.max(residual[norms > 0]/norms[norms > 0])
    assert worst <= 1e-8, (label, worst)
    np.testing.assert_allclose(u, splu(a).solve(b), rtol=2e-8, atol=1e-10)
    ij = p['sensor_ij']
    s = out['basis_fields'][:,ij[:,0],ij[:,1]].T
    z, y = p['nuisance_design'], p['Y_obs']
    nz, q = z.shape[1], y.shape[1]
    def fit(block, ridge):
        size = block.shape[1]
        aug = np.vstack((np.column_stack((z, block)), np.diag(np.sqrt([p['nuisance_ridge']]*nz + [ridge]*size))))
        return np.linalg.lstsq(aug, np.vstack((y, np.zeros((nz+size,q)))), rcond=None)[0][nz:]
    refs = {'single_source_profiled_amplitudes': np.vstack([fit(s[:,i:i+1], 0) for i in range(m)]), 'joint_source_weights': fit(s, p['source_ridge'])}
    for k, v in refs.items():
        np.testing.assert_allclose(out[k], v, rtol=1e-7, atol=1e-8)
    if expected is not None:
        assert out.keys() == expected.keys()
        for k,v in expected.items():
            assert out[k].shape == v.shape
            np.testing.assert_allclose(out[k], v, rtol=1e-7, atol=1e-8)
    print(label, 'worst_residual', worst, 'inverse_max_abs_errors', [float(np.max(np.abs(out[k]-v))) for k,v in refs.items()], 'solve_s', round(elapsed, 3), flush=True)


for n, case in enumerate(cases):
    check('public-'+str(n), case['problem'], case['gt'])

base = cases[0]['problem']
i, j = np.indices(base['kappa'].shape)
rng = np.random.default_rng(48)
loads = rng.normal(size=base['candidate_sources'].shape)
loads[0] = 0
loads[1] = 0
loads[1,0,0] = 1
loads[2] = 1
check('striped-anisotropic-domain', dict(base, Lx=2.4, Ly=0.7, gamma_robin=4.0, kappa=np.where((j//3)%2, 80., .02), candidate_sources=loads))
check('high-k-weak-anchor', dict(base, kappa=np.full(i.shape, 80.), gamma_robin=.05, candidate_sources=loads, nuisance_ridge=0., nuisance_design=base['nuisance_design'][:,1:]))
check('high-contrast-channel-intersections', dict(base, kappa=np.where((i%12<2)|(j%9<2), .02, 80.), gamma_robin=2.9, source_ridge=0.))
