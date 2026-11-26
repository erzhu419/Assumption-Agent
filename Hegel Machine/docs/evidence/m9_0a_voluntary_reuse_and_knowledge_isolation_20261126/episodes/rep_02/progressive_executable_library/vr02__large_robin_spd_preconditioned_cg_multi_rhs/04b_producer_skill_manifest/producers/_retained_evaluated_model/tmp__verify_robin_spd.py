import importlib.util
import pickle
import time
import numpy as np
from scipy.sparse.linalg import spsolve

path = '/holdout_task/robin_spd_profiler.py'
spec = importlib.util.spec_from_file_location('robin_spd_profiler', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
keys = ('basis_fields', 'single_source_profiled_amplitudes', 'joint_source_weights')

def check(name, p, expected=None):
    start = time.perf_counter()
    result = module.solve_robin_spd_and_profile_sources(p)
    rhs = p['candidate_sources']
    k = p['kappa']
    nx, ny = k.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    gamma = p['gamma_robin']
    u = result['basis_fields']
    i, j = p['sensor_ij'].T
    s = u[:, i, j].T
    z, y = p['nuisance_design'], p['Y_obs']
    m, n, q = s.shape[1], z.shape[1], y.shape[1]
    assert set(result) == set(keys)
    for key, shape in zip(keys, (rhs.shape, (m,q), (m,q))):
        assert result[key].shape == shape and np.isfinite(result[key]).all(), key

    # Independent flux-form application; no call to the assembly routine.
    r = (hx*hy) * (p['sigma']*u-rhs)
    fx = (u[:, :-1, :] - u[:, 1:, :]) * ((hy/hx)/(0.5/k[:-1, :] + 0.5/k[1:, :]))
    fy = (u[:, :, :-1] - u[:, :, 1:]) * ((hx/hy)/(0.5/k[:, :-1] + 0.5/k[:, 1:]))
    r[:, :-1, :] += fx
    r[:, 1:, :] -= fx
    r[:, :, :-1] += fy
    r[:, :, 1:] -= fy
    for a in (0, -1):
        r[:, a, :] += u[:, a, :] * (hy*gamma/(1 + 0.5*hx*gamma/k[a, :]))
        r[:, :, a] += u[:, :, a] * (hx*gamma/(1 + 0.5*hy*gamma/k[:, a]))
    norm = hx*hy*np.linalg.norm(rhs, axis=(1,2))
    relative = np.divide(np.linalg.norm(r, axis=(1,2)), norm, out=np.zeros_like(norm), where=norm > 0)
    assert np.all(u[norm == 0] == 0)
    assert relative.max() <= 1e-8, relative

    a = module._robin_operator(k, p['sigma'], hx, hy, gamma)
    ref = spsolve(a.tocsc(), (hx*hy)*rhs.reshape(m,-1).T).T.reshape(rhs.shape)
    assert np.allclose(u, ref, rtol=1e-7, atol=1e-10)
    zr = np.sqrt(p['nuisance_ridge'])*np.eye(n)
    single = np.zeros((m,q))
    for l in range(m):
        design = np.block([[z, s[:,l:l+1]], [zr, np.zeros((n,1))]])
        single[l] = np.linalg.lstsq(design, np.vstack((y,np.zeros((n,q)))), rcond=None)[0][-1]
    design = np.block([[z,s], [zr,np.zeros((n,m))], [np.zeros((m,n)),np.sqrt(p['source_ridge'])*np.eye(m)]])
    joint = np.linalg.lstsq(design, np.vstack((y,np.zeros((n+m,q)))), rcond=None)[0][-m:]
    for key, target in zip(keys, (ref, single, joint)):
        assert np.allclose(result[key],target,rtol=2e-7,atol=1e-9), key
        if expected is not None:
            assert np.allclose(result[key],expected[key],rtol=2e-7,atol=1e-9), key
    errors = [np.max(np.abs(result[key]-target)) for key,target in zip(keys, (ref,single,joint))]
    print(f'{name}: worst residual={relative.max():.3e}; max errors fields/single/joint={errors}; elapsed={time.perf_counter()-start:.3f}s', flush=True)

with open('/holdout_task/public_robin_spd_cases.pkl','rb') as f:
    cases = pickle.load(f)
for c, case in enumerate(cases):
    check(f'public-{c}', case['problem'], case['gt'])
p = cases[0]['problem']
x, y = np.indices(p['kappa'].shape)
for name, k in [('checkerboard', np.where((x+y)%2, 0.02, 80.0)), ('conducting-stripes', np.where((x//4)%2, 0.02, 80.0)), ('high-conductivity-core', np.where((x>8)&(x<62)&(y>8)&(y<62), 80.0, 0.02))]:
    q = p['candidate_sources'].copy()
    q[0] = 0
    q[1] -= q[2]
    check(name, dict(p, Lx=2.0, Ly=0.8, kappa=k, candidate_sources=q))
