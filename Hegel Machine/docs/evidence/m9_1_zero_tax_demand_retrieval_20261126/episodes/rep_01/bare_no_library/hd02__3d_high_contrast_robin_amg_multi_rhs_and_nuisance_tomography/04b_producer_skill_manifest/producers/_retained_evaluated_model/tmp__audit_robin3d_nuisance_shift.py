import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    cases = pickle.load(f)
for case in cases:
    p = case['problem']
    out = mod.solve_3d_robin_tomography(p)
    Z, Y = p['nuisance_design'], p['Y_obs']
    S = out['basis_fields'][(slice(None), *p['sensor_ijk'].T)].T
    lz, ls = p['nuisance_ridge'], p['source_ridge']
    shift = np.cos(np.arange(Z.shape[1] * Y.shape[1])).reshape(Z.shape[1], Y.shape[1])
    shifted = mod.solve_3d_robin_tomography(dict(p, Y_obs=Y + Z @ shift))
    # Under the specified nuisance ridge, a pure nuisance shift is not
    # canceled exactly: P Z C = lambda_z Z (Z.T Z + lambda_z I)^-1 C.
    G = Z.T @ Z + lz * np.eye(Z.shape[1])
    P_S = S - Z @ np.linalg.solve(G, Z.T @ S)
    h = S.T @ (lz * Z @ np.linalg.solve(G, shift))
    gram = S.T @ P_S
    expected = dict(single_source_profiled_amplitudes=h / np.diag(gram)[:, None],
                    joint_source_weights=np.linalg.solve(gram + ls * np.eye(S.shape[1]), h))
    for key, ref in expected.items():
        delta = shifted[key] - out[key]
        err = np.max(np.abs(delta - ref))
        print(f'{key}: largest nuisance-shift response {np.max(np.abs(ref)):.3e}; max response error {err:.3e}')
        np.testing.assert_allclose(delta, ref, rtol=1e-8, atol=1e-10)
