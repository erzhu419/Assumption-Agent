import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('tomography', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    p = pickle.load(f)[0]['problem']
out = mod.solve_3d_robin_tomography(p)
u = out['basis_fields']
kappa = p['kappa']
h = np.array([p['Lx'], p['Ly'], p['Lz']]) / kappa.shape
vol = np.prod(h)
loss = vol * np.sum(u * p['sigma'], axis=(1, 2, 3))
for a, step in enumerate(h):
    for end in (0, -1):
        k = np.take(kappa, end, axis=a)
        v = np.take(u, end, axis=a + 1)
        loss += np.sum(v * (vol / step) / (1 / p['gamma_robin'] + step / (2 * k)), axis=(1, 2))
supply = vol * np.sum(p['candidate_sources'], axis=(1, 2, 3))
balance = np.max(np.abs(loss - supply) / np.abs(supply))
assert balance < 1e-10, balance

# A nuisance-only perturbation changes weights by the finite-ridge leakage,
# not by the response from ignoring nuisance or projecting it out unpenalized.
i, j, k = p['sensor_ijk'].T
S = u[:, i, j, k].T
Z, Y = p['nuisance_design'], p['Y_obs']
lz, ls = p['nuisance_ridge'], p['source_ridge']
C = np.arange(1, Z.shape[1] * Y.shape[1] + 1, dtype=float).reshape(Z.shape[1], Y.shape[1]) / 3
G = Z.T @ Z + lz * np.eye(Z.shape[1])
PS = S - Z @ np.linalg.solve(G, Z.T @ S)
Pdelta = Z @ np.linalg.solve(G, lz * C)
cross = S.T @ Pdelta
expected = (cross / np.sum(S * PS, axis=0)[:, None], np.linalg.solve(S.T @ PS + ls * np.eye(S.shape[1]), cross))
shifted = mod._profile_sources(S, Z, Y + Z @ C, lz, ls)
errors = []
for key, actual, ref in zip(('single_source_profiled_amplitudes', 'joint_source_weights'), shifted, expected):
    delta = actual - out[key]
    np.testing.assert_allclose(delta, ref, rtol=1e-7, atol=1e-10)
    errors.append(np.max(np.abs(delta - ref)))
print('Public worst integrated conservation defect:', balance)
print('Public nuisance-shift max absolute error (single, joint):', errors)
