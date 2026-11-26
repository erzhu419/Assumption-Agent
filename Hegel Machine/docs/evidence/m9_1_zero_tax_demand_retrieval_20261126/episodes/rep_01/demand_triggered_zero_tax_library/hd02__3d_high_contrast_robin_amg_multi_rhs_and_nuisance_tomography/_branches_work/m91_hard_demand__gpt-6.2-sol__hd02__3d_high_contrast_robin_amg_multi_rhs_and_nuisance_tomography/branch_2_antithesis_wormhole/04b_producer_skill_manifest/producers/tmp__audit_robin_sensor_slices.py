import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin3d', '/holdout_task/robin3d_amg_tomography.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin3d_cases.pkl', 'rb') as f:
    case = pickle.load(f)[0]
p = case['problem']
sensors = p['sensor_ijk']
Z, Y = p['nuisance_design'], p['Y_obs']
S = case['gt']['basis_fields'][:, sensors[:, 0], sensors[:, 1], sensors[:, 2]].T
P, M = Z.shape[1], S.shape[1]
Rz = np.sqrt(p['nuisance_ridge']) * np.eye(P)
Rs = np.sqrt(p['source_ridge']) * np.eye(M)
slabs = 3 * sensors[:, 0] // p['kappa'].shape[0]
worst = 0.
for slab in range(3):
    held = slabs == slab
    keep = np.flatnonzero(~held)[::-1]
    z, y, s = Z[keep], Y[keep], S[keep]
    sub = dict(p, sensor_ijk=sensors[keep], nuisance_design=z, Y_obs=y)
    out = mod.solve_3d_robin_tomography(sub)
    single_ref = np.stack([np.linalg.lstsq(np.block([[z, s[:, j:j+1]], [Rz, np.zeros((P, 1))]]), np.pad(y, ((0, P), (0, 0))), rcond=None)[0][-1] for j in range(M)])
    joint_ref = np.linalg.lstsq(np.block([[z, s], [Rz, np.zeros((P, M))], [np.zeros((M, P)), Rs]]), np.pad(y, ((0, P+M), (0, 0))), rcond=None)[0][-M:]
    errors = []
    for key, ref in [('single_source_profiled_amplitudes', single_ref), ('joint_source_weights', joint_ref)]:
        np.testing.assert_allclose(out[key], ref, rtol=1e-8, atol=1e-10)
        errors.append(np.max(np.abs(out[key] - ref)))
    worst = max(worst, *errors)
    print(f'Excluded x-slab {slab}: {held.sum()} held-out sensors; max single/joint coefficient errors {errors}')
print(f'Worst sensor-slice coefficient error: {worst:.3e}')
