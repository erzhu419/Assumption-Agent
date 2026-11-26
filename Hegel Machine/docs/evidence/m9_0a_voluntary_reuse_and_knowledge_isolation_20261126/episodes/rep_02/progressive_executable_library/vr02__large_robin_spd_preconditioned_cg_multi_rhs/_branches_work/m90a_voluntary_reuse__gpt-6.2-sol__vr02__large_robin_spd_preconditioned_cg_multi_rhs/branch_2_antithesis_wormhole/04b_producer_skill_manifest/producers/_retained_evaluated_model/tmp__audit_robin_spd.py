import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for idx, case in enumerate(cases):
    p, ref = case['problem'], case['gt']
    k, z = p['kappa'], p['nuisance_design']
    q = p['candidate_sources']
    hx, hy = p['Lx']/k.shape[0], p['Ly']/k.shape[1]
    g = p['gamma_robin']
    # Axis exchange and a signed source contrast give exact transformations of
    # the measured-case fields, without using the solver's assembled matrix.
    expected = ref['basis_fields'].copy()
    expected[0] -= expected[1]
    q_contrast = q.copy()
    q_contrast[0] -= q_contrast[1]
    i, j = p['sensor_ij'].T
    col = expected[0, i, j]
    # Force source/nuisance overlap at the actual positive ridge, leaving the
    # observations unchanged. Compare with unprofiled augmented least squares.
    z_test = np.column_stack((z, col / np.linalg.norm(col)))
    transformed = dict(p, Lx=p['Ly'], Ly=p['Lx'], kappa=k.T,
                       sigma=p['sigma'].T, candidate_sources=q_contrast.swapaxes(1, 2),
                       sensor_ij=p['sensor_ij'][:, ::-1], nuisance_design=z_test)
    result = mod.solve_robin_spd_and_profile_sources(transformed)
    fields = result['basis_fields'].swapaxes(1, 2)
    np.testing.assert_allclose(fields, expected, rtol=1e-7, atol=1e-10)
    # All interior face pairs cancel in the global finite-volume budget.
    lhs = (hx*hy)*(p['sigma']*fields).sum(axis=(1, 2))
    for edge in (0, -1):
        lhs += (fields[:, edge, :] * (hy*g/(1 + 0.5*hx*g/k[edge, :]))).sum(axis=1)
        lhs += (fields[:, :, edge] * (hx*g/(1 + 0.5*hy*g/k[:, edge]))).sum(axis=1)
    balance = np.max(np.abs(lhs - (hx*hy)*q_contrast.sum(axis=(1, 2))) / ((hx*hy)*np.abs(q_contrast).sum(axis=(1, 2))))
    assert balance <= 1e-8, balance
    s, y = fields[:, i, j].T, p['Y_obs']
    nz, m = z_test.shape[1], s.shape[1]
    z_pen = np.sqrt(p['nuisance_ridge'])*np.eye(nz)
    y_aug = np.pad(y, ((0, nz), (0, 0)))
    single = np.stack([np.linalg.lstsq(np.block([[z_test, s[:, l:l+1]], [z_pen, np.zeros((nz, 1))]]), y_aug, rcond=None)[0][-1] for l in range(m)])
    joint_design = np.block([[z_test, s], [z_pen, np.zeros((nz, m))], [np.zeros((m, nz)), np.sqrt(p['source_ridge'])*np.eye(m)]])
    joint = np.linalg.lstsq(joint_design, np.pad(y_aug, ((0, m), (0, 0))), rcond=None)[0][-m:]
    errors = []
    for key, expected_fit in [('single_source_profiled_amplitudes', single), ('joint_source_weights', joint)]:
        np.testing.assert_allclose(result[key], expected_fit, rtol=2e-7, atol=1e-9)
        errors.append(np.max(np.abs(result[key] - expected_fit)))
    print(f'public-{idx} transformed: max field error={np.max(np.abs(fields-expected)):.3e}; relative global balance={balance:.3e}; max single/joint errors={errors}')
