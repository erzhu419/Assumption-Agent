import importlib.util
import pickle
import numpy as np

path = '/holdout_task/robin_spd_profiler.py'
spec = importlib.util.spec_from_file_location('robin_spd_profiler', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

for index, case in enumerate(cases):
    p = case['problem']
    out = mod.solve_robin_spd_and_profile_sources(p)
    k, q = p['kappa'], p['candidate_sources']
    hx, hy = p['Lx']/k.shape[0], p['Ly']/k.shape[1]
    g = p['gamma_robin']
    u = out['basis_fields']
    # Interior faces cancel in the domain-wide balance. Corners lose flux through both walls.
    walls = sum((u[:, i, :] * hy / (1/g + hx/(2*k[i, :]))).sum(axis=1) for i in (0, -1))
    walls += sum((u[:, :, j] * hx / (1/g + hy/(2*k[:, j]))).sum(axis=1) for j in (0, -1))
    reaction = hx*hy * (u * p['sigma']).sum(axis=(1, 2))
    production = hx*hy * q.sum(axis=(1, 2))
    balance = np.max(np.abs((walls + reaction - production) / production))
    assert balance < 1e-8, balance

    # For the positive nuisance ridge, P = I - Z(Z'Z + lambda I)^-1 Z'.
    # Check the resulting stationarity equations directly in sensor space.
    i, j = p['sensor_ij'].T
    s, z, y = u[:, i, j].T, p['nuisance_design'], p['Y_obs']
    inv = z.T @ z + p['nuisance_ridge'] * np.eye(z.shape[1])
    ps = s - z @ np.linalg.solve(inv, z.T @ s)
    py = y - z @ np.linalg.solve(inv, z.T @ y)
    cross = s.T @ py
    single = np.sum(s * ps, axis=0)[:, None] * out['single_source_profiled_amplitudes']
    joint = (s.T @ ps + p['source_ridge'] * np.eye(s.shape[1])) @ out['joint_source_weights']
    single_err, joint_err = (np.linalg.norm(fit-cross) / np.linalg.norm(cross) for fit in (single, joint))
    assert max(single_err, joint_err) < 1e-8, (single_err, joint_err)
    print(f'public-{index}: global balance relative error={balance:.3e}; single/joint stationarity relative errors={single_err:.3e}/{joint_err:.3e}')
