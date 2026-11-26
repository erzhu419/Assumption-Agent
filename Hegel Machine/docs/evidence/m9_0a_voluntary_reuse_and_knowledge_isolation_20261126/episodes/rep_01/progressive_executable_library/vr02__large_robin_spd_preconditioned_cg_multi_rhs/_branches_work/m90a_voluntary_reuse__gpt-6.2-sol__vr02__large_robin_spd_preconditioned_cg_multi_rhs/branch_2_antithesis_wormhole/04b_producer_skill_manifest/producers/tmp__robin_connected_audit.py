import importlib.util
import pickle
import time
import numpy as np

spec = importlib.util.spec_from_file_location('robin_spd_profiler', '/holdout_task/robin_spd_profiler.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_robin_spd_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

original = mod.solve_residual_checked_cg
for ix, case in enumerate(cases):
    base = case['problem']
    i, j = np.indices(base['kappa'].shape)
    k = np.where((i % 12 < 2) & (j % 12 < 2), 0.02, 80.)
    for tol in (5e-10, 1e-9, 3e-9):
        sparse_res = []
        def probe(A, b, **kwargs):
            kwargs['tol'] = tol
            x, residual = original(A, b, **kwargs)
            sparse_res.append(residual)
            return x, residual
        mod.solve_residual_checked_cg = probe
        p = dict(base, kappa=k, gamma_robin=0.03)
        start = time.perf_counter()
        try:
            out = mod.solve_robin_spd_and_profile_sources(p)
        except RuntimeError as exc:
            print(f'case {ix}, tolerance {tol:.1e}: rejected after {len(sparse_res)}/10 modes: {exc}')
            continue
        u = out['basis_fields']
        hx, hy = p['Lx'] / k.shape[0], p['Ly'] / k.shape[1]
        b = hx * hy * p['candidate_sources']
        r = hx * hy * p['sigma'] * u - b
        fx = hy * (u[:, :-1, :] - u[:, 1:, :]) / (0.5 * hx * (1/k[:-1, :] + 1/k[1:, :]))
        fy = hx * (u[:, :, :-1] - u[:, :, 1:]) / (0.5 * hy * (1/k[:, :-1] + 1/k[:, 1:]))
        r[:, :-1, :] += fx; r[:, 1:, :] -= fx
        r[:, :, :-1] += fy; r[:, :, 1:] -= fy
        for a in (0, -1):
            r[:, a, :] += hy * u[:, a, :] / (1/p['gamma_robin'] + hx/(2*k[a, :]))
            r[:, :, a] += hx * u[:, :, a] / (1/p['gamma_robin'] + hy/(2*k[:, a]))
        flux = np.max(np.linalg.norm(r, axis=(1, 2)) / np.linalg.norm(b, axis=(1, 2)))
        print(f'case {ix}, tolerance {tol:.1e}: sparse={max(sparse_res):.3g}, flux={flux:.3g}, seconds={time.perf_counter()-start:.3f}')
