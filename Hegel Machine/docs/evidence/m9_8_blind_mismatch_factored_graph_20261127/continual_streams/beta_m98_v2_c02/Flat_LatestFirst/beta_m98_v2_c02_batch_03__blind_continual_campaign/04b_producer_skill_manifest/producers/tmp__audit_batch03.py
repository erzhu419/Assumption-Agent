import importlib.util
import itertools
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root + 'batch_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def axis_action(field, kappa, velocity, h, axis):
    f = np.moveaxis(field, axis, 0)
    k = np.asarray(kappa)[:, None]
    v = np.asarray(velocity)[:, None]
    left = (k + np.roll(k, 1, axis=0)) / (2*h*h) + np.maximum(v, 0)/h
    right = (k + np.roll(k, -1, axis=0)) / (2*h*h) + np.maximum(-v, 0)/h
    return np.moveaxis(left*(f - np.roll(f, 1, axis=0)) + right*(f - np.roll(f, -1, axis=0)), 0, axis)


def exhaustive_qp(A, y, ridge, cap, budget):
    n = A.shape[1]
    H = A.T @ A + ridge*np.eye(n)
    b = A.T @ y
    best = (np.inf, None)
    for status in itertools.product((-1, 0, 1), repeat=n):
        st = np.array(status)
        free = np.flatnonzero(st == 0)
        upper = np.flatnonzero(st == 1)
        balance = budget - cap*len(upper)
        if balance < -1e-12 or balance > cap*len(free) + 1e-12:
            continue
        w = np.zeros(n)
        w[upper] = cap
        if len(free):
            hf = H[np.ix_(free, free)]
            q, p = np.linalg.solve(hf, np.column_stack((b[free] - cap*H[np.ix_(free, upper)].sum(axis=1), np.ones(len(free))))).T
            w[free] = q + p*((balance - q.sum())/p.sum())
            if np.min(w) < -1e-12 or np.max(w) > cap + 1e-12:
                continue
        value = 0.5*np.linalg.norm(A@w-y)**2 + 0.5*ridge*(w@w)
        if value < best[0]:
            best = (value, w)
    return best

with open(root + 'public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
for item in batch:
    s = item['study']
    result = solver.solve_batch(s)
    w, u, c = (result[k] for k in ('weights', 'terminal_field', 'kkt_certificate'))
    nx, ny = s['nx'], s['ny']
    dx, dy = s['Lx']/(nx-1), s['Ly']/(ny-1)
    forcing = np.tensordot(w, s['source_modes'], axes=1)
    applied = (axis_action(u, s['kappa_x'], s['vel_x'], dx, 0)
               + axis_action(u, s['kappa_y'], s['vel_y'], dy, 1)
               + (s['V_reaction_2d'] + s['mu_shifts'][-1])*u)
    err = np.abs(applied - forcing)
    wall = np.zeros((nx, ny), dtype=bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    S, _ = solver._forward(s)
    sweeps = s['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    fit = np.linalg.lstsq(np.c_[np.ones(len(ref)), ref], raw, rcond=None)[0]
    variants = []
    for data in [s['y_obs_raw']] + [p['y_pulse_raw'] for p in item.get('campaign_diagnostics') or []]:
        y = np.linalg.solve(fit[1:].T, (data.reshape(-1, 4)-fit[0]).T).T.ravel()
        Z = s['Z_nuisance']
        # An SVD implements the same linear nuisance map without reusing the solver's normal equations.
        Q, d, _ = np.linalg.svd(Z, full_matrices=False)
        M = np.column_stack((S, y))
        M -= Q @ ((d*d/(d*d+s['nuisance_ridge']))[:, None]*(Q.T@M))
        A, target = M[:, :-1], M[:, -1]
        ridge, cap, budget = (s[k] for k in ('source_ridge', 'u_box', 'total_budget'))
        out = solver.solve_batch(dict(s, y_obs_raw=data))
        ww = out['weights']
        cert = out['kkt_certificate']
        opt, exact = exhaustive_qp(A, target, ridge, cap, budget)
        g = A.T@(A@ww-target) + ridge*ww
        nu = cert['nu_sum']
        lower, upper = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper'])
        stat = np.max(np.abs(g - nu - lower + upper))
        gap = abs(opt - cert['dual_lower_bound'])
        weight_err = np.max(np.abs(exact - ww))
        variants.append((weight_err, stat, gap))
        assert abs(ww.sum() - budget) < 1e-10 and ww.min() >= -1e-10 and ww.max() <= cap + 1e-10
        assert abs(opt-cert['primal_objective']) < 1e-10 and max(weight_err, stat, gap) < 1e-9
    print('Independent finite differences: max wall and interior terminal residuals', err[wall].max(), err[~wall].max())
    print('Observed and two measured tracer responses: worst absolute weight, stationarity, and dual-bound errors', np.max(variants, axis=0))
    assert err.max() < 1e-10
