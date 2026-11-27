"""Finite-volume chamber inversion and box-constrained source estimation."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(k, v, h, robin, wall):
    face = 0.5 * (k[1:] + k[:-1]) / h**2
    left = -np.r_[face[:-1] + np.maximum(v[1:-1], 0) / h, k[-1] * wall[3] / h**2]
    right = -np.r_[k[0] * wall[2] / h**2, face[1:] + np.maximum(-v[1:-1], 0) / h]
    center = np.r_[-right[0] + robin * wall[0] / h,
                   face[:-1] + face[1:] + np.abs(v[1:-1]) / h,
                   -left[-1] + robin * wall[1] / h]
    return diags((left, center, right), (-1, 0, 1), format='csc')


def _response(study, wall, modes):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), study['robin_x'], wall[:4])
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), study['robin_y'], wall[4:])
    operator = (kron(eye(ny), ax) + kron(ay, eye(nx)) +
                diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
    identity = eye(nx * ny, format='csc')
    rhs = modes.transpose(0, 2, 1).reshape(len(modes), -1).T
    ij = np.asarray(study['sensor_ij'])
    shifts = study['mu_shifts']
    count = len(study['y_obs_raw']) // len(shifts)
    blocks = []
    for i, mu in enumerate(shifts):
        field = splu(operator + mu * identity).solve(rhs)
        sensors = ij if len(ij) == count else ij[i * count:(i + 1) * count]
        blocks.append(field[sensors[:, 0] + nx * sensors[:, 1]])
    return np.vstack(blocks), field.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _powers(degree):
    return np.array([np.bincount(c, minlength=4) for d in range(degree + 1)
                     for c in combinations_with_replacement(range(4), d)])


def _features(x, powers):
    return np.prod(x[:, None, :] ** powers[None, :, :], axis=-1)


def _rational_fit(x, target):
    """Profile linear coefficients for a quadratic coupling with saturating gain."""
    affine = np.column_stack((np.ones(len(x)), x))
    powers = _powers(2)[5:]
    quadratic = _features(x, powers)

    def design(q):
        return np.column_stack((affine, quadratic / (1. + (x @ q)**2)[:, None]))

    def profile(q):
        a = design(q)
        c = np.linalg.lstsq(a, target, rcond=1e-10)[0]
        return a @ c - target, c

    def cost(q):
        r, _ = profile(q)
        return r @ r

    eye4 = np.eye(4)
    starts = list(eye4 * .75)
    starts += [.55 * (eye4[i] + sign * eye4[j]) for i in range(4)
               for j in range(i + 1, 4) for sign in (-1, 1)]
    starts.sort(key=cost)
    best = min((least_squares(lambda q: profile(q)[0], start, bounds=(-6., 6.),
                              max_nfev=45, gtol=1e-12, ftol=1e-9, xtol=1e-9)
                for start in starts[:3]), key=lambda opt: opt.cost)
    return best.x, profile(best.x)[1]


def _rational_predict(x, q, c):
    return c[0] + x @ c[1:5] + (_features(x, _powers(2)[5:]) @ c[5:]) / (1. + (x @ q)**2)


def _calibration(sweeps):
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    folds = np.arange(len(x)) % 4
    models = [None] + [_powers(d) for d in range(1, 4)]
    errors = []
    for p in models:
        if p is None:
            err = x - y
        else:
            a = _features(x, p)
            err = np.empty_like(y)
            for k in range(4):
                train = folds != k
                err[~train] = a[~train] @ np.linalg.lstsq(a[train], (y - x)[train], rcond=None)[0] + x[~train] - y[~train]
        errors.append(np.max([np.sqrt(np.mean(err[folds == k]**2)) for k in range(4)]))
    selected = next(i for i, e in enumerate(errors) if e <= 1.12 * min(errors) + 1e-12)
    p = models[selected]
    # Near the sweep noise floor, a larger nonlinear family cannot improve inversion.
    rational = [None] * 4
    if errors[selected] > 5e-5:
        reference = np.asarray(sweeps.get('single_bank_sweeps', {}).get('ref_signal', []))
        raw_bank = np.asarray(sweeps.get('single_bank_sweeps', {}).get('raw_readout', []))
        noise = np.median(np.abs(raw_bank - reference)) if reference.size else 0.
        if errors[selected] > max(5e-5, min(1e-4, 4 * noise)):
            err = np.empty_like(y)
            poly_err = np.empty_like(y)
            for k in range(4):
                train = folds != k
                if p is None:
                    poly_err[~train] = x[~train] - y[~train]
                else:
                    a = _features(x, p)
                    poly_err[~train] = a[~train] @ np.linalg.lstsq(a[train], (y-x)[train], rcond=None)[0] + x[~train] - y[~train]
                for j in range(4):
                    q, c = _rational_fit(x[train], (y-x)[train, j])
                    err[~train, j] = _rational_predict(x[~train], q, c) + x[~train, j] - y[~train, j]
            for j in range(4):
                score = lambda e: max(np.sqrt(np.mean(e[folds == k, j]**2)) for k in range(4))
                if score(err) < 0.7 * score(poly_err):
                    rational[j] = _rational_fit(x, (y-x)[:, j])
    if any(item is not None for item in rational):
        powers = _powers(1) if p is None else p
        coeff = np.linalg.lstsq(_features(x, powers), y - x, rcond=None)[0]
        affine = coeff[:5].copy()
        for j, item in enumerate(rational):
            if item is not None:
                affine[:, j] = item[1][:5]

        def unmix(raw):
            values = np.asarray(raw, dtype=float).reshape(-1, 4)
            initial = np.linalg.solve((np.eye(4) + affine[1:]).T, (values-affine[0]).T).T

            def predict(z):
                z = z[None, :]
                delta = (_features(z, powers) @ coeff)[0]
                for j, item in enumerate(rational):
                    if item is not None:
                        delta[j] = _rational_predict(z, *item)[0]
                return z[0] + delta

            estimate = [least_squares(lambda z: predict(z) - goal, start, gtol=1e-12).x
                        for goal, start in zip(values, initial)]
            return np.asarray(estimate).reshape(np.shape(raw))

        return unmix
    if p is None:
        return lambda raw: np.array(raw, dtype=float, copy=True)
    coeff = np.linalg.lstsq(_features(x, p), y - x, rcond=None)[0]
    matrix = np.eye(4) + coeff[1:5]
    derivative_powers = [np.maximum(0, p - e) for e in np.eye(4, dtype=int)]

    def unmix(raw):
        values = np.asarray(raw, dtype=float).reshape(-1, 4)
        estimate = np.linalg.solve(matrix.T, (values - coeff[0]).T).T
        if len(p) > 5:
            for i, goal in enumerate(values):
                def fun(q):
                    return q + _features(q[None, :], p)[0] @ coeff - goal

                def jac(q):
                    delta = np.array([(_features(q[None, :], dp)[0] * p[:, j]) @ coeff
                                      for j, dp in enumerate(derivative_powers)])
                    return np.eye(4) + delta.T

                estimate[i] = least_squares(fun, estimate[i], jac=jac, gtol=1e-12).x
        return estimate.reshape(np.shape(raw))

    return unmix


def _walls(study, calibrate, source):
    nominal = np.ones(8)
    pulses = study.get('reference_tracer_pulses')
    if not pulses:
        return nominal
    modes = np.array([np.tensordot(p['w_pulse'], source, axes=1) for p in pulses])
    observed = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
    scale = max(np.sqrt(np.mean(observed**2)), 0.01)
    if np.sqrt(np.mean((_response(study, nominal, modes)[0] - observed)**2)) < 2e-4 * scale:
        return nominal
    groups = ([0, 1], [4, 5]), ([0], [1], [4], [5]), tuple([i] for i in range(8))
    best = (np.inf, nominal)
    for group in groups:
        def expand(x):
            result = nominal.copy()
            for z, inds in zip(x, group):
                result[inds] = z
            return result

        def fit(columns):
            def residual(x):
                return ((_response(study, expand(x), modes)[0] - observed)[:, columns] / scale).ravel()
            return expand(least_squares(residual, np.ones(len(group)), bounds=(0., np.inf),
                                       max_nfev=60, gtol=1e-10).x)

        wall = fit(slice(None))
        if len(pulses) > 1:
            score = max(np.sqrt(np.mean((_response(study, fit(np.arange(len(pulses)) != i), modes)[0][:, i]
                                         - observed[:, i])**2)) for i in range(len(pulses)))
        else:
            score = np.sqrt(np.mean((_response(study, wall, modes)[0] - observed)**2))
        if score < best[0]:
            best = (score, wall)
        if score < 2e-4 * scale:
            return wall
    baseline = np.max(np.sqrt(np.mean((_response(study, nominal, modes)[0] - observed)**2, axis=0)))
    return best[1] if best[0] < baseline else nominal


def _project(q, cap, total):
    if total <= 0:
        return np.zeros_like(q)
    if total >= len(q) * cap:
        return np.full_like(q, cap)
    lo, hi = np.min(q) - cap, np.max(q)
    for _ in range(53):
        mid = (lo + hi) * .5
        if np.clip(q - mid, 0., cap).sum() > total:
            lo = mid
        else:
            hi = mid
    return np.clip(q - (lo + hi) * .5, 0., cap)


def _dual(g, w, cap):
    free = (w > 1e-10) & (w < cap - 1e-10)
    if free.any():
        return float(np.mean(g[free]))
    upper = w > cap / 2
    if upper.all():
        return float(g.max())
    if not upper.any():
        return float(g.min())
    return float(.5 * (g[upper].max() + g[~upper].min()))


def _weights(s, y, ridge, cap, total):
    h = s.T @ s + ridge * np.eye(s.shape[1])
    b = s.T @ y
    w = np.full(len(b), total / len(b))
    eig = np.linalg.eigvalsh(h)
    lip = eig[-1]
    if lip == 0 or total <= 0 or total >= len(b) * cap:
        return w
    beta = (np.sqrt(lip) - np.sqrt(max(eig[0], 0.))) / (np.sqrt(lip) + np.sqrt(max(eig[0], 0.)))
    q = w.copy()
    for i in range(2000):
        nxt = _project(q - (h @ q - b) / lip, cap, total)
        q = nxt + beta * (nxt - w)
        w = nxt
        g = h @ w - b
        d = g - _dual(g, w, cap)
        gap = np.maximum(d, 0.) @ w + np.maximum(-d, 0.) @ (cap - w)
        if gap < 1e-12:
            break
        if i % 8 == 0:
            # Solve on the current face, accepting only a feasible KKT point.
            free = (w > 1e-10) & (w < cap - 1e-10)
            if free.any():
                trial = np.where(w >= cap - 1e-10, cap, 0.)
                hf = h[np.ix_(free, free)]
                ones = np.ones(free.sum())
                try:
                    sol, direction = np.linalg.solve(hf, np.column_stack((b[free] - h[free] @ trial, ones))).T
                except np.linalg.LinAlgError:
                    continue
                nu = (total - trial.sum() - sol.sum()) / direction.sum()
                trial[free] = sol + nu * direction
                d = h @ trial - b - nu
                if (trial.min() >= -1e-12 and trial.max() <= cap + 1e-12 and
                    np.min(d[trial <= 0], initial=0.) >= -1e-11 and
                    np.max(d[trial >= cap], initial=0.) <= 1e-11):
                    return np.clip(trial, 0., cap)
    return w


def solve_batch(study: dict) -> dict:
    source = np.asarray(study['source_modes'])
    calibrate = _calibration(study['sensor_calibration_sweeps'])
    wall = _walls(study, calibrate, source)
    response, fields = _response(study, wall, source)
    observed = calibrate(study['y_obs_raw'])
    z = np.asarray(study['Z_nuisance'])
    pz = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    s = response - z @ (pz @ response)
    y = observed - z @ (pz @ observed)
    ridge, box, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w = _weights(s, y, ridge, box, budget)
    residual = s @ w - y
    obj = float((residual @ residual + ridge * (w @ w)) / 2)
    g = s.T @ residual + ridge * w
    nu = _dual(g, w, box)
    lower = np.maximum(0., g - nu)
    upper = np.maximum(0., nu - g)
    dual = obj - float(g @ w) + nu * budget - box * float(upper.sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, fields, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': obj, 'dual_lower_bound': dual}}
