"""Calibrated steady transport and a certified box-simplex source fit."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _calibration(study):
    sweep = study['sensor_calibration_sweeps']
    ref = np.asarray(sweep['ref_rosette'])
    raw = np.asarray(sweep['raw_rosette'])
    coef = np.linalg.lstsq(np.column_stack((ref, np.ones(len(ref)))), raw, rcond=None)[0]

    def decode(values):
        groups = np.asarray(values).reshape(-1, 4)
        return np.linalg.solve(coef[:4].T, (groups - coef[4]).T).T.ravel()

    return decode


class _Transport:
    def __init__(self, study):
        self.nx, self.ny = int(study['nx']), int(study['ny'])
        self.shifts = study['mu_shifts']
        sensors = np.asarray(study['sensor_ij'])
        self.sensors = sensors[:, 0] + self.nx * sensors[:, 1]
        operators, nominal, faces = [], [], []
        for ax in 'xy':
            k = np.asarray(study['kappa_' + ax])
            h = study['L' + ax] / (len(k) - 1)
            v = np.asarray(study['vel_' + ax]) / h
            d = (k[:-1] + k[1:]) / (2 * h**2)
            lo = -d - np.maximum(v[1:], 0)
            hi = -d + np.minimum(v[:-1], 0)
            center = np.r_[d, 0.] + np.r_[0., d] + np.abs(v)
            center[[0, -1]] -= np.maximum(v[[0, -1]] * [1, -1], 0)
            operators.append(diags((lo, center, hi), (-1, 0, 1)))
            nominal.extend([study['robin_' + ax] / h] * 2)
            endpoints = np.zeros((len(k), 2))
            endpoints[0, 0] = endpoints[-1, 1] = 1
            faces.append(endpoints)
        self.nominal = np.array(nominal)
        self.faces = np.column_stack((np.tile(faces[0], (self.ny, 1)),
                                      np.repeat(faces[1], self.nx, axis=0)))
        self.base = (kron(eye(self.ny), operators[0]) + kron(operators[1], eye(self.nx))
                     + diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
        self.sources = np.column_stack([f.ravel(order='F') for f in study['source_modes']])

    def response(self, losses, rhs, with_jac=False):
        base = self.base + diags(self.faces @ losses)
        readings, deriv = [], []
        for shift in self.shifts:
            lu = splu((base + diags(np.full(base.shape[0], shift))).tocsc())
            field = lu.solve(rhs)
            readings.append(field[self.sensors])
            if with_jac:
                drhs = (-field[:, :, None] * self.faces[:, None, :]).reshape(len(field), -1)
                deriv.append(lu.solve(drhs)[self.sensors].reshape(-1, 4))
        readings = np.concatenate(readings, axis=0)
        return readings, field, np.concatenate(deriv, axis=0) if with_jac else None


def _pulse_losses(transport, pulses, decode):
    rhs = transport.sources @ np.column_stack([p['w_pulse'] for p in pulses])
    observed = np.column_stack([decode(p['y_pulse_raw']) for p in pulses])
    last, cached = None, None

    def evaluate(loss):
        nonlocal last, cached
        if last is None or not np.array_equal(loss, last):
            signals, _, jac = transport.response(loss, rhs, with_jac=True)
            cached = (signals - observed).ravel(), jac
            last = loss.copy()
        return cached

    fit = least_squares(lambda r: evaluate(r)[0], np.maximum(transport.nominal, 0),
                        jac=lambda r: evaluate(r)[1], bounds=(0, np.inf),
                        ftol=1e-11, xtol=1e-11, gtol=1e-12)
    return fit.x


def _box_simplex(H, c, cap, budget):
    """Feasible active-set Newton, including the all-at-bounds case."""
    n = len(c)
    w = cap * (budget / cap.sum()) if cap.sum() else np.zeros(n)
    if budget <= 0:
        return np.zeros(n), float(np.max(c))
    if budget >= cap.sum():
        return cap.copy(), -float(np.max(H @ cap - c))
    active = np.zeros(n, dtype=np.int8)
    active[cap == 0] = -1
    tol = 1e-12 * max(1., float(np.max(np.abs(c))), float(np.max(np.abs(H @ cap))))
    for _ in range(50*n + 50):
        free = np.flatnonzero(active == 0)
        grad = H @ w - c
        if free.size == 0:
            lo = np.flatnonzero((active == -1) & (cap > 0))
            hi = np.flatnonzero(active == 1)
            i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[j] - grad[i] <= tol:
                return w, -float((grad[i] + grad[j]) / 2)
            active[[i, j]] = 0
            continue
        w_star, inv_one = np.linalg.solve(H[np.ix_(free, free)], np.column_stack((
            c[free] - H[np.ix_(free, active != 0)] @ w[active != 0], np.ones(free.size)))).T
        nu = (w_star.sum() - (budget - w[active != 0].sum())) / inv_one.sum()
        w_star -= nu * inv_one
        old = w[free]
        step = w_star - old
        alpha = np.ones(free.size)
        down = (w_star < -1e-13) & (step < 0)
        up = (w_star > cap[free] + 1e-13) & (step > 0)
        alpha[down] = -old[down] / step[down]
        alpha[up] = (cap[free][up] - old[up]) / step[up]
        k = np.argmin(alpha)
        if alpha[k] < 1:
            w[free] = old + alpha[k] * step
            i = free[k]
            active[i] = 1 if up[k] else -1
            w[i] = cap[i] if up[k] else 0.
            continue
        w[free] = np.clip(w_star, 0., cap[free])
        q = H @ w - c + nu
        viol = np.where(cap > 0, q * active, 0.)
        i = np.argmax(viol)
        if viol[i] <= tol:
            return w, float(nu)
        active[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _source_fit(study, S, y):
    Z = np.asarray(study['Z_nuisance'])
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    X = S - Z @ (Pz @ S)
    target = y - Z @ (Pz @ y)
    n = X.shape[1]
    ridge = study['source_ridge']
    H = X.T @ X + ridge * np.eye(n)
    c = X.T @ target
    cap = np.broadcast_to(np.asarray(study['u_box'], dtype=float), (n,))
    budget = float(study['total_budget'])
    w, nu = _box_simplex(H, c, cap, budget)
    q = H @ w - c + nu
    lower = np.where(w <= 0., np.maximum(q, 0.), 0.)
    upper = np.where(w >= cap, np.maximum(-q, 0.), 0.)
    stationarity = q - lower + upper
    residual = X @ w - target
    primal = (residual @ residual + ridge * (w @ w)) / 2
    dual = (primal + nu * (w.sum() - budget) - lower @ w + upper @ (w - cap)
            - stationarity @ np.linalg.solve(H, stationarity) / 2)
    return w, {'weights': w.copy(), 'nu_sum': nu, 'lam_lower': lower, 'lam_upper': upper,
               'primal_objective': float(primal), 'dual_lower_bound': float(min(primal, dual))}


def solve_batch(study: dict) -> dict:
    decode = _calibration(study)
    transport = _Transport(study)
    pulses = study.get('reference_tracer_pulses')
    loss = _pulse_losses(transport, pulses, decode) if pulses else transport.nominal
    signals, terminal, _ = transport.response(loss, transport.sources)
    w, certificate = _source_fit(study, signals, decode(study['y_obs_raw']))
    return {'weights': w, 'terminal_field': (terminal @ w).reshape((transport.nx, transport.ny), order='F'),
            'kkt_certificate': certificate}
