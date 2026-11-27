"""Calibrate the rosette, identify the wall stencil, and solve the source QP."""
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu


def _features(x, degree, cross, saturation=None):
    parts = [np.ones((len(x), 1)), x]
    if saturation is not None:
        # This basis spans both a linear component and x/(1+a*x*x),
        # without becoming degenerate as a approaches zero.
        parts.append(x**3/(1+saturation*x*x))
    else:
        parts.extend(x**d for d in range(2, degree+1))
    for i, j in cross:
        parts.append((x[:, i]*x[:, j])[:, None])
        if degree == 4:
            parts.extend(((x[:, i]**2*x[:, j])[:, None], (x[:, i]*x[:, j]**2)[:, None]))
    if degree == 4:
        parts.extend((x[:, i]*x[:, j]*x[:, k])[:, None]
                     for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4))
    return np.column_stack(parts)


def _calibrator(sweeps):
    ref, raw = (np.asarray(sweeps[k], dtype=float) for k in ('ref_rosette', 'raw_rosette'))
    delta = raw-ref
    banks = sweeps.get('single_bank_sweeps')
    if banks is not None:
        bank = np.asarray(banks['bank_id'], dtype=int)
        signal = np.asarray(banks['ref_signal'])
        isolated = np.zeros((len(bank), 4)); isolated[np.arange(len(bank)), bank] = signal
        isolated_delta = np.asarray(banks['raw_readout'])-signal
    pairs = tuple((i, j) for i in range(4) for j in range(i+1, 4))
    models = [(0, (), False), (1, (), False), (2, (), False), (2, pairs, False),
              (3, (), False), (3, pairs, False), (4, pairs, False),
              (3, (), True), (3, pairs, True)]
    indices = np.arange(len(ref))

    def fit(model, keep):
        degree, cross, rational = model
        def coefficients(saturation):
            X = _features(ref[keep], degree, cross, saturation)
            if banks is None:
                return np.linalg.lstsq(X, delta[keep], rcond=None)[0]
            B = _features(isolated, degree, cross, saturation)
            return np.column_stack([np.linalg.lstsq(np.vstack((X, B[bank == i])),
                                   np.r_[delta[keep, i], isolated_delta[bank == i]],
                                   rcond=None)[0] for i in range(4)])
        saturation = None
        if rational:
            def residual(a):
                coef = coefficients(a)
                e = _features(ref[keep], degree, cross, a) @ coef - delta[keep]
                if banks is None:
                    return e.ravel()
                b = (_features(isolated, degree, cross, a)@coef)[np.arange(len(bank)), bank]
                return np.r_[e.ravel(), b-isolated_delta]
            saturation = least_squares(residual, np.full(4, .15), bounds=(0., 4.),
                                       gtol=1e-13, max_nfev=40).x
        return saturation, coefficients(saturation)

    scores = [np.mean(delta**2)]
    for model in models[1:]:
        degree, cross, rational = model
        err = np.empty_like(delta)
        for f in range(5):
            keep = indices % 5 != f
            sat, coef = fit(model, keep)
            err[~keep] = _features(ref[~keep], degree, cross, sat) @ coef-delta[~keep]
        scores.append(np.mean(err**2))
    cutoff = min(scores)*1.06 + 1e-27
    model = next(m for m, score in zip(models, scores) if score <= cutoff)
    degree, cross, _ = model
    if degree == 0:
        return lambda y: np.asarray(y, dtype=float)
    sat, coef = fit(model, np.ones(len(ref), dtype=bool))
    linear = np.eye(4) + coef[1:5]
    def forward(x):
        return x+_features(x, degree, cross, sat)@coef
    def apply(y):
        raw = np.asarray(y, dtype=float).reshape(-1, 4)
        x = np.linalg.lstsq(linear.T, (raw-coef[0]).T, rcond=None)[0].T
        if degree == 1:
            return x.ravel()
        # Damped Newton iteration starting at the affine inverse. Each
        # accepted update has a smaller raw-space error for that rosette.
        e = forward(x)-raw
        steps = np.eye(4)*1e-5
        for _ in range(20):
            if np.max(np.abs(e), initial=0.) < 1e-12:
                break
            J = np.stack([(forward(x+h)-forward(x-h))/2e-5 for h in steps], axis=2)
            inv = np.linalg.pinv(J, rcond=1e-9)
            d = (inv@e[:, :, None])[:, :, 0]
            d /= np.maximum(1., np.max(np.abs(d), axis=1, keepdims=True))
            old = np.sum(e*e, axis=1)
            pending = np.ones(len(x), dtype=bool)
            for t in (1., .5, .25, .125, .0625, .03125):
                trial = x-t*d
                candidate = forward(trial)-raw
                take = pending & (np.sum(candidate*candidate, axis=1) < old)
                x[take], e[take] = trial[take], candidate[take]
                pending[take] = False
            if pending.all():
                break
        return x.ravel()
    return apply


class _Chamber:
    def __init__(self, study):
        self.nx, self.ny = nx, ny = int(study['nx']), int(study['ny'])
        self.F = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx*ny).T
        ij = np.asarray(study['sensor_ij'], dtype=int)
        self.sensors = ij[:, 0] + nx*ij[:, 1]
        self.shifts = np.asarray(study['mu_shifts'])
        self.eye = sparse.eye(nx*ny, format='csc')
        axes, nominal, inward, scales, offsets = [], [], [], [], []
        for a in 'xy':
            k, v = (np.asarray(study[key+'_'+a]) for key in ('kappa', 'vel'))
            n = len(k); h = study['L'+a] / (n-1)
            d = (k[:-1]+k[1:])/(2*h*h)
            lo = -d-np.maximum(v[1:], 0)/h
            hi = -d-np.maximum(-v[:-1], 0)/h
            diag = np.abs(v)/h
            diag[:-1] += d; diag[1:] += d
            axes.append(sparse.diags((lo, diag, hi), (-1, 0, 1), format='csc'))
            # Boundary rows replace the interior stencil: endpoint diffusion
            # to the inward neighbor, Robin loss, and no boundary advection.
            edge = np.array([0, n-1])
            edge_diffusion = k[edge]/h**2
            nominal.extend(study['robin_'+a]/h + edge_diffusion - diag[edge])
            inward.extend([hi[0], lo[-1]])
            offsets.extend(-edge_diffusion - np.array([hi[0], lo[-1]]))
            scales.extend([d[0], d[-1]])
        ix, iy = sparse.eye(nx), sparse.eye(ny)
        self.base = (sparse.kron(iy, axes[0])+sparse.kron(axes[1], ix)
                     + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
        self.prior = np.r_[nominal, offsets]
        self.inward, self.scales = np.array(inward), np.array(scales)
        self.directions = []
        for coupling in (False, True):
            for axis, n in enumerate((nx, ny)):
                for row, inner in ((0, 1), (n-1, n-2)):
                    edge = sparse.coo_matrix(([1.], ([row], [inner if coupling else row])), shape=(n, n))
                    term = sparse.kron(iy, edge) if axis == 0 else sparse.kron(edge, ix)
                    self.directions.append(term.tocsc())

    def response(self, wall, rhs=None, derivatives=False, shifts=None):
        if rhs is None:
            rhs = self.F
        A = self.base.copy()
        for a, D in zip(wall, self.directions):
            A += a*D
        obs, jac = [], []
        for mu in self.shifts if shifts is None else shifts:
            lu = splu(A + mu*self.eye)
            field = lu.solve(rhs)
            obs.append(field[self.sensors])
            if derivatives:
                du = lu.solve(np.concatenate([-D@field for D in self.directions], axis=1))
                jac.append(du[self.sensors].reshape(len(self.sensors), 8, rhs.shape[1]).transpose(0, 2, 1))
        return np.vstack(obs), field, np.concatenate(jac) if derivatives else None


def _box_qp(H, g, upper, budget):
    """Primal feasible active set for a strictly convex equality-constrained QP."""
    n = len(g)
    if budget <= 0 or budget >= upper.sum():
        w = np.zeros(n) if budget <= 0 else upper.copy()
        grad = H@w+g
        return w, -grad.min() if budget <= 0 else -grad.max()
    w = upper*(budget/upper.sum())
    active = np.zeros(n, dtype=int)
    active[upper == 0] = -1
    for _ in range(20*n+20):
        free = np.flatnonzero(active == 0)
        grad = H@w+g
        if not len(free):
            lo = np.flatnonzero((active < 0) & (upper > 0)); hi = np.flatnonzero(active > 0)
            i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[j]-grad[i] <= 2e-12:
                return w, -(grad[i]+grad[j])/2
            active[[i, j]] = 0
            continue
        Hf = H[np.ix_(free, free)]
        q, a = solve(Hf, np.column_stack((grad[free], np.ones(len(free)))), assume_a='pos').T
        nu = -q.sum()/a.sum()
        direction = -q-nu*a
        with np.errstate(divide='ignore', invalid='ignore'):
            steps = np.where(direction > 1e-14, (upper[free]-w[free])/direction,
                             np.where(direction < -1e-14, -w[free]/direction, np.inf))
        j = int(np.argmin(steps)); t = min(1., max(0., steps[j]))
        w[free] += t*direction
        if t < 1.-1e-12:
            i = free[j]
            active[i] = 1 if direction[j] > 0 else -1
            w[i] = upper[i] if active[i] == 1 else 0.
            continue
        reduced = H@w+g+nu
        violation = np.where(upper > 0, active*reduced, 0.)
        i = int(np.argmax(violation))
        if violation[i] <= 2e-12:
            return w, float(nu)
        active[i] = 0
    raise RuntimeError('Source QP active set did not converge')


def _projector(Z, ridge):
    Pz = np.linalg.solve(Z.T@Z + ridge*np.eye(Z.shape[1]), Z.T)
    return lambda a: a-Z@(Pz@a)


def _fit_walls(ch, study, calibrate):
    pulses = study.get('reference_tracer_pulses')
    prior = ch.prior
    if not pulses:
        return prior.copy()
    W = np.column_stack([p['w_pulse'] for p in pulses])
    target = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
    rhs = ch.F@W
    def fun(w):
        return (ch.response(w, rhs)[0]-target).ravel()
    def jac(w):
        return ch.response(w, rhs, True)[2].reshape(-1, 8)
    # Exact or nearly exact agreement does not justify fitting extra wall terms.
    if np.sqrt(np.mean(fun(prior)**2)) < 5e-7:
        return prior.copy()
    scales = np.r_[ch.scales, ch.scales]
    lo = prior - .65*scales
    hi = prior + .65*scales
    hi[4:] = np.minimum(hi[4:], prior[4:] - .9*(ch.inward+prior[4:]))
    return least_squares(fun, prior, jac=jac, x_scale=scales, bounds=(lo, hi),
                         gtol=2e-14, ftol=1e-12, xtol=1e-12).x


def solve_batch(study: dict) -> dict:
    calibrate = _calibrator(study['sensor_calibration_sweeps'])
    y = calibrate(study['y_obs_raw'])
    chamber = _Chamber(study)
    walls = _fit_walls(chamber, study, calibrate)
    S, U, _ = chamber.response(walls)
    P = _projector(np.asarray(study['Z_nuisance']), study['nuisance_ridge'])
    X, b = P(S), P(y)
    ridge = study['source_ridge']; budget = study['total_budget']
    upper = np.broadcast_to(study['u_box'], (S.shape[1],))
    H = X.T@X + ridge*np.eye(S.shape[1]); g = -X.T@b
    w, nu = _box_qp(H, g, upper, budget)
    reduced = H@w+g+nu
    lam_lower = np.where(w <= 1e-10, np.maximum(reduced, 0.), 0.)
    lam_upper = np.where(w >= upper-1e-10, np.maximum(-reduced, 0.), 0.)
    r = X@w-b
    primal = float((r@r+ridge*(w@w))/2)
    stationarity = reduced-lam_lower+lam_upper
    lagrangian = primal + nu*(w.sum()-budget)-lam_lower@w+lam_upper@(w-upper)
    dual = min(primal, float(lagrangian-.5*stationarity@solve(H, stationarity, assume_a='pos')))
    return {'weights': w, 'terminal_field': (U@w).reshape((chamber.nx, chamber.ny), order='F'),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': float(nu), 'lam_lower': lam_lower,
                                'lam_upper': lam_upper, 'primal_objective': primal, 'dual_lower_bound': dual}}
