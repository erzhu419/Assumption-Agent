"""Invert straight-ray crosswell picks for a Gaussian plume and receiver clocks."""

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import brentq, least_squares, minimize
from scipy.special import erf, logsumexp, roots_legendre


class _Rays:
    def __init__(self, width, radius, sources, receivers):
        dz = receivers[None, :] - sources[:, None]
        self.length = np.hypot(width, dz).ravel()
        self.c = width / self.length
        self.s = dz.ravel() / self.length
        self.z = np.broadcast_to(sources[:, None], dz.shape).ravel()
        self.r = radius
        self.bg = np.column_stack((self.length, -self.length * self.s**2))

    def plume(self, x, z, derivatives=False):
        u = self.c * x + self.s * (z - self.z)
        v = -self.s * x + self.c * (z - self.z)
        end = self.length - u
        r = self.r
        cross = np.exp(-0.5 * (v / r)**2)
        span = (erf(end / (np.sqrt(2) * r)) + erf(u / (np.sqrt(2) * r)))
        k = r * np.sqrt(np.pi / 2) * cross * span
        if not derivatives:
            return k
        du = cross * (np.exp(-0.5 * (u / r)**2) - np.exp(-0.5 * (end / r)**2))
        dv = -v * k / r**2
        return k, self.c * du - self.s * dv, self.s * du + self.c * dv


def _linear(a, y, weights):
    return np.linalg.lstsq(a * weights[:, None], y * weights, rcond=None)[0]


def _seeds(rays, a, y, width, height):
    # Adapt Cauchy reweighting to scalar picks. Remove clock and background
    # components from every plume hypothesis before scoring its ray pattern.
    scale = 0.35  # milliseconds; broad enough to retain signal under the null
    weights = np.ones_like(y)
    for _ in range(9):
        beta = _linear(a, y, weights)
        weights = 1.0 / np.hypot(1, (y - a @ beta) / scale)
    q = np.linalg.qr(a * weights[:, None], mode='reduced')[0]
    wy = y * weights
    residual = wy - q @ (q.T @ wy)
    step = 0.55 * rays.r
    xs = np.linspace(0, width, int(np.ceil(width / step)) + 1)
    zs = np.linspace(0, height, int(np.ceil(height / step)) + 1)
    x, z = (t.ravel() for t in np.meshgrid(xs, zs))
    score = np.empty_like(x)
    for i in range(0, len(x), 256):
        sl = slice(i, i + 256)
        k = rays.plume(x[sl, None], z[sl, None]) * weights
        projection = np.einsum('ij,jk->ik', k, q)
        norm = np.sqrt(np.maximum(1e-20, np.sum(k*k, axis=1) - np.sum(projection**2, axis=1)))
        score[sl] = np.maximum(0, np.einsum('ij,j->i', k, residual)) / norm
    grid = score.reshape(len(zs), len(xs))
    peaks = np.flatnonzero((grid == maximum_filter(grid, size=3)).ravel() & (score > 0))
    if not len(peaks):
        peaks = np.array([np.argmax(score)])
    best = peaks[np.argsort(-score[peaks])[:8]]
    return zip(x[best], z[best]), weights


def _bounded_clocks(means, counts):
    # Profile the clock nuisance exactly: the shared multiplier enforces the
    # zero-sum gauge even if one or more clocks are on the box boundary.
    def clocks(multiplier):
        return np.clip(means - multiplier / counts, -4., 4.)

    offset = np.sum(means) / np.sum(1 / counts)
    c = means - offset / counts
    if np.max(np.abs(c)) <= 4.:
        return c
    root = brentq(lambda t: np.sum(clocks(t)),
                  np.min((means - 4.) * counts), np.max((means + 4.) * counts))
    return clocks(root)


_NODES, _WEIGHTS = roots_legendre(24)

def _marginal_location(rays, a, y, keep, initial, width, height):
    keep = keep.ravel()
    q = np.linalg.qr(a[keep], mode='reduced')[0]
    def project(v):
        return v - q @ (q.T @ v)
    res = project(y[keep])
    k, kx, kz = rays.plume(*initial[:2], derivatives=True)
    e = res - initial[2] * project(k[keep])
    sigma = max(.012, np.median(abs(e - np.median(e))) / .67449)
    jac = np.column_stack((initial[2]*kx[keep], initial[2]*kz[keep], k[keep]))
    _, singular, axes = np.linalg.svd(project(jac), full_matrices=False)
    uncertainty = np.linalg.norm(axes.T[:2] * (sigma / np.maximum(singular, 1e-12)), ord=2)
    # Avoid quantizing well-resolved likelihood peaks on the edge-search grid.
    if uncertainty < .40 and min(initial[0], width-initial[0], initial[1], height-initial[1]) > 2:
        return initial[:2]
    xs = np.linspace(0, width, int(np.ceil(width / .55)) + 1)
    zs = np.linspace(0, height, int(np.ceil(height / .55)) + 1)
    x, z = (v.ravel() for v in np.meshgrid(xs, zs))
    score = np.empty(len(x))
    logref = np.log(max(.008, .17 * initial[3]))
    for i in range(0, len(x), 128):
        j = slice(i, i+128)
        k = rays.plume(x[j,None], z[j,None])[:, keep]
        f = k - np.einsum('ij,kj->ik', np.einsum('ij,jk->ik', k, q), q)
        ss = np.maximum(np.sum(f*f, axis=1), 1e-14)
        ty = np.einsum('ij,j->i', f, res)
        mean, sd = ty / ss, sigma / np.sqrt(ss)
        lo = np.clip(mean-8*sd, .003, .79)
        hi = np.clip(mean+8*sd, lo+1e-7, .8)
        half = .5 * (hi - lo)
        amp = (lo+half)[:,None] + half[:,None]*_NODES
        ls = ((ty[:,None]*amp - .5*ss[:,None]*amp**2)/sigma**2
              - .5*((np.log(amp)-logref)/0.48)**2 - np.log(amp))
        score[j] = logsumexp(ls, b=_WEIGHTS, axis=1) + np.log(half)
    mass = np.exp(score - np.max(score)).reshape(len(zs),len(xs))
    mass[[0,-1],:] *= .5
    mass[:,[0,-1]] *= .5
    mass = mass.ravel() / mass.sum()
    return np.array([mass @ x, mass @ z])


def invert_plume_and_clocks(survey: dict) -> dict:
    """Return the plume center in metres and receiver clock biases in seconds."""
    width, height = float(survey['W_m']), float(survey['H_m'])
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    y = np.asarray(survey['T_obs_s'], dtype=float).ravel() * 1000.
    rays = _Rays(width, float(survey.get('R_plume_m', 5.5)), src, rcv)
    clock = np.eye(len(rcv))[:, :-1]
    clock[-1] = -1
    a = np.column_stack((rays.bg, np.tile(clock, (len(src), 1))))
    def fit_model(target):
        seeds, weights = _seeds(rays, a, target, width, height)

        # p = (plume x, plume z, amplitude, background coefficients, free clocks).
        def residual(p):
            return a @ p[3:] + p[2] * rays.plume(*p[:2]) - target

        def jacobian(p):
            k, kx, kz = rays.plume(*p[:2], derivatives=True)
            return np.column_stack((p[2] * kx, p[2] * kz, k, a))

        best = None
        for x, z in seeds:
            beta = _linear(np.column_stack((rays.plume(x, z), a)), target, weights)
            beta[0] = max(beta[0], 0.001)
            fit = least_squares(residual, np.r_[x, z, beta], jac=jacobian,
                                bounds=(np.r_[0., 0., 0., np.full(a.shape[1], -np.inf)],
                                        np.r_[width, height, np.full(a.shape[1]+1, np.inf)]),
                                x_scale='jac', loss='cauchy', f_scale=0.18, max_nfev=90)
            if best is None or fit.cost < best.cost:
                best = fit
        return best

    best = fit_model(y)
    initial_raw = (-best.fun).reshape(len(src), len(rcv))
    # A majority of skips on one receiver can put its fitted clock on the
    # wrong cycle. Compare the two modes under the shared zero-sum gauge.
    def rank(fit):
        errors = np.abs(a @ fit.x[3:] + fit.x[2] * rays.plume(*fit.x[:2]) - y)
        n = max(1, int(0.90 * errors.size))
        trimmed = np.partition(errors, n-1)[:n]
        biases = clock @ fit.x[5:]
        return float(np.mean(trimmed**2) + np.mean(np.maximum(0., np.abs(biases)-4.)**2))

    best_rank = rank(best)
    print("initial",round(best_rank,5),np.round(clock @ best.x[5:],2), flush=True)
    for j in range(len(rcv)):
        r = initial_raw[:, j]
        order = np.argsort(r)
        values = r[order]
        # Find the mode separation independently of the initial clock origin.
        minimum = max(3, int(np.ceil(.25 * len(src))))
        cuts = np.arange(minimum, len(src) - minimum + 1)
        if not cuts.size:
            continue
        gaps = values[cuts] - values[cuts-1]
        if np.max(gaps) > 2:
            print("gap diagnostic", j, np.round(values,2), flush=True)
        cut = cuts[np.argmax(gaps)]
        if np.max(gaps) < 3.0:
            continue
        low, high = order[:cut], order[cut:]
        step = np.median(r[high]) - np.median(r[low])
        if not 4. <= step <= 8.:
            continue
        for rows, shift in ((low, step), (high, -step)):
            alternative = y.copy().reshape(initial_raw.shape)
            alternative[rows, j] += shift
            fit = fit_model(alternative.ravel())
            score = rank(fit)
            print("alternative",j,round(shift,3),round(score,5),np.round(clock @ fit.x[5:],2),flush=True)
            if score < best_rank:
                best_rank = score
                best = fit
    # Final screening always uses the original observations.
    raw = y - (a @ best.x[3:] + best.x[2] * rays.plume(*best.x[:2]))
    # Reuse skipped picks only if they have a survey-wide shift resolved at
    # the inlier noise scale. Do not assume the nominal 6 ms is exact.
    suspect = np.abs(raw) >= 2.5
    clean = raw[~suspect]
    if suspect.sum() >= 4 and clean.size:
        shift = np.median(np.abs(raw[suspect]))
        sigma = max(0.012, np.median(np.abs(clean - np.median(clean))) / 0.67449)
        scatter = np.abs(raw[suspect]) - shift
        if (4 < shift < 8 and np.sqrt(np.mean(scatter**2)) < 1.6 * sigma
                and np.max(np.abs(scatter)) < 3.4 * sigma):
            adjustment = np.where(suspect, shift * np.sign(raw), 0.)
            y = y - adjustment
            raw = raw - adjustment
    y2 = y.reshape(len(src), len(rcv))
    p = best.x[:5].copy()
    p[3] = max(p[3], 0.)
    keep = (np.abs(raw) < 2.5).reshape(y2.shape)
    # A clock needs at least one usable pick. Retain the closest pick in a
    # receiver column if it has no residual inside the inlier gate.
    for j in np.flatnonzero(~keep.any(axis=0)):
        keep[np.argmin(np.abs(raw.reshape(y2.shape)[:, j])), j] = True
    counts = keep.sum(axis=0)

    def profile(p, clocks_only=False):
        k, kx, kz = rays.plume(*p[:2], derivatives=True)
        e = y2 - (p[2] * k + rays.bg @ p[3:]).reshape(y2.shape)
        c = _bounded_clocks(np.sum(np.where(keep, e, 0.), axis=0) / counts, counts)
        if clocks_only:
            return c
        e = np.where(keep, e - c, 0.).ravel()
        grad = -np.r_[p[2]*(e @ kx), p[2]*(e @ kz), e @ k, e @ rays.bg]
        return 0.5 * (e @ e), grad

    scales = np.array([rays.r, rays.r, 0.08, 0.4, 0.08])

    def objective(q):
        loss, gradient = profile(q * scales)
        return loss, gradient * scales

    final = minimize(objective, p / scales, jac=True, method='L-BFGS-B',
                     bounds=((0, width/scales[0]), (0, height/scales[1]),
                             (0, None), (0, None), (None, None)),
                     options={'ftol': 1e-13, 'gtol': 1e-8, 'maxiter': 160})
    p = final.x * scales
    center = _marginal_location(rays, a, y, keep, p, width, height)
    if not np.array_equal(center, p[:2]):
        # Refit the linear physical parameters and the bounded clocks at the
        # selected center, rather than returning clocks from a different fit.
        p[:2] = center
        fixed = center / scales[:2]

        def fixed_center(q):
            loss, grad = objective(np.r_[fixed, q])
            return loss, grad[2:]

        refit = minimize(fixed_center, p[2:]/scales[2:], jac=True,
                         method='L-BFGS-B', bounds=((0, None), (0, None), (None, None)),
                         options={'ftol': 1e-13, 'gtol': 1e-9})
        p[2:] = refit.x * scales[2:]
    return {'plume_xz_m': p[:2].tolist(),
            'rcv_clock_bias_s': (profile(p, clocks_only=True) / 1000.).tolist()}
