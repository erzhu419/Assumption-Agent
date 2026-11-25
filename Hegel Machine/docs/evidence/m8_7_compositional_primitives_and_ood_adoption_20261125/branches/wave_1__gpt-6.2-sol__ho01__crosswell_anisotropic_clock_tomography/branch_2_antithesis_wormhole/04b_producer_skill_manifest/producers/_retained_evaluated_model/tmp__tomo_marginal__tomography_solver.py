"""Robust straight-ray inversion for a circular slowness anomaly."""

def __hegel_load_skill_autogenous_prim_ridge_profiled_matching():
    import types as __hegel_types
    import numpy as np
    import numpy as np

    def fit_ridge_nuisance(design: 'np.ndarray', values: 'np.ndarray', *, ridge: 'float | np.ndarray'=1e-06) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        """Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."""
        design = np.asarray(design)
        values = np.asarray(values)
        if design.ndim != 2 or values.ndim not in (1, 2) or len(values) != len(design):
            raise ValueError('Design and values must share their observation axis')
        (n, p) = design.shape
        penalty = np.broadcast_to(np.asarray(ridge, dtype=float), (p,))
        if np.any(~np.isfinite(penalty)) or np.any(penalty < 0):
            raise ValueError('Ridge penalties must be finite and nonnegative')
        augmented = np.vstack((design, np.diag(np.sqrt(penalty))))
        (u, s, vh) = np.linalg.svd(augmented, full_matrices=False)
        cutoff = np.finfo(s.dtype).eps * max(augmented.shape) * s.max(initial=0.0)
        keep = s > cutoff
        whitener = vh[keep] / s[keep, None]
        coefficients = whitener.conj().T @ (u[:n, keep].conj().T @ values)
        return (coefficients, values - design @ coefficients, whitener)

    def profile_candidate_amplitudes(residual_dot: 'np.ndarray', candidate_energy: 'np.ndarray', nuisance_cross: 'np.ndarray', whitener: 'np.ndarray', *, rtol: 'float'=1e-12) -> 'tuple[np.ndarray, np.ndarray]':
        """Profile residual inner products (m,), raw energies (m,), and nuisance cross-products (p, m) using a whitener (r, p), returning amplitudes and remaining energies, both (m,)."""
        dot = np.asarray(residual_dot)
        energy = np.asarray(candidate_energy, dtype=float)
        cross = np.asarray(nuisance_cross)
        whitener = np.asarray(whitener)
        if dot.ndim != 1 or energy.shape != dot.shape or whitener.ndim != 2:
            raise ValueError('Expected matching candidate vectors and a matrix whitener')
        if cross.shape != (whitener.shape[1], dot.size):
            raise ValueError('Nuisance cross-products must have shape (p, m)')
        if not 0 <= rtol < 1 or np.any(energy < 0):
            raise ValueError('Expected nonnegative energies and 0 <= rtol < 1')
        leverage = np.sum(np.abs(whitener @ cross) ** 2, axis=0)
        remaining = np.maximum(energy - leverage, 0.0)
        amplitude = np.zeros(dot.shape, dtype=np.result_type(dot.dtype, np.float64))
        np.divide(dot, remaining, out=amplitude, where=remaining > rtol * energy)
        return (amplitude, remaining)
    return __hegel_types.SimpleNamespace(fit_ridge_nuisance=fit_ridge_nuisance, profile_candidate_amplitudes=profile_candidate_amplitudes)
__hegel_skill_ns_autogenous_prim_ridge_profiled_matching = __hegel_load_skill_autogenous_prim_ridge_profiled_matching()
import numpy as np
from scipy.linalg import helmert
from scipy.optimize import least_squares
from scipy.special import erf, log_ndtr
fit_ridge_nuisance = __hegel_skill_ns_autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
profile_candidate_amplitudes = __hegel_skill_ns_autogenous_prim_ridge_profiled_matching.profile_candidate_amplitudes

def _integral(x, z, radius, zs, ex, ez, length, jac=False):
    """Integrate the unit-peak Gaussian along each finite ray segment."""
    dz = z - zs
    u = x * ex + dz * ez
    d = dz * ex - x * ez
    root2r = np.sqrt(2) * radius
    transverse = np.exp(-0.5 * (d / radius) ** 2)
    g = transverse * (radius * np.sqrt(np.pi / 2)) * (erf((length - u) / root2r) + erf(u / root2r))
    if not jac:
        return g
    gu = transverse * (np.exp(-0.5 * (u / radius) ** 2) - np.exp(-0.5 * ((length - u) / radius) ** 2))
    gd = -d * g / radius ** 2
    return (g, ex * gu - ez * gd, ez * gu + ex * gd)

def invert_plume_and_clocks(survey: dict) -> dict:
    (width, height) = (float(survey['W_m']), float(survey['H_m']))
    radius = float(survey.get('R_plume_m', 5.5))
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    picks = np.asarray(survey['T_obs_s'], dtype=float).ravel() * 1000
    good = np.isfinite(picks)
    y = picks[good]
    zs = np.repeat(src, len(rcv))[good]
    zr = np.tile(rcv, len(src))[good]
    length = np.hypot(width, zr - zs)
    (ex, ez) = (width / length, (zr - zs) / length)
    gauge = helmert(len(rcv)).T
    clocks = np.tile(gauge, (len(src), 1))[good]
    design = np.column_stack((length / width, length / width * ez ** 2, clocks))
    ncoef = design.shape[1]
    lo = np.r_[[0.0, 0.0, 0.0], np.full(ncoef, -np.inf)]
    hi = np.r_[[width, height, np.inf], np.full(ncoef, np.inf)]
    ray = (radius, zs, ex, ez, length)
    full_gaussian = radius * np.sqrt(2 * np.pi)
    w = np.ones(len(y))
    for _ in range(6):
        a = design * w[:, None]
        (_, residual, _) = fit_ridge_nuisance(a, y * w, ridge=0.0)
        w = 1.0 / np.sqrt(1.0 + (residual / w / 1.4) ** 2)
    a = design * w[:, None]
    (_, residual, whitener) = fit_ridge_nuisance(a, y * w, ridge=0.0)
    step = radius * 0.4
    grid = np.stack(np.meshgrid(np.linspace(0, width, int(np.ceil(width / step)) + 1), np.linspace(0, height, int(np.ceil(height / step)) + 1)), -1).reshape(-1, 2)
    kernel = _integral(grid[:, 0], grid[:, 1], radius, zs[:, None], ex[:, None], ez[:, None], length[:, None]) / full_gaussian
    weighted = kernel * w[:, None]
    dot = np.einsum('i,ij->j', residual, weighted)
    energy = np.einsum('ij,ij->j', weighted, weighted)
    (amplitude, remaining) = profile_candidate_amplitudes(dot, energy, a.T @ weighted, whitener)
    amplitude = np.maximum(amplitude, 0.0)
    score = amplitude ** 2 * remaining

    def fun(p):
        g = _integral(p[0], p[1], *ray) / full_gaussian
        return p[2] * g + np.einsum('ij,j->i', design, p[3:]) - y

    def jac(p):
        (g, gx, gz) = _integral(p[0], p[1], *ray, jac=True)
        return np.column_stack((p[2] * gx / full_gaussian, p[2] * gz / full_gaussian, g / full_gaussian, design))
    best = None
    centers = []
    for i in np.argsort(-score, kind='stable'):
        if any((np.linalg.norm(grid[i] - c) < 1.5 * radius for c in centers)):
            continue
        centers.append(grid[i])
        (b, _, _) = fit_ridge_nuisance(a, (y - amplitude[i] * kernel[:, i]) * w, ridge=0.0)
        fit = least_squares(fun, np.r_[grid[i], amplitude[i], b], jac=jac, bounds=(lo, hi), loss='cauchy', f_scale=0.35, max_nfev=85)
        if best is None or fit.cost < best.cost:
            best = fit
        if len(centers) == 4:
            break
    p = best.x
    for _ in range(3):
        r = fun(p)
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = np.abs(r) < np.clip(5.5 * scale, 0.40, 2.5)
        # A weak constraint on fractional plume strength matters only when
        # the ray data poorly separate plume amplitude from position.
        factor = full_gaussian / width
        prior = np.zeros_like(p)
        prior[2], prior[3] = 1., -0.17 * factor
        prior *= max(scale, 0.01) / (0.06 * max(p[3] * factor, 0.1))
        pnew = least_squares(lambda q: np.r_[fun(q)[keep], prior @ q], p,
                             jac=lambda q: np.vstack((jac(q)[keep], prior)),
                             bounds=(lo, hi), max_nfev=50).x
        change = np.max(np.abs(pnew - p))
        p = pnew
        if change < 1e-7:
            break

    # When ray coverage makes the local position uncertain, integrate the
    # likelihood over the physical domain rather than returning its peak.
    j = jac(p)[keep]
    sigma = max(0.005, np.linalg.norm(fun(p)[keep]) / np.sqrt(max(1, keep.sum() - len(p))))
    cov = np.linalg.pinv(j.T @ j, rcond=1e-12)
    if sigma * np.sqrt(np.linalg.eigvalsh(cov[:2, :2])[-1]) > 0.35:
        dx = min(0.5, radius / 10.)
        xx = np.linspace(0., width, int(np.ceil(width / dx)) + 1)
        zz = np.linspace(0., height, int(np.ceil(height / dx)) + 1)
        gx, gz = np.meshgrid(xx, zz)
        coords = np.column_stack((gx.ravel(), gz.ravel()))
        a_clean = design[keep]
        b0, r0, white = fit_ridge_nuisance(a_clean, y[keep], ridge=0.)
        mean = 0.17 * full_gaussian / width * p[3]
        sd = 0.06 * full_gaussian / width * p[3]
        prior_precision = (sigma / sd)**2
        logmass = np.empty(len(coords))
        bv = np.empty((len(b0), len(coords)))
        for start in range(0, len(coords), 1024):
            sl = slice(start, start + 1024)
            xy = coords[sl]
            k = _integral(xy[:, 0], xy[:, 1], radius, zs[keep, None],
                          ex[keep, None], ez[keep, None], length[keep, None]) / full_gaussian
            cross = a_clean.T @ k
            dot = r0 @ k
            energy = np.einsum('ij,ij->j', k, k)
            _, remaining = profile_candidate_amplitudes(dot, energy, cross, white)
            prec = remaining + prior_precision
            nat = dot + prior_precision * mean
            t = nat / (sigma * np.sqrt(prec))
            logcdf = log_ndtr(t)
            logmass[sl] = 0.5 * (t**2 - np.log(prec)) + logcdf
            amp = nat / prec + sigma / np.sqrt(prec) * np.exp(-0.5 * t**2 - 0.5 * np.log(2*np.pi) - logcdf)
            bv[:, sl] = b0[:, None] - (white.T @ (white @ cross)) * amp
        mass = np.exp(logmass - np.max(logmass)).reshape(gx.shape)
        mass[[0, -1], :] *= 0.5
        mass[:, [0, -1]] *= 0.5
        mass = mass.ravel() / mass.sum()
        p[:2] = mass @ coords
        p[3:] = bv @ mass

    bias = gauge @ p[5:]
    if np.max(np.abs(bias)) > 4.0:
        (low, high) = (np.min(bias) - 4.0, np.max(bias) + 4.0)
        for _ in range(48):
            mid = (low + high) / 2
            if np.clip(bias - mid, -4.0, 4.0).sum() > 0:
                low = mid
            else:
                high = mid
        bias = np.clip(bias - (low + high) / 2, -4.0, 4.0)
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': (bias / 1000.0).tolist()}
