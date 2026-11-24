import numpy as np
from scipy.optimize import least_squares


def invert_plume_and_clocks(survey: dict) -> dict:
    W, H = float(survey['W_m']), float(survey['H_m'])
    zs = np.asarray(survey['z_src_m'], dtype=float)
    zr = np.asarray(survey['z_rcv_m'], dtype=float)
    T = np.asarray(survey['T_obs_s'], dtype=float)
    ns, nr = T.shape
    # Straight-ray integral using Gauss-Legendre quadrature along every source-receiver path.
    q, w = np.polynomial.legendre.leggauss(48)
    u = (q + 1.0) / 2.0
    weights = w / 2.0
    dz = zr[None, :] - zs[:, None]
    length = np.hypot(W, dz)
    sin2 = dz * dz / (W * W + dz * dz)
    X = W * u[:, None, None]
    Z = zs[None, :, None] + dz[None, :, :] * u[:, None, None]
    # Parameters: s0, eps, delta_s, plume x/z, first nr-1 clock offsets.
    def predict(p):
        s0, eps, ds, xp, zp = p[:5]
        clocks = np.r_[p[5:], -np.sum(p[5:])]
        plume = np.exp(-0.5 * ((X - xp)**2 + (Z - zp)**2) / 5.5**2)
        integ = np.sum(weights[:, None, None] * plume, axis=0)
        bg = length * s0 * (1.0 - eps * sin2)
        return bg + length * ds * integ + clocks[None, :]
    def residual(p):
        return (predict(p) - T).ravel()
    # Robust soft-L1 suppresses cycle skips; multiple plume starts reduce local minima.
    lo = np.r_[0.0001, -0.2, 0.0, 0.0, 0.0, np.full(nr-1, -0.004)]
    hi = np.r_[0.002, 0.2, 0.002, W, H, np.full(nr-1, 0.004)]
    best = None
    for xp0, zp0 in [(W/2, H/2), (W/4, H/3), (W/4, 2*H/3),
                     (3*W/4, H/3), (3*W/4, 2*H/3), (W/2, H*0.6)]:
        p0 = np.r_[0.0005, 0.03, 0.00012, xp0, zp0, np.zeros(nr-1)]
        fit0 = least_squares(residual, p0, bounds=(lo, hi), loss='soft_l1', f_scale=0.00012,
                             max_nfev=180, x_scale='jac')
        r = residual(fit0.x) / 0.00012
        score = float(np.sum(2 * (np.sqrt(1 + r*r) - 1) * 0.00012**2))
        if best is None or score < best[0]:
            best = (score, fit0)
    fit = best[1]
    clocks = np.r_[fit.x[5:], -np.sum(fit.x[5:])]
    return {'plume_xz_m': [float(fit.x[3]), float(fit.x[4])],
            'rcv_clock_bias_s': clocks.astype(float).tolist()}
