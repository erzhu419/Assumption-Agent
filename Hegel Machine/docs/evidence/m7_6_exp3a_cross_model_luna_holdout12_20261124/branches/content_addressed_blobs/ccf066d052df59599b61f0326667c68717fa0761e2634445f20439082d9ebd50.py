import numpy as np
from scipy.optimize import least_squares


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate a Gaussian plume and zero-mean receiver clock offsets."""
    W = float(survey['W_m'])
    H = float(survey['H_m'])
    zs = np.asarray(survey['z_src_m'], dtype=float)
    zr = np.asarray(survey['z_rcv_m'], dtype=float)
    T = np.asarray(survey['T_obs_s'], dtype=float)
    ns, nr = T.shape
    R = float(survey.get('R_plume_m', 5.5))
    dz = zr[None, :] - zs[:, None]
    L = np.hypot(W, dz)
    sintheta2 = (dz / L) ** 2
    # Straight-ray quadrature captures the specified slowness integral.
    q, qw = np.polynomial.legendre.leggauss(40)
    u, weights = (q + 1) / 2, qw / 2
    xx = W * u[None, None, :]
    zz = zs[:, None, None] + dz[:, :, None] * u[None, None, :]

    def predict(p):
        xp, zp, amp, s0, eps = p[:5]
        clocks = p[5:] - np.mean(p[5:])
        plume = np.sum(np.exp(-0.5 * ((xx-xp)**2 + (zz-zp)**2) / R**2) * weights, axis=2) * L
        return s0 * L * (1 - eps * sintheta2) + amp * plume + clocks[None, :]

    lo = [0., 0., 0., 0.00015, -0.15] + [-0.004] * nr
    hi = [W, H, 0.003, 0.0015, 0.15] + [0.004] * nr
    best = None
    for xp in np.linspace(4, W-4, 7):
        for zp in np.linspace(5, H-5, 9):
            p0 = [xp, zp, 0.0005, 0.00035, 0.] + [0.] * nr
            def residual(p):
                return (predict(p) - T).ravel()
            fit = least_squares(residual, p0, bounds=(lo, hi), loss='soft_l1',
                                f_scale=0.00015, max_nfev=100)
            score = np.median(np.abs(residual(fit.x)))
            if best is None or score < best[0]:
                best = score, fit.x
    p = best[1]
    clocks = p[5:] - np.mean(p[5:])
    return {'plume_xz_m': [float(p[0]), float(p[1])],
            'rcv_clock_bias_s': clocks.astype(float).tolist()}
