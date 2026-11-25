"""Invert a borehole voltage survey with additive electrode contacts."""
import numpy as np
from scipy.linalg import helmert, qr, solve_triangular
from scipy.optimize import least_squares


def invert_ert_anomaly(survey: dict) -> dict:
    """Return the anomaly coordinates and zero-sum receiver voltages."""
    L, D = float(survey['L_m']), float(survey['D_m'])
    a = float(survey.get('a0_m', 6.0))
    tx = np.asarray(survey['z_tx_m'], dtype=float)
    rx = np.asarray(survey['z_rx_m'], dtype=float)
    y = np.asarray(survey['V_obs'], dtype=float).ravel()
    m, n = len(tx), len(rx)
    dz = rx[None, :] - tx[:, None]
    R = np.hypot(L, dz)
    Ht, Hr = helmert(m).T, helmert(n).T
    background = np.column_stack((
        (L / R).ravel(), (L * dz**2 / R**3).ravel(),
        np.repeat(Ht, n, axis=0), np.tile(Hr, (m, 1))))
    ii, jj = np.repeat(tx, n), np.tile(rx, m)

    def kernel(p, i, j):
        x, z = p
        return L**2 / np.sqrt((x*x + (z-i)**2 + a*a)
                              * ((L-x)**2 + (z-j)**2 + a*a))

    gx, gz = np.meshgrid(np.linspace(0, L, 23), np.linspace(0, D, 43))
    grid = np.array([gx.ravel(), gz.ravel()])
    K = kernel(grid[:, None, :], ii[:, None], jj[:, None])

    def fit(mask, weights=None):
        w = 1. if weights is None else np.sqrt(weights[mask])
        w = np.broadcast_to(w, (mask.sum(),))
        Q, T = qr(background[mask]*w[:,None], mode='economic')
        yw = y[mask]*w
        yp = yw - Q @ (Q.T @ yw)
        kw = K[mask]*w[:,None]
        kp = kw - Q @ (Q.T @ kw)
        num = np.maximum(yp @ kp, 0.)
        denom = np.einsum('ij,ij->j', kp, kp)
        score = num / np.sqrt(np.maximum(denom, 1e-30))
        i, j = ii[mask], jj[mask]

        def profile(p):
            k = kernel(p, i, j)*w
            proj = k - Q @ (Q.T @ k)
            amp = max(0., (proj @ yp) / max(proj @ proj, 1e-30))
            return amp * proj - yp, amp, k

        def objective(p):
            return profile(p)[0]

        starts = []
        for idx in np.argsort(score)[::-1]:
            p = grid[:, idx]
            if all(np.linalg.norm((p - q) / (L, D)) >= .12 for q in starts):
                starts.append(p)
                if len(starts) == 4:
                    break
        results = [least_squares(objective, p, bounds=([0., 0.], [L, D]),
                                 x_scale=[L, D], max_nfev=50,
                                 ftol=1e-10, gtol=1e-10)
                   for p in starts]
        best = min(results, key=lambda s: s.cost)
        _, amp, k = profile(best.x)
        coeff = solve_triangular(T, Q.T @ (yw - amp*k))
        residual = y - background @ coeff - amp * kernel(best.x, ii, jj)
        return best.x, coeff, residual

    mask = np.ones(y.size, dtype=bool)
    p, coeff, residual = fit(mask)
    for _ in range(4):
        # The specified polarization jumps are about 60 mV; this cut is
        # halfway between a clean channel and either polarization state.
        clean = np.abs(residual) < .03
        if np.array_equal(clean, mask):
            break
        mask = clean
        p, coeff, residual = fit(mask)
    # If the whole spike population is consistent with signed 60 mV
    # jumps, reuse those channels at reduced precision. This allows some
    # uncertainty in the jump size without treating it as clean noise.
    weights = np.ones(y.size)
    if np.any(~mask):
        sigma = np.sqrt((residual[mask] @ residual[mask]) / (mask.sum()-background.shape[1]-3))
        jump = .06 * np.sign(residual)
        discrepancy = residual[~mask] - jump[~mask]
        if np.sqrt(np.mean(discrepancy**2)) < 2.2 * max(sigma, 1e-5):
            weights[~mask] = .65
            y = y - np.where(mask, 0., jump)
            mask = np.ones(y.size, dtype=bool)
            p, coeff, residual = fit(mask, weights)
    # Average over the local position likelihood only away from the walls,
    # where truncating the likelihood would bias the estimate inward.
    if 4 < p[0] < L-4 and 4 < p[1] < D-4:
        sw = np.sqrt(weights[mask])
        Q, T = qr(background[mask]*sw[:,None], mode='economic')
        yy = y[mask]*sw
        yp = yy - Q @ (Q.T @ yy)
        rad = np.array([min(6., p[0], L-p[0]), min(6., p[1], D-p[1])])
        xg, zg = np.meshgrid(p[0] + rad[0]*np.linspace(-1, 1, 33),
                             p[1] + rad[1]*np.linspace(-1, 1, 33))
        kg = kernel((xg.ravel()[None,:], zg.ravel()[None,:]), ii[mask,None], jj[mask,None])
        kg *= sw[:,None]
        kg -= Q @ (Q.T @ kg)
        score = np.maximum(yp @ kg, 0.)**2 / np.maximum(np.einsum('ij,ij->j', kg, kg), 1e-30)
        variance = max(((residual[mask]*weights[mask]) @ residual[mask]) / max(mask.sum()-Q.shape[1]-3, 1), 1e-14)
        w = np.exp((score-score.max())/(2*variance))
        p = np.array([w @ xg.ravel(), w @ zg.ravel()])/w.sum()
        k = kernel(p, ii[mask], jj[mask])*sw
        kp = k-Q@(Q.T@k)
        amp = max(0., (yp @ kp)/max(kp @ kp, 1e-30))
        coeff = solve_triangular(T, Q.T@(yy-amp*k))
    offsets = Hr @ coeff[2 + m - 1:]
    return {'anomaly_xz_m': p.tolist(), 'rx_contact_offset_v': offsets.tolist()}
