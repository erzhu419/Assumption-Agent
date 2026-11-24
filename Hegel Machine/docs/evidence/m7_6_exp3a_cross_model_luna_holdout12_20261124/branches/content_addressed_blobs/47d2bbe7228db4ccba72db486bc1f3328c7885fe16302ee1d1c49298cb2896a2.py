"""Invert sparse downstream free-surface observations for the canal initial mode and sill."""
import numpy as np


def _simulate(high, sill, T=191):
    N, dx, dt, g = 80, 10.0, 0.5, 9.81
    x = (np.arange(N) + 0.5) * dx
    b = sill * np.exp(-0.5 * ((x - 400.0) / 45.0) ** 2)
    eta0 = 2.65 if high else 1.45
    eta = np.full(N, 2.0)
    eta[:8] = eta0
    h = np.maximum(eta - b, 1e-8)
    q = np.zeros(N)
    out = np.empty((T, N), dtype=float)
    out[0] = h + b
    for n in range(1, T):
        # Reflective ghost states, preserving bed at boundary cell.
        he = np.concatenate(([h[0]], h, [h[-1]]))
        be = np.concatenate(([b[0]], b, [b[-1]]))
        qe = np.concatenate(([-q[0]], q, [-q[-1]]))
        hL, hR = he[:-1], he[1:]
        bL, bR = be[:-1], be[1:]
        qL, qR = qe[:-1], qe[1:]
        uL = qL / np.maximum(hL, 1e-9)
        uR = qR / np.maximum(hR, 1e-9)
        bs = np.maximum(bL, bR)
        hsL = np.maximum(0.0, hL + bL - bs)
        hsR = np.maximum(0.0, hR + bR - bs)
        f0L, f0R = hsL*uL, hsR*uR
        f1L = hsL*uL*uL + 0.5*g*hsL*hsL
        f1R = hsR*uR*uR + 0.5*g*hsR*hsR
        speed = np.maximum(np.maximum(np.abs(uL)+np.sqrt(g*hsL), np.abs(uR)+np.sqrt(g*hsR)), 1e-6)
        f0 = 0.5*(f0L+f0R)-0.5*speed*(hsR-hsL)
        f1 = 0.5*(f1L+f1R)-0.5*speed*(hsR*uR-hsL*uL)
        # Hydrostatic pressure correction, one-sided per cell as specified.
        b_ip = np.maximum(b, np.concatenate((b[1:], b[-1:])))
        b_im = np.maximum(np.concatenate((b[:1], b[:-1])), b)
        hm = np.maximum(0.0, h+b-b_ip)
        hp = np.maximum(0.0, h+b-b_im)
        hn = h - (dt/dx)*(f0[1:]-f0[:-1])
        qn = q - (dt/dx)*(f1[1:]-f1[:-1]) + (dt/dx)*0.5*g*(hm*hm-hp*hp)
        h = np.maximum(hn, 1e-8)
        q = qn
        out[n] = h+b
    return out


def invert_canal_regime(obs_bundle: dict) -> dict:
    """Estimate initial sluice state and sill; return the reconstructed full eta history."""
    obs = np.asarray(obs_bundle['eta_obs'], dtype=float)
    steps = np.asarray(obs_bundle['obs_step_indices'], dtype=int)
    gauges = np.asarray(obs_bundle.get('gauge_cells', [48, 62, 74]), dtype=int)
    # A small one-dimensional grid followed by local refinement avoids dependence on
    # optimizer availability and makes both discrete mode hypotheses explicit.
    best = None
    for high in (False, True):
        def score(s):
            pred = _simulate(high, float(s))
            r = pred[np.ix_(steps, gauges)] - obs
            return float(np.mean(r*r))
        grid = np.linspace(0.0, 0.75, 76)
        vals = [score(s) for s in grid]
        j = int(np.argmin(vals))
        lo, hi = grid[max(0, j-1)], grid[min(len(grid)-1, j+1)]
        # Golden-section minimization in the winning grid bracket.
        a, c = lo, hi
        phi = (np.sqrt(5.0)-1.0)/2.0
        p, qv = c-phi*(c-a), a+phi*(c-a)
        fp, fq = score(p), score(qv)
        for _ in range(24):
            if fp < fq:
                c, qv, fq = qv, p, fp
                p = c-phi*(c-a); fp = score(p)
            else:
                a, p, fp = p, qv, fq
                qv = a+phi*(c-a); fq = score(qv)
        s = (a+c)/2
        val = score(s)
        if best is None or val < best[0]:
            best = (val, high, s)
    _, high, sill = best
    return {'sluice_high_mode': bool(high), 'sill_height_m': float(sill), 'eta_full': _simulate(high, sill)}
