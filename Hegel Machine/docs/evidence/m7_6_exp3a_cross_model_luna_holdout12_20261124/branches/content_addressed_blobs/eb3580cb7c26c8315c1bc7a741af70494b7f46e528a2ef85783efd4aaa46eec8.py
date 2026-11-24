import numpy as np
from scipy.optimize import minimize_scalar

_G = 9.81

def _forward(high, sill, n=80, dx=10.0, dt=0.5, steps=191):
    x = (np.arange(n) + 0.5) * dx
    b = sill * np.exp(-0.5 * ((x - 400.0) / 45.0) ** 2)
    eta = np.full(n, 2.0)
    eta[:8] = 2.65 if high else 1.45
    h = np.maximum(eta - b, 1e-9)
    q = np.zeros(n)
    out = np.empty((steps, n))
    out[0] = h + b
    for k in range(1, steps):
        he = np.r_[h[0], h, h[-1]]
        qe = np.r_[-q[0], q, -q[-1]]
        be = np.r_[b[0], b, b[-1]]
        hl, hr = he[:-1], he[1:]
        ql, qr = qe[:-1], qe[1:]
        bl, br = be[:-1], be[1:]
        ul, ur = ql / np.maximum(hl, 1e-9), qr / np.maximum(hr, 1e-9)
        bs = np.maximum(bl, br)
        hls = np.maximum(0.0, hl + bl - bs)
        hrs = np.maximum(0.0, hr + br - bs)
        fl0, fr0 = hls * ul, hrs * ur
        fl1 = hls * ul**2 + 0.5 * _G * hls**2
        fr1 = hrs * ur**2 + 0.5 * _G * hrs**2
        speed = np.maximum(np.maximum(np.abs(ul) + np.sqrt(_G*hls), np.abs(ur) + np.sqrt(_G*hrs)), 1e-6)
        f0 = 0.5*(fl0+fr0) - 0.5*speed*(hrs-hls)
        f1 = 0.5*(fl1+fr1) - 0.5*speed*(hrs*ur-hls*ul)
        # cell i's right-face reconstructed depth and left-face reconstructed depth
        hm = np.maximum(0.0, h + b - np.maximum(b, np.r_[b[1:], b[-1]]))
        hp = np.maximum(0.0, h + b - np.maximum(np.r_[b[0], b[:-1]], b))
        hn = h - dt/dx*(f0[1:]-f0[:-1])
        qn = q - dt/dx*(f1[1:]-f1[:-1]) + dt/dx*0.5*_G*(hm**2-hp**2)
        h, q = np.maximum(hn, 1e-9), qn
        out[k] = h+b
    return out

def invert_canal_regime(obs_bundle: dict) -> dict:
    obs = np.asarray(obs_bundle['eta_obs'], dtype=float)
    indices = np.asarray(obs_bundle['obs_step_indices'], dtype=int)
    gauges = np.asarray(obs_bundle.get('gauge_cells', [48, 62, 74]), dtype=int)
    n = int(obs_bundle.get('N', 80)); dx = float(obs_bundle.get('dx', 10.0))
    dt = float(obs_bundle.get('dt', 0.5)); steps = int(obs_bundle.get('T_steps', int(indices.max())+1))
    best = None
    for high in (False, True):
        def loss(s):
            pred = _forward(high, s, n, dx, dt, steps)[indices[:, None], gauges[None, :]]
            return float(np.mean((pred-obs)**2))
        fit = minimize_scalar(loss, bounds=(0.0, 0.75), method='bounded', options={'xatol': 1e-7})
        for s in (0.0, 0.75, fit.x):
            score = loss(s)
            if best is None or score < best[0]: best = (score, high, float(s))
    _, high, sill = best
    return {'sluice_high_mode': bool(high), 'sill_height_m': sill,
            'eta_full': _forward(high, sill, n, dx, dt, steps)}
