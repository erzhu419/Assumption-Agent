import numpy as np
from mag_validate import m, s, g, errors, report, subset, synthetic, CASES

baseline = m._fit

def exact_residual(p, x, axes):
    # Project onto the raw-space ellipse at each known tilt. The two
    # horizontal basis vectors need not be orthogonal in raw coordinates.
    w = m._correction(p)
    soft = np.linalg.inv(w)
    level = np.einsum('nij,nj->ni', axes, (x - p[5:8]) @ w)
    yaw = np.arctan2(level[:, 1], level[:, 0])
    u, v = (p[8] * (axes[:, :2] @ soft)).transpose(1, 0, 2)
    offset = p[5:8] + p[9] * (axes[:, 2] @ soft) - x
    for _ in range(5):
        c, t = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        radial = c*u + t*v
        tangent = -t*u + c*v
        residual = offset + radial
        slope = np.einsum('ij,ij->i', residual, tangent)
        curvature = np.einsum('ij,ij->i', tangent, tangent) - np.einsum('ij,ij->i', residual, radial)
        yaw -= np.clip(slope / np.maximum(curvature, 1e-12), -.25, .25)
    return offset + np.cos(yaw)[:, None]*u + np.sin(yaw)[:, None]*v

def exact_fit(x, axes):
    p = baseline(x, axes)
    distance = np.linalg.norm(m._residual(p, x, axes), axis=1)
    keep = distance <= 4 * np.median(distance) / np.sqrt(2*np.log(2))
    return m.least_squares(lambda q: exact_residual(q, x[keep], axes[keep]).ravel(), p, max_nfev=25).x

rp = s['calib_roll_pitch_rad']
axes = m._level_axes(rp)
level = np.einsum('nij,nj->ni', axes, (s['calib_m_raw']-g['b_hard']) @ np.linalg.inv(g['S_soft']))
sector = np.floor((np.arctan2(-level[:,1],level[:,0])+np.pi)*8/(2*np.pi)).astype(int)%8
quadrant = (rp[:,0]>0).astype(int) + 2*(rp[:,1]>0)
public = [(s,g)] + [(subset(k!=i),g) for k in (sector,quadrant,np.arange(len(rp))//70) for i in np.unique(k)]
cases = [('public + jackknife', public)]
for name, kw in CASES:
    ids = range(44,172) if name == 'combined stress' else range(20)
    cases.append((name, [synthetic(i, **kw) for i in ids]))
for name, pairs in cases:
    for label, fit in [('whitened', baseline), ('exact', exact_fit)]:
        m._fit = fit
        a = np.array([errors(m.calibrate_and_estimate_heading(sess), truth) for sess, truth in pairs])
        report(f'{name} {label}', a)
        print('target exceedances (heading, bias):', (a > [.45,.35]).sum(axis=0), flush=True)
m._fit = baseline
