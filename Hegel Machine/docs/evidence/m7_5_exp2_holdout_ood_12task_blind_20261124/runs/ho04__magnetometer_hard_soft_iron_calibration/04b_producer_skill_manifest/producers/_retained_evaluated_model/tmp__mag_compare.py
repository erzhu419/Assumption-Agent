import numpy as np
from mag_validate import m, s, g, errors, report, subset, synthetic, CASES

baseline = m._fit

def refit(rule):
    def fit(x, axes):
        p = baseline(x, axes)
        for _ in range(4):
            distance = np.linalg.norm(m._residual(p, x, axes), axis=1)
            sigma = max(np.median(distance) / np.sqrt(2 * np.log(2)), 1e-6)
            if rule == 'trim':
                weight = (distance <= 4 * sigma).astype(float)
            else:
                weight = np.maximum(0., 1 - (distance / (6 * sigma))**2)
            q = m.least_squares(lambda q: (m._residual(q, x, axes) * weight[:, None]).ravel(), p, max_nfev=25).x
            if np.linalg.norm(q-p) < 1e-7:
                return q
            p = q
        return p
    return fit

rp = s['calib_roll_pitch_rad']
axes = m._level_axes(rp)
level = np.einsum('nij,nj->ni', axes, (s['calib_m_raw']-g['b_hard']) @ np.linalg.inv(g['S_soft']))
yaw = np.arctan2(-level[:,1],level[:,0])
sectors = np.floor((yaw+np.pi)*8/(2*np.pi)).astype(int)%8
quadrants = (rp[:,0]>0).astype(int) + 2*(rp[:,1]>0)
public = [(s,g)]+[(subset(group!=i),g) for group in (sectors,quadrants,np.arange(len(rp))//70) for i in np.unique(group)]
cases = [('public + calibration jackknife',public)]
for name, kw in CASES:
    cases.append((name, [synthetic(i, **kw) for i in range(20, 44)]))
for name, pairs in cases:
    for label, fn in [('cauchy',baseline),('trim',refit('trim')),('tukey',refit('tukey'))]:
        m._fit = fn
        report(f'{name} {label}',[errors(m.calibrate_and_estimate_heading(sess),truth) for sess,truth in pairs])
m._fit = baseline
