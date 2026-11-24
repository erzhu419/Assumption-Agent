import numpy as np
from mag_validate import m, s, g, errors, report, subset

raw = s['calib_m_raw']
rp = s['calib_roll_pitch_rad']
axes = m._level_axes(rp)
level = np.einsum('nij,nj->ni', axes, (raw-g['b_hard']) @ np.linalg.inv(g['S_soft']))
yaw = np.arctan2(-level[:,1],level[:,0])
sector = np.floor((yaw+np.pi)*8/(2*np.pi)).astype(int)%8
tilt = np.linalg.norm(rp, axis=1)
joint_fit = m._fit

# Match sample counts before comparing low tilt with broad tilt. Omit one
# yaw sector from both arms so dense coverage cannot mask a weak direction.
for fraction in (.50,.25):
    groups = {k: [] for k in ('low joint','broad joint','low vertical','broad vertical')}
    for hold in range(8):
        ids = np.flatnonzero(sector != hold)
        ranked = ids[np.argsort(tilt[ids])]
        n = round(fraction * len(ids))
        picks = {'low': ranked[:n], 'broad': ranked[((np.arange(n)+.5)*len(ids)/n).astype(int)]}
        for arm, keep in picks.items():
            fold = subset(keep)
            for method, fit in [('joint',joint_fit),('vertical',m._seed)]:
                m._fit = fit
                groups[f'{arm} {method}'].append(errors(m.calibrate_and_estimate_heading(fold), g))
    m._fit = joint_fit
    for name, values in groups.items():
        report(f'{fraction:.0%} {name}', values)
