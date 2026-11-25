import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading, _level_axes

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    p = pickle.load(f)
s, g = p['session'], p['gt']

def score(session, truth):
    o = calibrate_and_estimate_heading(session)
    A = o['S_soft']
    assert np.allclose(A, A.T) and np.linalg.eigvalsh(A).min() > 0
    assert np.isclose(np.linalg.det(A), 1, atol=1e-10)
    yaw = o['heading_deg']
    assert yaw.shape == truth['gt_heading_deg'].shape and np.all((yaw > -180) & (yaw <= 180))
    e = (yaw - truth['gt_heading_deg'] + 180) % 360 - 180
    return np.r_[np.sqrt(np.mean(e**2)), max(np.sqrt(np.mean(x**2)) for x in np.array_split(e, 4)), np.linalg.norm(o['b_hard'] - truth['b_hard'])]

print('columns: heading RMSE, worst test quartile RMSE, bias L2')
yr = s['calib_roll_pitch_rad']
ref = (s['calib_m_raw'] - g['b_hard']) @ np.linalg.inv(g['S_soft'])
ax = _level_axes(yr)
yaw = np.arctan2(-np.sum(ref*ax[1], axis=1), np.sum(ref*ax[0], axis=1))
lev = np.sum(ref*ax[2], axis=1)
h = np.sqrt(np.maximum(0, np.sum(ref**2, axis=1) - lev**2))
print('public levels H D, radial and vertical MAD:', np.median(h), np.median(lev), 1.4826*np.median(abs(h-np.median(h))), 1.4826*np.median(abs(lev-np.median(lev))))
for name, dim in [('order', np.arange(len(yaw))), ('estimated yaw', yaw), ('roll', yr[:,0]), ('pitch', yr[:,1])]:
    vals = []
    for block in np.array_split(np.argsort(dim), 4):
        keep = ~np.isin(np.arange(len(yaw)), block)
        ss = dict(s, calib_m_raw=s['calib_m_raw'][keep], calib_roll_pitch_rad=yr[keep])
        vals.append(score(ss, g))
    print(name, 'worst', np.max(vals, axis=0).round(4))

def forward(angles, heading, H, D, S, b):
    u, v, w = _level_axes(angles)
    return (H*np.cos(heading)[:,None]*u - H*np.sin(heading)[:,None]*v + D*w) @ S + b

for coverage, ratio, cond, spike in [(1, 0.45, 1.5, 75), (.5, .45, 2.8, 100), (.25, .45, 2.8, 150), (.5, .30, 2.8, 100)]:
    vals = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(3,3)))
        S = (q * np.array([cond**.5, 1., cond**-.5])) @ q.T
        b = rng.uniform(-75, 75, 3)
        cy = rng.uniform(-np.pi, np.pi, len(yr))
        ty = np.linspace(-np.pi, np.pi, 120)
        ca = yr * coverage
        ta = s['test_roll_pitch_rad']
        m = forward(ca, cy, 50*ratio, 50, S, b) + rng.normal(0, .05, (len(ca),3))
        ii = rng.choice(len(ca), round(.04*len(ca)), replace=False)
        dipole = rng.normal(size=3); dipole /= np.linalg.norm(dipole)
        m[ii] += rng.uniform(.35, 1, len(ii))[:,None] * spike * dipole
        tm = forward(ta, ty, 50*ratio, 50, S, b) + rng.normal(0,.05,(len(ta),3))
        vals.append(score(dict(calib_m_raw=m, calib_roll_pitch_rad=ca, test_m_raw=tm, test_roll_pitch_rad=ta), dict(b_hard=b, gt_heading_deg=np.rad2deg(ty))))
    print('stress', (coverage, ratio, cond, spike), 'worst', np.max(vals, axis=0).round(4))
