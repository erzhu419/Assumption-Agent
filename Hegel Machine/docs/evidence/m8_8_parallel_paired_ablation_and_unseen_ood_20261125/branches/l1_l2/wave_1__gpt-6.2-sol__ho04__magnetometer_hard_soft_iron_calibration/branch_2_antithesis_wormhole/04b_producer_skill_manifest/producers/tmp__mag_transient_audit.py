import pickle
import sys
import numpy as np

sys.path.insert(0, '/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading, _level_axes

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
raw, angles = s['calib_m_raw'], s['calib_roll_pitch_rad']
ref = (raw-gt['b_hard']) @ np.linalg.inv(gt['S_soft'])
u, v, w = _level_axes(angles)
h = np.hypot(np.sum(u*ref, axis=1), np.sum(v*ref, axis=1))
d = np.sum(w*ref, axis=1)
r = np.column_stack((h-np.median(h), d-np.median(d)))
scale = 1.4826*np.median(np.abs(r), axis=0)
keep = np.linalg.norm(r/scale, axis=1) <= 4.3
raw, angles = raw[keep], angles[keep]

def score(m):
    o = calibrate_and_estimate_heading(dict(s, calib_m_raw=m, calib_roll_pitch_rad=angles))
    z = (o['heading_deg'] - gt['gt_heading_deg'] + 180) % 360 - 180
    return [np.sqrt(np.mean(z*z)), np.linalg.norm(o['b_hard']-gt['b_hard'])]

print('Clean measured calibration rows:', len(raw), 'baseline RMSE / bias:', np.round(score(raw), 5))
rng = np.random.default_rng(704)
for amplitude in (0.25, 2, 20, 120):
    scores = []
    count = round(.04*len(raw))
    for start in np.linspace(0, len(raw)-count, 7, dtype=int):
        for _ in range(4):
            dipole = rng.normal(size=3)
            dipole /= np.linalg.norm(dipole)
            m = raw.copy()
            m[start:start+count] += amplitude*dipole
            scores.append(score(m))
    scores = np.array(scores)
    print('Burst amplitude', amplitude, 'worst RMSE / bias', scores.max(axis=0).round(5), 'target misses', np.count_nonzero((scores[:,0]>.45)|(scores[:,1]>.35)))
