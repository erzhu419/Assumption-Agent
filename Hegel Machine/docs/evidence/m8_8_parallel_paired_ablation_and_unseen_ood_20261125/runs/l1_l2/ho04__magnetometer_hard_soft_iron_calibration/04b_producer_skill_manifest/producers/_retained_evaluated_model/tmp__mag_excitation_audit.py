import pickle
import sys
import numpy as np

sys.path.insert(0, '/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading, _level_axes

with open('/holdout_task/public_mag.pkl', 'rb') as f:
    data = pickle.load(f)
s, gt = data['session'], data['gt']
angles = s['calib_roll_pitch_rad']
raw = s['calib_m_raw']
n = len(raw)
# Select calibration rows using only acquisition order and measured roll/pitch.
# None of these selections uses the labeled test headings or calibration truth.
order = np.arange(n)
roll, pitch = angles.T
tilt = np.arccos(np.clip(np.cos(roll)*np.cos(pitch), -1, 1))
strata = {'acquisition': order, 'roll': np.argsort(roll),
          'pitch': np.argsort(pitch), 'absolute tilt': np.argsort(tilt)}

u, v, _ = _level_axes(s['test_roll_pitch_rad'])
z = (s['test_m_raw']-gt['b_hard']) @ np.linalg.inv(gt['S_soft'])
oracle = np.rad2deg(np.arctan2(-np.sum(z*v, axis=1), np.sum(z*u, axis=1)))
# Define held-out test slices from measured tilt, not fitted residuals.
test_order = np.argsort(np.linalg.norm(s['test_roll_pitch_rad'], axis=1))
test_groups = np.array_split(test_order, 4)
def error(h):
    e = (h-gt['gt_heading_deg']+180)%360-180
    return np.sqrt(np.mean(e**2)), max(np.sqrt(np.mean(e[k]**2)) for k in test_groups)
print('oracle: overall and worst test tilt-quartile RMSE', np.round(error(oracle), 5))
for name, idx in strata.items():
    for fraction in (0.50, 0.25):
        size = round(fraction*n)
        records = []
        for start in np.unique(np.linspace(0,n-size,5,dtype=int)):
            k = idx[start:start+size]
            out = calibrate_and_estimate_heading(dict(s, calib_m_raw=raw[k], calib_roll_pitch_rad=angles[k]))
            bias = np.linalg.norm(out['b_hard']-gt['b_hard'])
            records.append([*error(out['heading_deg']), bias])
        records = np.array(records)
        print(name, fraction, 'max RMSE / test tilt-quartile RMSE / bias:', records.max(axis=0).round(5), 'target misses:', int(np.sum((records[:,0]>.45)|(records[:,2]>.35))))
