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

def forward(angles, heading, H, D, S, b):
    u, v, w = _level_axes(angles)
    return (H*np.cos(heading)[:,None]*u - H*np.sin(heading)[:,None]*v + D*w) @ S + b


print('columns: circular RMSE, worst test quartile RMSE, bias L2')
for count, coverage, ratio, cond, spike, sigma in [(420,.25,.30,2.8,150,.085), (220,.50,.30,2.8,100,.085), (220,.25,.45,2.8,150,.085), (420,.5,.30,2.8,100,.10)]:
    vals, oracle, clean = [], [], []
    for seed in range(40):
        rng = np.random.default_rng(1700 + seed)
        q, _ = np.linalg.qr(rng.normal(size=(3,3)))
        S = (q * np.array([cond**.5, 1., cond**-.5])) @ q.T
        b = rng.uniform(-75,75,3)
        ca = s['calib_roll_pitch_rad'][rng.choice(len(s['calib_roll_pitch_rad']), count, replace=False)] * coverage
        ta = s['test_roll_pitch_rad']
        cy = rng.uniform(-np.pi,np.pi,count)
        ty = np.linspace(-np.pi,np.pi,len(ta))
        m = forward(ca, cy, 50*ratio, 50,S,b) + rng.normal(0,sigma,(count,3))
        ii = rng.choice(count,round(.04*count),replace=False)
        dipole = rng.normal(size=3); dipole /= np.linalg.norm(dipole)
        m[ii] += rng.uniform(.35,1,len(ii))[:,None]*spike*dipole
        tm = forward(ta,ty,50*ratio,50,S,b) + rng.normal(0,sigma,(len(ta),3))
        vals.append(score(dict(calib_m_raw=m,calib_roll_pitch_rad=ca,test_m_raw=tm,test_roll_pitch_rad=ta),dict(b_hard=b,gt_heading_deg=np.rad2deg(ty))))
        clean_rows = np.ones(count, dtype=bool)
        clean_rows[ii] = False
        clean.append(score(dict(calib_m_raw=m[clean_rows],calib_roll_pitch_rad=ca[clean_rows],test_m_raw=tm,test_roll_pitch_rad=ta),dict(b_hard=b,gt_heading_deg=np.rad2deg(ty))))
        z = (tm-b) @ np.linalg.inv(S)
        u,v,_ = _level_axes(ta)
        d = (np.rad2deg(np.arctan2(-np.sum(z*v,axis=1),np.sum(z*u,axis=1))-ty)+180)%360-180
        oracle.append(np.sqrt(np.mean(d*d)))
    vals, clean, oracle = np.array(vals), np.array(clean), np.array(oracle)
    failing = np.flatnonzero((vals[:,0] > .45) | (vals[:,2] > .35))
    j = vals[:,2].argmax()
    print((count,coverage,ratio,cond,spike,sigma), 'bias worst actual/clean', vals[j,2], clean[j,2], 'max paired delta in bias', np.max(vals[:,2]-clean[:,2]))
    print('failed heading with/without spikes and with true calibration:', [(int(i)+1700, *np.round([vals[i,0],clean[i,0],oracle[i]],4)) for i in failing if vals[i,0] > .45])
