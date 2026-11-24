import numpy as np
from scipy.optimize import least_squares

def _level(v, rp, roll_sign=1.0, pitch_sign=1.0):
    p = roll_sign * rp[:, 0]
    t = pitch_sign * rp[:, 1]
    cp, sp, ct, st = np.cos(p), np.sin(p), np.cos(t), np.sin(t)
    return np.column_stack((ct*v[:,0] + st*sp*v[:,1] + st*cp*v[:,2],
                            cp*v[:,1] - sp*v[:,2],
                            -st*v[:,0] + ct*sp*v[:,1] + ct*cp*v[:,2]))

def _fit(raw, rp):
    raw, rp = np.asarray(raw, float), np.asarray(rp, float)
    def matrix(q):
        h = np.array([[q[3],q[6],q[7]],[q[6],q[4],q[8]],[q[7],q[8],q[5]]])
        w,u = np.linalg.eigh(h)
        a = (u*np.exp(w))@u.T
        return a / np.linalg.det(a)**(1/3)
    def residual(q):
        v = _level((raw-q[:3])@matrix(q).T, rp)
        horiz = np.hypot(v[:,0], v[:,1])
        return np.r_[v[:,2]-np.median(v[:,2]), horiz-np.median(horiz)]
    starts = [np.r_[np.median(raw, axis=0), np.zeros(6)]]
    starts += [np.r_[np.median(raw, axis=0)+np.random.default_rng(i).normal(0,5,3),np.zeros(6)] for i in range(4)]
    best = None
    for q0 in starts:
        opt = least_squares(residual, q0, loss='soft_l1', f_scale=.5, max_nfev=1500)
        score = np.median(np.abs(residual(opt.x)))
        if best is None or score < best[0]: best = (score, opt.x)
    return best[1][:3], matrix(best[1])

def calibrate_and_estimate_heading(session: dict) -> dict:
    b, a = _fit(session['calib_m_raw'], session['calib_roll_pitch_rad'])
    corrected = (np.asarray(session['test_m_raw'], float)-b)@a.T
    leveled = _level(corrected, np.asarray(session['test_roll_pitch_rad'], float))
    heading = (np.degrees(np.arctan2(-leveled[:,1], leveled[:,0]))+180.)%360.-180.
    heading = np.where(heading <= -180., 180., heading)
    return {'S_soft_hat': np.linalg.inv(a), 'b_hard_hat': b, 'heading_deg': heading}
