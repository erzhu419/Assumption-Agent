import numpy as np
from scipy.optimize import least_squares

def _level(v, rp):
    phi, theta = rp[:, 0], rp[:, 1]
    c, s = np.cos(phi), np.sin(phi)
    x = v[:, 0]
    y = c * v[:, 1] - s * v[:, 2]
    z = s * v[:, 1] + c * v[:, 2]
    ct, st = np.cos(theta), np.sin(theta)
    return np.column_stack((ct * x + st * z, y, -st * x + ct * z))

def calibrate_and_estimate_heading(session: dict) -> dict:
    m = np.asarray(session['calib_m_raw'], dtype=float)
    rp = np.asarray(session['calib_roll_pitch_rad'], dtype=float)
    def unpack(p):
        b = p[:3]
        L = np.array([[np.exp(p[3]), 0., 0.], [p[6], np.exp(p[4]), 0.], [p[7], p[8], np.exp(p[5])]])
        C = L @ L.T
        C /= np.linalg.det(C) ** (1. / 3.)
        return b, C
    def residual(p):
        b, C = unpack(p)
        v = (m - b) @ C.T
        leveled = _level(v, rp)
        norm = np.linalg.norm(v, axis=1)
        return np.r_[norm - np.median(norm), leveled[:, 2] - np.median(leveled[:, 2])]
    p = np.zeros(9)
    p[:3] = np.median(m, axis=0)
    p[3:6] = np.log(1. / 50.)
    result = least_squares(residual, p, loss='soft_l1', f_scale=0.15, max_nfev=3000)
    result = least_squares(residual, result.x, loss='soft_l1', f_scale=0.05, max_nfev=3000)
    b, C = unpack(result.x)
    v = (np.asarray(session['test_m_raw'], dtype=float) - b) @ C.T
    leveled = _level(v, np.asarray(session['test_roll_pitch_rad'], dtype=float))
    heading = (np.degrees(np.arctan2(-leveled[:, 1], leveled[:, 0])) + 180.) % 360. - 180.
    heading = np.where(heading <= -180., 180., heading)
    return {'heading_deg': heading, 'b_hard': b, 'S_soft': np.linalg.inv(C)}
