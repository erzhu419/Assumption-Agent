import pickle
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares

def ray_integral(width, source, receiver, x, z, radius):
    dz = receiver - source
    length = np.hypot(width, dz)
    ux, uz = width / length, dz / length
    parallel = ux * x + uz * (z - source)
    normal = -uz * x + ux * (z - source)
    rr = np.sqrt(2) * radius
    return np.sqrt(np.pi / 2) * radius * np.exp(-0.5 * (normal / radius)**2) * (erf((length - parallel) / rr) + erf(parallel / rr))

if __name__ == '__main__':
    with open('/holdout_task/public_surveys.pkl', 'rb') as f:
        data = pickle.load(f)
    for k, row in enumerate(data):
        s, gt = row['survey'], row['gt']
        src = s['z_src_m'][:, None]; rcv = s['z_rcv_m'][None, :]
        dz = rcv - src; w = s['W_m']; l = np.hypot(w, dz)
        y = 1000 * (s['T_obs_s'] - gt['rcv_clock_bias_s']).ravel()
        x, z = gt['plume_xz_m']
        p = ray_integral(w, src, rcv, x, z, s['R_plume_m'])
        a = np.column_stack([l.ravel(), (-dz**2/l).ravel(), p.ravel()])
        b = least_squares(lambda b: a@b-y, [0.4, 0.03, 0.1], jac=lambda b: a.copy(), loss='soft_l1', f_scale=0.3).x
        keep = abs(y-a@b) < 3
        b = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
        e = (y-a@b).reshape(l.shape)
        print('survey', k, 'shape', l.shape, 's0, eps, amplitude (s/m, dimensionless, s/m):', [b[0]/1000,b[1]/b[0],b[2]/1000])
        print('inlier RMS/max (ms):', np.sqrt(np.mean(e.ravel()[keep]**2)), np.max(abs(e.ravel()[keep])), 'outlier residuals (ms):', np.round(e.ravel()[~keep], 3))
        print('inlier row-median range (ms):', np.round([np.nanmin(np.nanmedian(np.where(abs(e)<3,e,np.nan),axis=1)), np.nanmax(np.nanmedian(np.where(abs(e)<3,e,np.nan),axis=1))], 4), 'ground truth clock sum (s):', np.sum(gt['rcv_clock_bias_s']))
