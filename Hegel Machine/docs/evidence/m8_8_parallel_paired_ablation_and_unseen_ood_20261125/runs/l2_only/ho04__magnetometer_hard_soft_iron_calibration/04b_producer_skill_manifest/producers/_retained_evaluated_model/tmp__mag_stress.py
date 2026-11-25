import pickle
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/holdout_task')
from mag_calibrator import calibrate_and_estimate_heading


def evaluate(session, truth):
    out = calibrate_and_estimate_heading(session)
    e = (out['heading_deg'] - truth['gt_heading_deg'] + 180) % 360 - 180
    assert out['heading_deg'].shape == e.shape
    assert np.all((-180 < out['heading_deg']) & (out['heading_deg'] <= 180))
    S = out['S_soft']
    assert np.allclose(S, S.T) and min(np.linalg.eigvalsh(S)) > 0
    assert abs(np.linalg.det(S)-1) < 1e-10
    assert out['B_ned_uT'][0] > 0 and out['B_ned_uT'][1] == 0
    yaw = truth['gt_heading_deg']
    sectors = [e[(yaw >= lo) & (yaw < lo+90)] for lo in (-180,-90,0,90)]
    rmse = lambda a: np.sqrt(np.mean(a*a))
    return rmse(e), max(rmse(a) for a in sectors if len(a)), np.linalg.norm(out['b_hard']-truth['b_hard'])


def synthetic(seed, n, tilt, h, d, sigma, condition, narrow_spikes):
    rng = np.random.default_rng(seed)
    q = Rotation.random(random_state=rng).as_matrix()
    S = (q * np.array([1/np.sqrt(condition), 1., np.sqrt(condition)])) @ q.T
    b = rng.uniform(-80,80,3)
    field = np.array([h,0,d])
    def sample(count, limit, noise):
        r, p = rng.uniform(-limit,limit,(2,count))
        y = rng.uniform(-np.pi,np.pi,count)
        R = Rotation.from_euler('ZYX', np.column_stack((y,p,r))).as_matrix()
        raw = (field @ R) @ S + b + rng.normal(0,noise,(count,3))
        return raw, np.column_stack((r,p)), np.rad2deg(y)
    cal, cal_rp, _ = sample(n, tilt, sigma)
    ind = rng.choice(n, round(.04*n), replace=False)
    directions = rng.normal(size=(len(ind),3))
    if narrow_spikes:
        directions[:,1:] *= .04
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    cal[ind] += directions*rng.uniform(12,54,(len(ind),1))
    test, test_rp, yaw = sample(240, .45, .065)
    return dict(calib_m_raw=cal, calib_roll_pitch_rad=cal_rp, test_m_raw=test, test_roll_pitch_rad=test_rp), dict(b_hard=b, gt_heading_deg=yaw)

if __name__ == '__main__':
    data = pickle.load(open('/holdout_task/public_mag.pkl','rb'))
    s,g = data['session'],data['gt']
    keys = ('calib_m_raw','calib_roll_pitch_rad')
    x = np.arange(len(s[keys[0]]))
    print('columns: heading RMSE deg, worst yaw sector RMSE deg, bias L2 uT')
    for label, keep in [('full',x>=0)] + [(f'block-{i}',x//84!=i) for i in range(5)] + [('inner tilt',np.max(np.abs(s[keys[1]]),axis=1)<.44)]:
        z = dict(s, **{k:s[k][keep] for k in keys})
        print(label, int(keep.sum()), np.round(evaluate(z,g),5))
    configs = {'public-like':(420,.65,23.5,42,.09,1.35,False), 'narrow tilt':(250,.31,23,42,.12,1.8,False), 'small horizontal':(320,.49,16,49,.13,2.2,True), 'opposite inclination':(240,.45,24,-44,.12,2.1,True), 'near horizontal':(240,.42,44,4,.12,2.1,True)}
    for label, cfg in configs.items():
        a = np.array([evaluate(*synthetic(seed,*cfg)) for seed in range(12)])
        print(label, 'max', np.round(a.max(axis=0),5), 'worst seeds', a.argmax(axis=0), 'misses', np.sum(a[:,[0,2]]>[.45,.35],axis=0))
