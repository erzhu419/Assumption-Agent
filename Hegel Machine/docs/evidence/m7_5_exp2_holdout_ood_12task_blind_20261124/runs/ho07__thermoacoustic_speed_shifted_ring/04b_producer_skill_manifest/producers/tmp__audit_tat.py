import importlib.util
import pickle
import numpy as np

path = '/holdout_task/thermoacoustic_recon.py'
spec = importlib.util.spec_from_file_location('thermoacoustic_recon', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_tat.pkl', 'rb') as f:
    cases = pickle.load(f)

for i, e in enumerate(cases):
    s, gt = e['scan'], e['gt']
    y, xy, pin = map(np.asarray, (s['signals'], s['sensor_xy_mm'], s['calib_pin_xy_mm']))
    factor = s['fs_hz'] / 1000
    dp = np.linalg.norm(xy-pin, axis=1)
    angle_order = np.argsort(np.arctan2(xy[:,1], xy[:,0]))
    tests = [('sector', angle_order[(np.arange(4)+4*k) % len(xy)]) for k in range(4)]
    tests += [('interleaved', angle_order[k::2]) for k in range(2)]
    joint_err, direct_err, wave_rms, echo_snr, polarity_conflicts = [], [], [], [], 0
    time = np.arange(len(y))[None,:,None]
    q = np.asarray(gt['absorbers_xy_mm'])
    for label, hold in tests:
        keep = ~np.isin(np.arange(len(xy)), hold)
        args = (y[:,keep], xy[keep], dp[keep], factor)
        seed = m._seed(*args, 4.4)
        joint = m._fit(*args, seed)
        direct = m._fit(*args, seed, pin_speed=seed[4])
        for dst, x in [(joint_err, joint), (direct_err, direct)]:
            z = x[:4].reshape(2,2)
            d = np.linalg.norm(z[:,None]-q,axis=-1)
            dst.append([abs(x[4]-gt['sound_speed_m_per_s']),max(d.min(0).max(), d.min(1).max())])
        z = joint[:4].reshape(2,2)
        dt = np.linalg.norm(xy[hold,None]-z[None],axis=-1)
        scale = factor / joint[4]
        base = m._pulse(time - np.column_stack((dt,dp[hold]))[:,None,:]*scale, joint[5])
        test = y[:,hold].T
        r, amps = m._project(base,test)
        polarity_conflicts += int(np.count_nonzero(np.any(amps[:,:2]*amps[:,2:]<0,axis=1)))
        wave_rms.append(np.sqrt(np.mean(r*r)))
        # Remove the fitted direct+pin subspace before checking the two-leg basis.
        echo_t = (dp[hold,None]+np.linalg.norm(z-pin,axis=1))*scale
        echo = m._pulse(time-echo_t[:,None,:],joint[5])
        for j in range(2):
            v = m._project(base,echo[:,:,j])[0]
            score = np.sum(v*r,axis=1) / np.maximum(np.linalg.norm(v,axis=1),1e-12)
            echo_snr.append(np.sum(score*np.sign(amps[:,2])) / (np.sqrt(len(hold))*wave_rms[-1]))
    print(i, 'joint worst [m/s, mm]',np.max(joint_err,axis=0),'fixed-pin-timing worst',np.max(direct_err,axis=0),'held-out RMS range',np.round([min(wave_rms),max(wave_rms)],5),'abs coherent two-leg noise units max',round(max(abs(np.array(echo_snr))),3),'polarity conflicts',polarity_conflicts,flush=True)
    assert np.all(np.asarray(joint_err)<=[2.5,.85])
