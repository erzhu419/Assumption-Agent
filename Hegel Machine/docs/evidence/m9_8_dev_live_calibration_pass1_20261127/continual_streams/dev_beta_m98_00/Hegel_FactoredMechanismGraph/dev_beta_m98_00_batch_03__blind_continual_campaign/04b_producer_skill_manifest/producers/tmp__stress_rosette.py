import sys, pickle, time
sys.path.insert(0, '/holdout_task')
import numpy as np
import upstream_transducer as u
from batch_solver import solve_batch
base = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
bank = base['sensor_calibration_sweeps']['single_bank_sweeps']
cal = base['sensor_calibration_sweeps']; x, y = cal['ref_rosette'], cal['raw_rosette']
a = u._basis(x)[:, :5]
pub = np.empty_like(y)
for i in range(len(x)):
    use = np.arange(len(x)) != i
    c = u._robust_linear(a[use], y[use])
    pub[i] = np.linalg.solve(c[1:].T, y[i]-c[0]) - x[i]
print('Public LOOCV max abs axis/mixed:', *(np.max(np.abs(pub[s])) for s in (slice(16), slice(16,None))), flush=True)
for seed in range(3):
    rng = np.random.default_rng(seed + 738)
    z = rng.uniform(-.8, .8, (96,4)); z[64:] = rng.uniform(-1.3,1.65,(32,4))
    m = np.diag([1.29,.78,1.15,.83])+rng.uniform(-.09,.09,(4,4))
    b = np.array([.08,-.07,.06,-.09]); d = rng.normal(0,.032,(10,4)); e = rng.normal(0,.014,(20,4))
    q = np.array([[.72,-.25,.45,.26],[.47,.9,-.23,.2],[-.22,-.12,.74,.58],[.2,.38,-.15,.78]])
    for family in ('affine', 'quadratic', 'cubic', 'rational', 'mixed'):
        def f(v):
            a = u._basis(v); main = v@m+b; quad = a[:,5:15]@d; cubic = a[:,15:]@e
            if family == 'affine': return main
            if family == 'quadratic': return main+quad
            if family == 'cubic': return main+quad+cubic
            if family == 'rational': return main+quad/(1+(v@q)**2)
            return main+np.column_stack((quad[:,0]/(1+(v@q[:,0])**2), quad[:,1]+cubic[:,1], quad[:,2], np.zeros(len(v))))
        raw = f(x)+rng.normal(0,1e-5,x.shape)
        idx = rng.choice(len(x), 5 if seed < 2 else 9, replace=False)
        raw[idx] += rng.uniform(.5,1.7,(len(idx),4))*rng.choice([-1,1],(len(idx),4))
        obs = f(z[:24]).ravel(); pulses = [{'y_pulse_raw':f(z[24:48]).ravel()}, {'y_pulse_raw':f(z[48:]).reshape(-1,4)}, {'y_pulse_raw':None}]
        study = dict(base, y_obs_raw=obs, sensor_calibration_sweeps=dict(cal, raw_rosette=raw), reference_tracer_pulses=pulses)
        t = time.monotonic(); got = u.calibrate_upstream(study); dt = time.monotonic()-t
        pp = got['reference_tracer_pulses']; err = np.vstack((got['y_obs_raw'].reshape(-1,4), pp[0]['y_pulse_raw'].reshape(-1,4), pp[1]['y_pulse_raw']))-z
        assert pp[1]['y_pulse_raw'].shape == pulses[1]['y_pulse_raw'].shape and pp[2]['y_pulse_raw'] is None
        assert study['y_obs_raw'] is obs and study['sensor_calibration_sweeps']['raw_rosette'] is raw
        sw = got['sensor_calibration_sweeps']; assert np.array_equal(sw['raw_rosette'], x) and sw['raw_rosette'] is not x
        assert np.array_equal(sw['single_bank_sweeps']['raw_readout'], bank['ref_signal']) and sw['single_bank_sweeps']['raw_readout'] is not bank['ref_signal']
        print(seed,family,'EMI',len(idx),'worst core/edge',*(f'{np.max(np.abs(err[s])):.3g}' for s in (slice(64),slice(64,None))),'RMSE',f'{np.sqrt(np.mean(err**2)):.3g}','sec',round(dt,2),flush=True)
