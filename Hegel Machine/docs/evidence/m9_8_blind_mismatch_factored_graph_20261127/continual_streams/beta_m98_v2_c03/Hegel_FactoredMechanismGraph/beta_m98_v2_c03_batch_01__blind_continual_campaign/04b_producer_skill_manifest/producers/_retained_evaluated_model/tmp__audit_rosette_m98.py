import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    s = item['study']; c = s['sensor_calibration_sweeps']
    x = np.asarray(c['ref_rosette']); y = np.asarray(c['raw_rosette'])
    S, _ = solver._forward(s)
    pulses = [(np.asarray(S@p['w_pulse']).reshape(-1, 4), np.asarray(p['y_pulse_raw']).reshape(-1, 4)) for p in item.get('campaign_diagnostics') or []]
    banks = c['single_bank_sweeps']
    print('single-bank layouts', {str(k): ({str(a): np.shape(b) for a,b in v.items()} if isinstance(v,dict) else np.shape(v)) for k,v in banks.items()} if isinstance(banks,dict) else type(banks).__name__)
    for rational in (False, True):
        sweeps, tracers = [], []
        for fold in range(5):
            test = np.arange(len(x)) % 5 == fold
            def design(z): return np.c_[np.ones(len(z)),z]
            coef = np.linalg.lstsq(design(x[~test]),y[~test],rcond=None)[0]
            def predict(z, p):
                numerator = design(z)@p[:20].reshape(5,4)
                return numerator if not rational else numerator/(1+z*z*p[20:])
            p = coef.ravel()
            if rational:
                p = least_squares(lambda q: (predict(x[~test],q)-y[~test]).ravel(), np.r_[p,np.zeros(4)], bounds=(np.r_[np.full(20,-np.inf),np.zeros(4)],np.full(24,np.inf)), gtol=1e-12).x
            sweeps.append(np.sqrt(np.mean((predict(x[test],p)-y[test])**2,axis=0)).max())
            for ref, raw in pulses:
                err = (predict(ref,p)-raw).reshape(len(s['mu_shifts']),-1,4)
                tracers.append(np.sqrt(np.mean(err**2,axis=1)).max())
        print('saturating rational' if rational else 'affine','worst held-sweep and held-tracer slice RMS',max(sweeps),max(tracers,default=float('nan')))
