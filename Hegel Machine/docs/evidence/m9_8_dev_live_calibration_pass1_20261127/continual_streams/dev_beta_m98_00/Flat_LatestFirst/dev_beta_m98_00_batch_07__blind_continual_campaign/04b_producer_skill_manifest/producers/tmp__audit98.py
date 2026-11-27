import importlib.util
import pickle
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
b = c['single_bank_sweeps']
source = s['source_modes']
pulses = s['reference_tracer_pulses']
modes = np.array([np.tensordot(p['w_pulse'], source, axes=1) for p in pulses])
expected = m._response(s, 'periodic', modes)[0]
rmse = {key: [] for key in ('deployed_inverse', 'sparse_forward', 'enlarged_forward', 'periodic_pulse')}
gross_total = 0
for fold in range(5):
    hold = np.arange(len(x)) % 5 == fold
    bank_hold = np.arange(len(b['bank_id'])) % 5 == fold
    sub = dict(ref_rosette=x[~hold], raw_rosette=raw[~hold], single_bank_sweeps={k: np.asarray(v)[~bank_hold] for k, v in b.items()})
    inv = m._calibration(sub)
    # Keep gross excursions as a separate validation slice, not Gaussian sweep noise.
    clean_rows = np.ones(hold.sum(), dtype=bool)
    sparse_error, enlarged_error = [], []
    for j in range(4):
        def features(t):
            return np.c_[np.ones(len(t)), t, t[:, j]**3, t[:, j]*t[:, (j+1)%4]]
        def extra(t):
            return np.column_stack([t[:, a]*t[:, d] for a,d in combinations(range(4), 2) if {a,d} != {j,(j+1)%4}] + [t[:, j]**2] + [t[:, j]*t[:, k]/(1+(t[:, j]+sign*t[:, k])**2) for k in range(4) if k != j for sign in (-1,1)])
        keep_bank = ~bank_hold & (b['bank_id'] == j)
        bank_x = np.eye(4)[j]*b['ref_signal'][keep_bank,None]
        train_x = np.vstack((x[~hold], bank_x))
        y = np.r_[raw[~hold,j], b['raw_readout'][keep_bank]]
        A = features(train_x)
        p0 = np.linalg.lstsq(A,y,rcond=None)[0]
        p = least_squares(lambda p: A@p-y,p0,jac=lambda p: A.copy(),loss='cauchy',f_scale=.03,max_nfev=250).x
        use = abs(A@p-y) < .003
        assert use.sum() >= .85*len(y)
        D = np.c_[A, extra(train_x)]
        p = np.linalg.lstsq(A[use],y[use],rcond=None)[0]
        q = np.linalg.lstsq(D[use],y[use],rcond=None)[0]
        E = features(x[hold])
        e = E@p-raw[hold,j]
        ee = np.c_[E,extra(x[hold])]@q-raw[hold,j]
        valid = abs(e) < .003
        sparse_error.append(np.sqrt(np.mean(e[valid]**2)))
        enlarged_error.append(np.sqrt(np.mean(ee[valid]**2)))
        clean_rows &= valid
    gross_total += int((~clean_rows).sum())
    test_x, test_raw = x[hold][clean_rows], raw[hold][clean_rows]
    rmse['deployed_inverse'].append(np.sqrt(np.mean((inv(test_raw.ravel()).reshape(-1,4)-test_x)**2,axis=0)).max())
    rmse['sparse_forward'].append(max(sparse_error))
    rmse['enlarged_forward'].append(max(enlarged_error))
    measured = np.column_stack([inv(p['y_pulse_raw']) for p in pulses])
    rmse['periodic_pulse'].append(np.sqrt(np.mean((expected-measured).reshape(len(s['mu_shifts']),-1,len(pulses))**2,axis=1)).max())
print('worst fold/channel or shift/pulse RMSE', {k: max(v) for k,v in rmse.items()}, 'gross validation rows', gross_total)
assert max(rmse['deployed_inverse']) < 1e-4 and max(rmse['periodic_pulse']) < 2e-5
