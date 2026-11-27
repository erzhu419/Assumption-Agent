import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
st = item['study']
sw = st['sensor_calibration_sweeps']
x, y = sw['ref_rosette'], sw['raw_rosette']
beta = np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0]
single = sw['single_bank_sweeps']
bank = single['bank_id'].astype(int)
signal, readout = single['ref_signal'], single['raw_readout']
print('row bank IDs match rosette positions:', np.array_equal(st['row_bank_id'], np.tile(np.arange(4), len(st['y_obs_raw'])//4)))
S, _ = m._forward(st)
for j in np.unique(bank):
    mask = bank == j
    A = np.c_[np.ones(mask.sum()), signal[mask]]
    prior = beta[[0,j+1],j]
    fitted = np.linalg.lstsq(A, readout[mask], rcond=None)[0]
    loo = np.array([A[i]@np.linalg.lstsq(np.delete(A,i,axis=0), np.delete(readout[mask],i), rcond=None)[0]-readout[mask][i] for i in range(len(A))])
    print('bank', j, 'full-sweep bias/gain', prior, 'single-bank bias/gain', fitted, 'single-bank transfer/LOO RMS', np.sqrt(np.mean((A@prior-readout[mask])**2)), np.sqrt(np.mean(loo**2)))

def tracer_metric(b):
    errors = []
    for p in item['campaign_diagnostics']:
        ycal = np.linalg.solve(b[1:].T, (p['y_pulse_raw'].reshape(-1,4)-b[0]).T).T.ravel()
        errors.append((ycal-S@p['w_pulse']).reshape(len(st['mu_shifts']),-1,4))
    return np.sqrt(np.mean(np.asarray(errors)**2,axis=2))

B = np.zeros((len(bank),5,4))
B[np.arange(len(bank)),0,bank] = 1
B[np.arange(len(bank)),bank+1,bank] = signal
pooled = beta.copy()
for j in range(4):
    mask = bank == j
    pooled[:,j] = np.linalg.lstsq(np.vstack([np.c_[np.ones(len(x)),x],B[mask,:,j]]),np.r_[y[:,j],readout[mask]],rcond=None)[0]
for name, b in [('rosette-only', beta), ('pooled sweeps', pooled)]:
    rms = tracer_metric(b)
    print(name, 'worst shift-by-bank RMS per pulse', rms.max(axis=(1,2)), 'bank 1 by pulse and shift', rms[:,:,1])
