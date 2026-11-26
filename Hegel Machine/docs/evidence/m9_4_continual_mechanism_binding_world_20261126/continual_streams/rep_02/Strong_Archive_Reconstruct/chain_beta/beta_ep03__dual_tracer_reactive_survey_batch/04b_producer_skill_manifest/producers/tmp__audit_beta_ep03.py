import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_ep03_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_beta_ep03.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, gt = item['study'], item['gt']
sweep = s['sensor_calibration_sweeps']
bank, x, y = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
bank_error, pooled_error = [], []
for i in range(len(x)):
    keep = np.arange(len(x)) != i
    for mask, errors in ((keep & (bank == bank[i]), bank_error), (keep, pooled_error)):
        gain, offset = np.linalg.lstsq(np.column_stack((x[mask], np.ones(mask.sum()))), y[mask], rcond=None)[0]
        errors.append(abs(gain*x[i] + offset - y[i]))
for b in np.unique(bank):
    mask = bank == b
    print('bank', b, 'held-out max raw error: bank-specific', max(np.asarray(bank_error)[mask]), 'pooled', max(np.asarray(pooled_error)[mask]))
assert max(bank_error) < 1e-6

out = solver.solve_beta_ep03(s)
w = out['weights']
source = np.einsum('m,mij->ij', w, s['source_modes'])
def apply_axis(field, axis):
    values = field if axis == 'x' else field.T
    n = len(values)
    h = s['L'+axis]/n
    k, velocity, reaction = (np.asarray(s[key+'_'+axis]) for key in ('kappa', 'vel', 'reaction'))
    robin = np.broadcast_to(s['robin_'+axis], (2,))
    result = reaction[:,None] * values
    for i in range(n):
        for j in (i-1, i+1):
            if 0 <= j < n:
                result[i] += (2/(1/k[i]+1/k[j])) * (values[i]-values[j])/h**2
        upwind = i-1 if velocity[i] >= 0 else i+1
        other = values[upwind] if 0 <= upwind < n else 0.
        result[i] += abs(velocity[i]) * (values[i]-other)/h
    for i, r in zip((0,n-1), robin):
        result[i] += (r/h)/(1+r*h/(2*k[i])) * values[i]
    return result if axis == 'x' else result.T
for channel in ('a','b'):
    v = s['V_channel_'+channel]
    defect = np.max(np.abs(v-v[:,:1]-v[:1,:]+v[0,0]))
    field = out['terminal_field_channel_'+channel]
    residual = apply_axis(field, 'x') + apply_axis(field, 'y') + (v+s['mu_shifts'][-1])*field - source
    maximum = np.max(np.abs(residual))
    boundary = np.concatenate((residual[0], residual[-1], residual[1:-1,0], residual[1:-1,-1]))
    print('channel', channel, 'potential additivity defect', defect, 'terminal residual max', maximum, 'boundary max', np.max(np.abs(boundary)))
    assert maximum < 1e-9
