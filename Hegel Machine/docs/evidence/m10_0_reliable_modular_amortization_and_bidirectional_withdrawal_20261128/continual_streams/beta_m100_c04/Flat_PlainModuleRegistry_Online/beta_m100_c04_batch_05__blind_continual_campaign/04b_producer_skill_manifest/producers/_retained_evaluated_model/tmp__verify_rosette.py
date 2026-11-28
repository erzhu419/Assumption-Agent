import sys
sys.path.insert(0, '/holdout_task')
import pickle, numpy as np
import upstream_transducer as u
from batch_solver import solve_batch
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
s = items[0]['study']
c = s['sensor_calibration_sweeps']
x,y = c['ref_rosette'],c['raw_rosette']
good = np.max(abs(y),axis=1)<5
rows = np.flatnonzero(good)
blocked = np.array_split(rows[np.argsort(np.max(abs(x[rows]),axis=1))],6)
loo = [np.array([i]) for i in rows]
for name, folds in [('amplitude blocks',blocked), ('leave one out',loo)]:
    max_raw, max_inv = 0., 0.
    for fold in folds:
        train = np.ones(len(x),dtype=bool)
        train[fold] = False
        fit = u._fit(x[train],y[train])
        re = np.max(abs(u._response(x[fold],fit)[0]-y[fold]))
        ie = np.max(abs(u._invert(y[fold],fit)-x[fold]))
        max_raw,max_inv = max(max_raw,re),max(max_inv,ie)
        if name == 'amplitude blocks': print('block raw/inverse',re,ie)
    print(name,'max raw/inverse',max_raw,max_inv)
    assert max_raw < 2e-4 and max_inv < 5e-4
model = u._fit(x,y)
coef,freq = model
margin = coef[:,1]-abs(freq[0]*coef[:,3])-abs(coef[:,2])-abs(freq[1]*coef[:,4])
print('frequencies',freq,'global diagonal dominance margin',margin)
assert np.min(margin) > 0.
# Additional gross and moderate calibration-row contamination must not bias the inverse.
corrupt = y.copy()
corrupt[rows[::13]] += [21,-18,17,-20]
corrupt[rows[5::13]] += [.23,-.35,.4,-.19]
fit = u._fit(x,corrupt)
err = np.max(abs(u._invert(y[good],fit)-x[good]))
print('added EMI maximum inverse error',err)
assert err < 5e-4
# Use real sweep pairs as tracer pulses; test optional and absent entries separately.
pulse_rows = rows[np.argsort(np.max(abs(x[rows]),axis=1))[-8:]]
study = dict(s,reference_tracer_pulses=[None,{'y_pulse_raw':None}, {'y_pulse_raw':y[pulse_rows].copy(),'tag':'outer'}])
out = u.calibrate_upstream(study)
pulses = out['reference_tracer_pulses']
assert pulses[0] is None and pulses[1]['y_pulse_raw'] is None and pulses[2]['tag']=='outer'
assert np.max(abs(pulses[2]['y_pulse_raw']-x[pulse_rows])) < 5e-4
assert np.array_equal(study['reference_tracer_pulses'][2]['y_pulse_raw'],y[pulse_rows])
assert out['y_obs_raw'].shape == s['y_obs_raw'].shape
assert np.max(abs(u._response(out['y_obs_raw'].reshape(-1,4),model)[0].ravel()-s['y_obs_raw'])) < 1e-9
nc = out['sensor_calibration_sweeps']
for group, a, b in [(nc,'raw_rosette','ref_rosette'),(nc['single_bank_sweeps'],'raw_readout','ref_signal')]:
    assert np.array_equal(group[a],group[b]) and not np.shares_memory(group[a],group[b])
assert np.array_equal(u.calibrate_upstream(out)['y_obs_raw'],out['y_obs_raw'])
for item in items:
    result = solve_batch(item['study'])
    assert set(result) == {'weights','terminal_field','kkt_certificate'}
    assert np.all(np.isfinite(result['weights'])) and np.all(np.isfinite(result['terminal_field']))
print('isolated calibration and downstream checks passed for',len(items),'public items')
