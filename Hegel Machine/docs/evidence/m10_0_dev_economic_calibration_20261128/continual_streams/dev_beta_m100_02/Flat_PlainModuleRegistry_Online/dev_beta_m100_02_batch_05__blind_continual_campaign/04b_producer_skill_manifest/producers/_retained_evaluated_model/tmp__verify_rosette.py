import copy, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _predict, _invert, calibrate_upstream
from batch_solver import solve_batch
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c = s['sensor_calibration_sweeps']; x,y = c['ref_rosette'],c['raw_rosette']; spike = np.max(abs(y),axis=1)>5
worst = np.zeros(3)
for order in (np.arange(len(x)), np.argsort(np.max(abs(x),axis=1)), np.random.default_rng(11).permutation(len(x))):
    for te in np.array_split(order,5):
        tr = np.setdiff1d(np.arange(len(x)),te); te = te[~spike[te]]
        f,p = _fit(x[tr],y[tr]); e = _invert(y[te],f,p)-x[te]
        worst = np.maximum(worst,[np.sqrt(np.mean((_predict(x[te],f,p)-y[te])**2)),np.sqrt(np.mean(e**2)),np.max(abs(e))])
print('public worst fold [forward RMSE, inverse RMSE, inverse max]',worst)
assert np.all(worst<[3e-5,6e-5,2e-4])
f,p = _fit(x,y)
rng=np.random.default_rng(82)
for amp in (.8,1.8,4.,12.):
    t=rng.uniform(-amp,amp,(20000,4)); raw=_predict(t,f,p); out=_invert(raw,f,p)
    print('inverse roundtrip',amp,np.max(abs(out-t)), np.sqrt(np.mean((out-t)**2)))
    assert np.max(abs(out-t))<1e-8
s=copy.deepcopy(s); obs=s['y_obs_raw'].copy(); raw_sweep=s['sensor_calibration_sweeps']['raw_rosette'].copy(); t=rng.uniform(-1.6,1.6,(18,4)); pulse=_predict(t,f,p).reshape(2,3,12); s['reference_tracer_pulses']=[{'y_pulse_raw':pulse,'label':'example'},{'y_pulse_raw':None}]
o=calibrate_upstream(s); oc=o['sensor_calibration_sweeps']; b=oc['single_bank_sweeps']
np.testing.assert_allclose(o['reference_tracer_pulses'][0]['y_pulse_raw'],t.reshape(pulse.shape),atol=1e-9,rtol=0)
assert o['reference_tracer_pulses'][0]['label']=='example' and o['reference_tracer_pulses'][1]['y_pulse_raw'] is None
for lhs,rhs in ((oc['raw_rosette'],oc['ref_rosette']),(b['raw_readout'],b['ref_signal'])):
    np.testing.assert_array_equal(lhs,rhs); assert not np.shares_memory(lhs,rhs)
np.testing.assert_array_equal(s['y_obs_raw'],obs); np.testing.assert_array_equal(s['sensor_calibration_sweeps']['raw_rosette'],raw_sweep)
assert o['y_obs_raw'].shape==obs.shape
out=solve_batch(s); print('pipeline fields',list(out)); print('certificate summary',[(k,list(v)) for k,v in out.items() if isinstance(v,dict)])
