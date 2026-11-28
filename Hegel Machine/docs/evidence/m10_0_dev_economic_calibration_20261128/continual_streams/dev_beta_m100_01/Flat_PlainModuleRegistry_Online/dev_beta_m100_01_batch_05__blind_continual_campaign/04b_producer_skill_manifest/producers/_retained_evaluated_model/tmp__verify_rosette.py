import pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c = s['sensor_calibration_sweeps']; x,y,bank = (c[k] for k in ('ref_rosette','raw_rosette','single_bank_sweeps'))
clean = np.max(np.abs(y),axis=1)<5
worst = 0
for ids in np.array_split(np.argsort(np.linalg.norm(x,axis=1)),5):
    tr = np.ones(len(x),bool); tr[ids] = False
    p = up._fit(x[tr],y[tr],bank)
    ids = ids[clean[ids]]
    d = up._inverse(y[ids],p)-x[ids]
    err = np.sqrt(np.mean(d*d)); worst=max(worst,err)
    print('held-out inverse RMSE/max:',err,np.max(np.abs(d)))
assert worst<1e-4
p=up._fit(x,y,bank)
slope,amp,freq,cross,even,cf=p[8:]
margin=slope-abs(amp*freq)-abs(cross)-abs(even*cf)
print('parameters:',p,'global diagonal-dominance margin:',margin)
assert margin>0
z=np.array(np.meshgrid(*[np.linspace(lo,hi,9) for lo,hi in zip(x.min(0),x.max(0))])).reshape(4,-1).T
roundtrip=np.max(np.abs(up._inverse(up._response(z,p),p)-z))
print('edge-grid roundtrip max:',roundtrip)
assert roundtrip<1e-9
pulses=[{'y_pulse_raw':up._response(z[:6],p).reshape(2,3,4),'metadata':7},{'y_pulse_raw':None},None]
s2=dict(s,reference_tracer_pulses=pulses); out=up.calibrate_upstream(s2)
assert np.allclose(out['reference_tracer_pulses'][0]['y_pulse_raw'],z[:6].reshape(2,3,4),rtol=0,atol=1e-9)
assert out['reference_tracer_pulses'][0]['metadata']==7
assert out['reference_tracer_pulses'][1:]==pulses[1:]
assert out['y_obs_raw'].shape==s['y_obs_raw'].shape
assert np.max(np.abs(up._response(out['y_obs_raw'].reshape(-1,4),p)-s['y_obs_raw'].reshape(-1,4)))<1e-9
for target,refkey,rawkey in [(out['sensor_calibration_sweeps'],'ref_rosette','raw_rosette'),(out['sensor_calibration_sweeps']['single_bank_sweeps'],'ref_signal','raw_readout')]:
    assert np.array_equal(target[refkey],target[rawkey]) and not np.shares_memory(target[refkey],target[rawkey])
assert not np.array_equal(c['ref_rosette'],c['raw_rosette'])
assert np.array_equal(up.calibrate_upstream(out)['y_obs_raw'],out['y_obs_raw'])
sys.argv=['/holdout_task/validate_public_schema.py','/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0],run_name='__main__')
