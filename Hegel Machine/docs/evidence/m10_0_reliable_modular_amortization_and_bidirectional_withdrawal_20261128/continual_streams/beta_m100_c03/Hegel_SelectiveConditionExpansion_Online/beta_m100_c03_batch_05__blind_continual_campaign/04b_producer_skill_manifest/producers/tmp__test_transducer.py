import sys,pickle,copy,numpy as np
sys.path.insert(0,'/holdout_task')
from upstream_transducer import _fit,_forward,_inverse,calibrate_upstream
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; g=np.max(abs(y),1)<6
es=[]
for i in np.flatnonzero(g):
 keep=np.arange(len(x))!=i; m=_fit(dict(c,ref_rosette=x[keep],raw_rosette=y[keep])); es.append(_inverse(y[i],m)-x[i])
e=np.array(es); axis=np.count_nonzero(x[g],axis=1)==1; outer=np.max(abs(x[g]),axis=1)>1.25
for n,k in [('axis',axis),('joint',~axis),('outer',outer),('all',axis|~axis)]: print(n,'n',sum(k),'rmse',np.sqrt(np.mean(e[k]**2,axis=0)),'max',np.max(abs(e[k])))
m=_fit(c); rg=np.random.default_rng(9); bounds=m[-1]; z=rg.uniform(*bounds,size=(1000,4)); f,j=_forward(z,m); print('inverse round trip',np.max(abs(_inverse(f,m)-z)),'min singular value',np.linalg.svd(j,compute_uv=False).min()); print('observation calibrated range',np.min(_inverse(s['y_obs_raw'],m)),np.max(_inverse(s['y_obs_raw'],m)))
t=copy.deepcopy(s); t['reference_tracer_pulses']=[{'y_pulse_raw':f[:8].ravel(),'tag':'pulse'},{'y_pulse_raw':None},None]; a=calibrate_upstream(t); assert np.max(abs(a['reference_tracer_pulses'][0]['y_pulse_raw'].reshape(-1,4)-z[:8]))<1e-8; assert a['reference_tracer_pulses'][1:]==t['reference_tracer_pulses'][1:]; assert np.array_equal(t['reference_tracer_pulses'][0]['y_pulse_raw'],f[:8].ravel()); assert np.array_equal(a['sensor_calibration_sweeps']['raw_rosette'],x); assert np.array_equal(a['sensor_calibration_sweeps']['single_bank_sweeps']['raw_readout'],c['single_bank_sweeps']['ref_signal'])
o=solve_batch(s); w=o['weights']; cert=o['kkt_certificate']; print('budget',abs(w.sum()-s['total_budget']),'min weight',w.min(),'max weight',w.max(),'gap',cert['primal_objective']-cert['dual_lower_bound'])
