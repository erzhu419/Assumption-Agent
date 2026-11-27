import numpy as np, pickle, importlib.util
from itertools import product
from chamber_calibration import _calibration

def archived(n):
    p='/tmp/hegel_continual_archive/dev_beta_m98_00_batch_0'+str(n)+'_solver.py'
    spec=importlib.util.spec_from_file_location('prior',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; modes=s['source_modes']; inv,fwd,fit=_calibration(s['sensor_calibration_sweeps']); sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; ii=np.arange(len(x));
def design(x,j): return np.column_stack((np.ones(len(x)),x,x[:,j]*x[:,(j+1)%4],x[:,j]**3))
r=np.empty_like(y)
for k in range(4):
    tr=ii%4!=k
    for j in range(4): r[~tr,j]=design(x[~tr],j)@fit(j,tr)-y[~tr,j]
print('sweep worst fold inlier RMSE',max(np.sqrt(np.mean(r[(ii%4==k)&(abs(r).max(1)<.01)]**2)) for k in range(4)), 'outlier rows',ii[abs(r).max(1)>=.01]); p=s['reference_tracer_pulses']; y=np.column_stack([inv(t['y_pulse_raw']) for t in p]); w=np.column_stack([t['w_pulse'] for t in p]); print('inverse reproduction',max(np.max(abs(fwd(y[:,j].reshape(-1,4)).ravel()-p[j]['y_pulse_raw'])) for j in range(len(p))))
m=archived(3)
for pair in product(('open','sealed','periodic'),repeat=2):
    S,_=m._response(s,pair,modes); r=(S@w-y).reshape(len(s['mu_shifts']),-1,len(p)); print(pair,np.sqrt(np.mean(r*r,axis=1)).round(7).tolist())
m=archived(2); wall=m._walls(s,inv,modes); S,_=m._response(s,wall,modes); print('graded',wall,np.sqrt(np.mean((S@w-y)**2,axis=0))); print('rosette aux shape',s['aux_manifold_telemetry'].shape)
