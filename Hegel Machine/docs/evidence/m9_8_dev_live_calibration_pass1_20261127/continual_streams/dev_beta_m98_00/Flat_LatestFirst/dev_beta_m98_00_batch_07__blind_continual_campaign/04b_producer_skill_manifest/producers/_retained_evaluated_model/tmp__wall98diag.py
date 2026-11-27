import pickle,numpy as np
from scipy.optimize import least_squares
import model98 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=m._calibration(s['sensor_calibration_sweeps']); print('calibrator',f.__qualname__)
p=s['reference_tracer_pulses']; modes=np.array([np.tensordot(q['w_pulse'],s['source_modes'],axes=1) for q in p]); y=np.column_stack([f(q['y_pulse_raw']) for q in p]); print('calibrated y min,max',y.min(),y.max())
layouts=[[0,0,1,1,2,2,3,3],[0,1,2,3,4,5,6,7]]
for ids in layouts:
 ids=np.array(ids); n=ids.max()+1; robin=np.unique(ids[[0,1,4,5]]); coupling=np.unique(ids[[2,3,6,7]]); start=np.ones(n)
 for neg in (False,True):
  lb=np.zeros(n)+.025; lb[robin]=-.65 if neg else 0; ub=np.full(n,5.); ub[robin]=4.
  def pred(w): return m._response(s,w[ids],modes)[0]
  for j in (0,1,None):
   jj=slice(None) if j is None else j
   fun=lambda w:(pred(w)-y)[:,jj].ravel()
   opts=[least_squares(fun,np.where(np.isin(np.arange(n),robin),v,1.),bounds=(lb,ub),max_nfev=120,ftol=1e-11,gtol=1e-10) for v in ([1.,.1,-.2] if neg else [1.,.1])]
   opt=min(opts,key=lambda o:o.cost); e=pred(opt.x)-y
   print('layout',n,'negative',neg,'train pulse',j,'wall',np.round(opt.x[ids],6),'RMSE',np.sqrt(np.mean(e**2,0)),'worst shift RMSE',np.max(np.sqrt(np.mean(e.reshape(3,32,2)**2,1)),axis=0),'Jacobian sv',np.round(np.linalg.svd(opt.jac,compute_uv=False),5),flush=True)
