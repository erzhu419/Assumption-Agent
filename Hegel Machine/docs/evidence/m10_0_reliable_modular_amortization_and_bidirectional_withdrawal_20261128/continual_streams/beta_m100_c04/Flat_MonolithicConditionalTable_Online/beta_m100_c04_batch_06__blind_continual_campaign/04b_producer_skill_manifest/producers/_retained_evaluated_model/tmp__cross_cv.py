import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('batch','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=(c[k] for k in ('ref_rosette','raw_rosette')); good=np.max(np.abs(y),axis=1)<8; x,y=x[good],y[good]; pure=np.count_nonzero(x,axis=1)==1; op=[2,3,0,1]; b=c['single_bank_sweeps']; i,t,z=(b[k] for k in ('bank_id','ref_signal','raw_readout')); ok=np.abs(z)<8;i,t,z=i[ok],t[ok],z[ok]; v=x[:,op]; poly=np.polynomial.polynomial.polyval
forms={**{f'poly{k}':(lambda q,p:poly(q,np.r_[0.,p]),np.zeros(k)) for k in (4,5,7,9,11)}, 'rational':(lambda q,p:q*poly(q,p[:4])/(1+p[4]*q*q),[.13,-.07,.01,.01,.1]),'rational2':(lambda q,p:q*poly(q,p[:4])/(1+p[4]*q*q+p[5]*q**4),[.13,-.07,.01,.01,.1,.03]),'mixed':(lambda q,p:poly(q,np.r_[0.,p[:7]])+p[7]*q*x+p[8]*q*q*x+p[9]*q*x*x,np.zeros(10))}
for name,(cross,p0) in forms.items():
 errors=[]; biases=[]
 for fold in range(5):
  h=(np.arange(len(x))%5==fold)&~pure
  def selfr(q,p):return q*poly(q*q,np.r_[1.,p[8:14]])
  def pred(p):return p[:4]+p[4:8]*(selfr(x,p)+cross(v,p[14:]))
  def err(p):return np.r_[(pred(p)[~h]-y[~h]).ravel(),p[:4][i]+p[4:8][i]*selfr(t,p)-z]
  p=least_squares(err,np.r_[np.zeros(4),[2.,1.3,1.3,2.],-.4,.12,-.018,0.,0.,0.,p0],gtol=1e-12,max_nfev=300).x
  er=(pred(p)-y)[h]; errors.append((np.sqrt(np.mean(er**2)),np.max(abs(er)))); biases.append(p[14:])
 print(name,'worst RMS/max',np.max(errors,0),'cross',np.mean(biases,0).round(7))
out=m.solve_batch(s); q=out['kkt_certificate']; print('w',out['weights']); print('objective and gap',q['primal_objective'],q['primal_objective']-q['dual_lower_bound'])
