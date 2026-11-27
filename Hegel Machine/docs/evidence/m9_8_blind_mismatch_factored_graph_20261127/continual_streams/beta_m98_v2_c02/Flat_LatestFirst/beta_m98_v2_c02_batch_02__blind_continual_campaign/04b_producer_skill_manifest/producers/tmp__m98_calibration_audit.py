import importlib.util, json, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(p+'public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
with open('/tmp/hegel_continual_archive/archive_apparatus_log.json') as f:
 for entry in json.load(f):
  print('apparatus fields',[(k,np.shape(v) if isinstance(v,list) else list(v) if isinstance(v,dict) else v) for k,v in entry.items()])
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; groups=np.arange(len(x))%5; pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
q=np.column_stack([x]+[x[:,i:i+1]+sign*x[:,j:j+1] for i,j in pairs for sign in [-1,1]])
def fit_predict(z,tr,te,b):
 a=np.linalg.lstsq(z[tr],y[tr,b],rcond=None)[0]; return z[te]@a
A=np.c_[np.ones(len(x)),x]
D=[np.c_[np.ones(len(x)),x[:,b]] for b in range(4)]
R=[(i,a,np.c_[A,A/(1+a*q[:,i:i+1]**2)]) for i in range(q.shape[1]) for a in (0.03,0.1,0.3,1.)]
err={name:np.empty_like(y) for name in ['diagonal','affine','rational']}
for f in range(5):
 te=groups==f; tr=~te
 for b in range(4):
  for name,z in [('diagonal',D[b]),('affine',A)]: err[name][te,b]=fit_predict(z,tr,te,b)-y[te,b]
  def score(item):
   z=item[2]; return sum(np.sum((fit_predict(z,tr&(groups!=h),groups==h,b)-y[groups==h,b])**2) for h in range(5) if h!=f)
  z=min(R,key=score)[2]; err['rational'][te,b]=fit_predict(z,tr,te,b)-y[te,b]
for name,e in err.items(): print(name,'OOF per-channel RMSE',np.sqrt(np.mean(e**2,axis=0)),'worst-fold RMSE',max(np.sqrt(np.mean(e[groups==f]**2)) for f in range(5)))
co=np.linalg.lstsq(A,y,rcond=None)[0]; print('largest off-diagonal affine gain',np.max(abs(co[1:]-np.diag(np.diag(co[1:])))))
o=m.solve_batch(s); k=o['kkt_certificate']; print('isolated solve weights/field shapes',o['weights'].shape,o['terminal_field'].shape,'gap',k['primal_objective']-k['dual_lower_bound'])
