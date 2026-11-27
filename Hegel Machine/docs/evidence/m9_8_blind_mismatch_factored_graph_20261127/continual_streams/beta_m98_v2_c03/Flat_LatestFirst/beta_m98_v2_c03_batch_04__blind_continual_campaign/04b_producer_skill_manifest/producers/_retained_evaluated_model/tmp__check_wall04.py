import glob,importlib.util,pickle,numpy as np

def load(path):
 spec=importlib.util.spec_from_file_location('m',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

p=load('/holdout_task/batch_solver.py'); batch=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
for item in batch:
 s=item['study']; y=p._calibrate(s); Z=s['Z_nuisance']; ridge=s['nuisance_ridge']; sr=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']; idx=np.arange(len(Z)); shifts=len(s['mu_shifts']); sensors=len(s['sensor_ij']); designs=[(name,m._forward(s)[0]) for name,m in [('reflecting',p),('Robin',load(glob.glob('/tmp/hegel_continual_archive/*batch_02_solver.py')[0])),('periodic',load(glob.glob('/tmp/hegel_continual_archive/*batch_03_solver.py')[0]))]]
 for name,D in designs:
  def fit(tr):
   z=Z[tr]; v=np.c_[D[tr],y[tr]]; proj=np.linalg.solve(z.T@z+ridge*np.eye(z.shape[1]),z.T); d,t=(v-z@(proj@v))[:,:-1],(v-z@(proj@v))[:,-1]; w,nu=p._box_budget(d.T@d+sr*np.eye(d.shape[1]),-d.T@t,cap,budget); return w,proj@(y[tr]-D[tr]@w),np.linalg.norm(d@w-t)
  wall=np.any((s['sensor_ij']==0)|(s['sensor_ij']==np.array([s['nx']-1,s['ny']-1])),axis=1); wall=np.tile(wall,shifts)
  print(name,'full weights',np.round(fit(idx)[0],6),'full projected residual norm',fit(idx)[2]);
  for key,groups in [('rosette',idx.reshape(-1,4)),('sensor',idx.reshape(shifts,sensors).T),('shift',idx.reshape(shifts,sensors)),('wall',[idx[wall],idx[~wall]])]:
   errors=[]
   for te in groups:
    w,g,_=fit(np.setdiff1d(idx,te)); errors.append(y[te]-D[te]@w-Z[te]@g)
   print(key,'pooled',np.sqrt(np.mean(np.concatenate(errors)**2)),'worst',max(np.sqrt(np.mean(e**2)) for e in errors))
 sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; raw=sw['raw_rosette']; C=p._fit_calibration(x,raw); print('cubic coefficients >0.001:')
 for t,row in zip([(),*((i,) for i in range(4)),*p._TERMS],C):
  if max(abs(row))>0.001: print(t,np.round(row,5))
