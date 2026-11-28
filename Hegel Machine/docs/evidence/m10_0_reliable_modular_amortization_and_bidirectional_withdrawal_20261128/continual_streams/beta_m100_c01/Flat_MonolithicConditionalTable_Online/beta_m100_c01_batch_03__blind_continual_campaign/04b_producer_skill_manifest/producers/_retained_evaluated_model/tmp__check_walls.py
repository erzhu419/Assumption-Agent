import pickle,itertools,json,numpy as np
import wall_probe as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibrate(s); z=s['Z_nuisance']; ridge=s['source_ridge']; nr=s['nuisance_ridge']; cap=s['u_box']; budget=s['total_budget']; n=len(y); ns=len(s['sensor_ij']); sensor=np.tile(np.arange(ns),n//ns); shift=np.repeat(np.arange(n//ns),ns); rosette=sensor//4
print('gate',open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json').read()); t=s['aux_manifold_telemetry']; print('aux mean/std',t.mean(0),t.std(0)); print('y cal range',y.min(),y.max())
c=s['sensor_calibration_sweeps']; x,v=c['ref_rosette'],c['raw_rosette']; loo=[]
for i in range(len(x)):
 mask=np.arange(len(x))!=i
 loo.append((m._forward(x[i:i+1],m._fit_calibration(x[mask],v[mask]))-v[i:i+1])[0])
loo=np.array(loo); clean=np.max(abs(loo),axis=1)<0.01; print('calibration LOO bank RMSE',np.sqrt(np.mean(loo[clean]**2,axis=0)),'outlier rows',np.flatnonzero(~clean))
def fit(a, zz, yy):
 P=np.linalg.solve(zz.T@zz+nr*np.eye(zz.shape[1]),zz.T); A=a-zz@(P@a); r=yy-zz@(P@yy); w,nu=m._box_simplex(A.T@A+ridge*np.eye(A.shape[1]),-A.T@r,cap,budget); return w,P@(yy-a@w),np.sqrt(np.mean((A@w-r)**2))
for walls in itertools.product(('robin','reflect','periodic'),repeat=2):
 A,U=m._transport(s,walls); w,g,full=fit(A,z,y); stats=[]
 for labels in (sensor,rosette,shift):
  e=np.empty(n); worst=0
  for k in np.unique(labels):
   test=labels==k; tr=~test; wt,gt,_=fit(A[tr],z[tr],y[tr]); e[test]=y[test]-A[test]@wt-z[test]@gt; worst=max(worst,np.sqrt(np.mean(e[test]**2)))
  stats.append((round(np.sqrt(np.mean(e**2)),7),round(worst,7)))
 print(walls,'train',round(full,7),'CV pooled/worst by sensor/rosette/shift',stats,'weights',np.round(w,4))
