import importlib.util,pickle,itertools,numpy as np
p=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/beta_m100_c04_batch_06_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; print('pulse diagnostics:', type(i.get('campaign_diagnostics'))); s=i['study']; y=m._calibrate(s); Z=s['Z_nuisance']; ridge,cap,budget=[s[k] for k in ('source_ridge','u_box','total_budget')]; print('calibrated min max',y.min(),y.max()); print('cal rms clean',np.sqrt(np.mean((m._fit_rosette(s['sensor_calibration_sweeps'])(s['sensor_calibration_sweeps']['ref_rosette'])-s['sensor_calibration_sweeps']['raw_rosette'])[np.abs(s['sensor_calibration_sweeps']['raw_rosette'])<8]**2)))
Zr=s['nuisance_ridge']; results=[]; groups=np.tile(np.arange(len(y)//12),12) if False else np.arange(len(y)).reshape(3,-1,4); rosettes=np.broadcast_to(np.arange(groups.shape[1])[None,:,None],groups.shape).ravel(); sensors=np.tile(s['sensor_ij'],(3,1)); idxs=[rosettes%4==k for k in range(4)]+[sensors[:,0]==0,sensors[:,1]==0];
for laws in itertools.product(['periodic',(0.,0.),(1.,1.),(0.,1.),(1.,0.)],repeat=2):
 S,U=m._forward(s,laws); A=m._project(Z,Zr,np.column_stack((S,y))); w,nu=m._qp(A[:,:-1],A[:,-1],ridge,cap,budget); err=A[:,:-1]@w-A[:,-1]; metrics=[]
 for val in idxs:
  tr=~val; At=m._project(Z[tr],Zr,np.column_stack((S[tr],y[tr]))); wt,_=m._qp(At[:,:-1],At[:,-1],ridge,cap,budget); gam=np.linalg.solve(Z[tr].T@Z[tr]+Zr*np.eye(Z.shape[1]),Z[tr].T@(y[tr]-S[tr]@wt)); ev=S[val]@wt+Z[val]@gam-y[val]; metrics.append(np.sqrt(np.mean(ev**2)))
 results.append((np.sqrt(np.mean(err**2)),str(laws),np.round(metrics,6),np.round(w,4)))
for r in sorted(results,key=lambda r:r[0]):print(*r)
