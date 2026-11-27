import pickle,numpy as np,itertools,sys
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m98_00_batch_04_solver as b, dev_beta_m98_00_batch_03_solver as d
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=s['Z_nuisance']; p=np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); ridge=s['source_ridge']; cap=s['u_box']; total=s['total_budget']; src=s['source_modes']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; raw=c['raw_rosette']; inv,fwd,_=b._calibration(c); inv3=d._calibration(c)
for name,cal in [('03',inv3),('04',inv)]:
 err=cal(raw).reshape(x.shape)-x; print('sweep',name,'rms',np.sqrt(np.mean(err**2,axis=0)),'max',np.max(abs(err),axis=0))
 y=cal(s['y_obs_raw']);print('cal observations',name,np.quantile(y,[0,.5,1]));
 for walls in itertools.product(('open','sealed','periodic'),repeat=2):
  S,_=d._response(s,walls,src); w=b._weights(p@S,p@y,ridge,cap,total); r=p@(S@w-y); print(name,walls,'rmse by shift',np.round(np.sqrt(np.mean(r.reshape(3,-1)**2,axis=1)),6),'norm',np.linalg.norm(r),'w',np.round(w,4))
