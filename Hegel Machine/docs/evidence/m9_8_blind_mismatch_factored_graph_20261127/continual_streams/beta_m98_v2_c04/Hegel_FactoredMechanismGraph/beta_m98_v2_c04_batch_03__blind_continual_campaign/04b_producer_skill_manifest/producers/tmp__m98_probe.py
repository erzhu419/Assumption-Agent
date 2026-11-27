import glob,importlib.util,pickle,numpy as np
mods=[]
for path in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py'))[:2]:
 spec=importlib.util.spec_from_file_location('archive',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mods.append(mod)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; cal=s['sensor_calibration_sweeps']; x,y=cal['ref_rosette'],cal['raw_rosette']; z=s['Z_nuisance']; pz=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
def fit(x,y):
 return np.linalg.lstsq(x,y,rcond=None)[0]
pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
features={'affine':lambda x:np.c_[np.ones(len(x)),x], 'cubic':lambda x:np.c_[np.ones(len(x)),x,x*x,x**3], 'cross':lambda x:np.c_[np.ones(len(x)),x,x*x,x**3,*[x[:,i]*x[:,j] for i,j in pairs]]}
for name,f in features.items():
 d=f(x); es=[]
 for ix in np.array_split(np.arange(len(x)),5):
  tr=np.setdiff1d(np.arange(len(x)),ix); es.append(np.sqrt(np.mean((d[ix]@fit(d[tr],y[tr])-y[ix])**2,axis=0)))
 print(name,'CV worst/ch',np.max(es,axis=0)); print('coef',np.round(fit(d,y),5))
b=cal['single_bank_sweeps']; print('single bank',[(k,np.shape(v),np.asarray(v).ravel()[:12]) for k,v in b.items()]); print('rosette banks',s['row_bank_id'].reshape(-1,4)[:8]);print('cal ranges',x.min(0),x.max(0),'obs raw ranges',s['y_obs_raw'].min(),s['y_obs_raw'].max()); print('shift',s['mu_shifts']);print('Z size',np.sqrt(np.mean(z*z,axis=0)))
for m in mods:
 S,U=m._forward(s); print(m.__file__, 'sv',np.linalg.svd(S,compute_uv=False));
 for k,pu in enumerate(item['campaign_diagnostics']):
  yc=m._calibrate(dict(s,y_obs_raw=pu['y_pulse_raw'])); pred=S@pu['w_pulse']; r=yc-pred; rp=r-z@(pz@r); print('pulse',k,'rmse',np.sqrt(np.mean(r*r)), 'profiled shift rms',np.sqrt(np.mean(rp.reshape(3,-1)**2,axis=1)), 'yc/pred norms',np.linalg.norm(yc),np.linalg.norm(pred),'residual by channel',np.sqrt(np.mean(rp.reshape(-1,4)**2,axis=0)))
