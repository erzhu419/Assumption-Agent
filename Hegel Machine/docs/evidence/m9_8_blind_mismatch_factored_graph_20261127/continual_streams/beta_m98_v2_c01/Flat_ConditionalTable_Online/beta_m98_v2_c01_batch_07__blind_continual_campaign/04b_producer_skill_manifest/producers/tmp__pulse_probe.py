import glob, importlib.util, pickle, numpy as np
mods={}
for path in sorted(glob.glob('/tmp/hegel_continual_archive/*_solver.py')):
 k=path.split('_')[-2]; spec=importlib.util.spec_from_file_location(k,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[k]=m
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; p=it['campaign_diagnostics']; z=s['Z_nuisance']; w=np.array([a['w_pulse'] for a in p]).T
P=np.eye(len(z))-z@np.linalg.pinv(z)
for i in ['02','03','06']:
 S,U=mods[i]._forward(s)
 for j in ['02','04','06']:
  y=np.column_stack([mods[j]._calibrate(dict(s,y_obs_raw=a['y_pulse_raw'])) for a in p]); r=y-S@w
  print('wall/cal',i,j,'RMSE',np.sqrt(np.mean(r**2,axis=0)).round(6),'projected',np.sqrt(np.mean((P@r)**2,axis=0)).round(6),'shift slice',np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1)).round(6).tolist())
a=s['sensor_calibration_sweeps']; x=a['ref_rosette']; y=a['raw_rosette']; f=mods['06']._calibration_features
print('pulse ranges',[(float(a['y_pulse_raw'].min()),float(a['y_pulse_raw'].max())) for a in p]);print('reference range',x.min(axis=0),x.max(axis=0)); print('raw row maxima sorted',np.sort(np.abs(y).max(axis=1)).round(3)); print('banks',np.unique(s['row_bank_id'],return_counts=True),s['row_bank_id'][:16]);print('single sweeps',[(k,v.shape,np.min(v),np.max(v)) for k,v in a['single_bank_sweeps'].items()]);print('pulse weights',w.T)
good=np.abs(y).max(axis=1)<9; coef=mods['06']._fit_calibration(x[good],y[good]); er=y[good]-np.column_stack([f(x[good],j)@coef[j] for j in range(4)]); print('nonlinear sweep rmse',np.sqrt(np.mean(er**2,axis=0)),'max',np.abs(er).max(axis=0),'coef',coef.round(5))
