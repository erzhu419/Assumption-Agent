import pickle,numpy as np,glob,importlib.util
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=item['study']; sw=st['sensor_calibration_sweeps']; x,y=sw['ref_rosette'],sw['raw_rosette']; X=np.c_[np.ones(len(x)),x]; B=np.linalg.lstsq(X,y,rcond=None)[0]; print('affine',B,'sweep rms',np.mean((X@B-y)**2,axis=0)**.5); print('banks',st['row_bank_id'][:24]); print('single',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in sw['single_bank_sweeps'].items()]); print('aux',st['aux_manifold_telemetry'][:4]);
for f in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 sp=importlib.util.spec_from_file_location('m',f); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); S,U=m._forward(st); print(f)
 for p in item['campaign_diagnostics']:
  r=m._calibrate(dict(st,y_obs_raw=p['y_pulse_raw']))-S@p['w_pulse']; Z=st['Z_nuisance']; q=r-Z@np.linalg.lstsq(Z,r,rcond=None)[0]; print('rms',np.linalg.norm(r)/len(r)**.5,np.linalg.norm(q)/len(r)**.5,'by shift',np.mean(r.reshape(3,-1)**2,axis=1)**.5)
