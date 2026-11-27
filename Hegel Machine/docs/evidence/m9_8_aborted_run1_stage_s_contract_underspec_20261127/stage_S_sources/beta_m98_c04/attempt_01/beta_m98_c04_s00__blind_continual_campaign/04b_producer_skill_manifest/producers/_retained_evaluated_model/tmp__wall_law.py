import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('b','/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ch=b._Chamber(s); c=b._calibrator(s['sensor_calibration_sweeps']); y=c(s['y_obs_raw']); fit=b._fit_walls(ch,s,c,y)
wall=np.zeros(8)
for j,a in enumerate('xy'):
 k=np.asarray(s['kappa_'+a]); v=np.asarray(s['vel_'+a]); h=s['L'+a]/(len(k)-1); ix=slice(2*j,2*j+2)
 diffusion=(k[[1,-2]]-k[[0,-1]])/(2*h*h)
 wall[ix]=ch.prior[ix]-diffusion-np.abs(v[[0,-1]])/h
 wall[4+2*j:6+2*j]=-k[[0,-1]]/h**2-ch.inward[ix]
print('inferred wall law coefficients:',wall,'max parameter error',np.max(abs(wall-fit)))
W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Y=np.column_stack([c(p['y_pulse_raw']) for p in s['reference_tracer_pulses']]); e=ch.response(wall,ch.F@W)[0]-Y
print('no-fit shift RMS',np.sqrt(np.mean(e.reshape(len(ch.shifts),-1)**2,axis=1)))
P=b._projector(s['Z_nuisance'],s['nuisance_ridge']); S,U,_=ch.response(wall); X=P(S); u=np.broadcast_to(s['u_box'],S.shape[1]); w,_=b._box_qp(X.T@X+s['source_ridge']*np.eye(len(u)),-X.T@P(y),u,s['total_budget']); o=b.solve_batch(s)
print('no-fit weight and field relative errors',np.linalg.norm(w-o['weights'])/np.linalg.norm(o['weights']),np.linalg.norm(U@w-o['terminal_field'].ravel(order='F'))/np.linalg.norm(o['terminal_field']))
