import importlib.util,pickle,numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m.calibrate(s); sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; off,g,c,L,R,C=m.calibration_map(sw); nxt=np.roll(x,-1,axis=1); e=abs(sw['raw_rosette']-(off+g*x+c*x**3+L*np.roll(x,1,axis=1)+R*nxt+C*x*nxt)); print('calibration coefficients',m.calibration_map(sw).round(6),'max clean',e[e<1].max(),'spikes',e[e>=1]); print('calibrated range',y.min(),y.max())
a=np.repeat([s['robin_x'],s['robin_y']],2); Z=s['Z_nuisance']; blocks=np.arange(len(y))//len(s['sensor_ij']); sensors=np.arange(len(y))%len(s['sensor_ij']);
for name,loss in [('sealed',a*0),('robin',a),('x-only',a*[1,1,0,0]),('y-only',a*[0,0,1,1]),('half',a*.5)]:
 S,U=m.chamber_response(s,loss); w,cert=m._inverse(s,S,y); print(name,'objective and gap',cert['primal_objective'],cert['primal_objective']-cert['dual_lower_bound'],'w',w.round(5));
 for labels in (blocks,sensors//4):
  rmse=[]
  for j in np.unique(labels):
   test=labels==j; train=~test; st={**s,'Z_nuisance':Z[train]}; ww,_=m._inverse(st,S[train],y[train]); G=Z[train]; gamma=np.linalg.solve(G.T@G+s['nuisance_ridge']*np.eye(G.shape[1]), G.T@(y[train]-S[train]@ww)); rmse.append(np.mean((S[test]@ww+Z[test]@gamma-y[test])**2)**.5)
  print('  CV rmse',np.round(rmse,6),'mean',np.mean(rmse),'worst',max(rmse))
