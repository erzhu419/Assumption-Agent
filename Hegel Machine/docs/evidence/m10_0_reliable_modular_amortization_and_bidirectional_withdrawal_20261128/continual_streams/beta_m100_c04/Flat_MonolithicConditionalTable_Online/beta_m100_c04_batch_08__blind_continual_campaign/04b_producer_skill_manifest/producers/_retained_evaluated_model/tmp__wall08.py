import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('mod','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibrate(s); Z=s['Z_nuisance']; cap,budget=s['u_box'],s['total_budget']; ij=np.tile(s['sensor_ij'],(len(s['mu_shifts']),1)); block=np.arange(len(y)).reshape(3,-1,4)
masks=[np.isin(np.arange(len(y)),block[:,k::4,:]) for k in range(4)]+[ij[:,a]==v for a,v in [(0,0),(0,s['nx']-2),(1,0),(1,s['ny']-2)]]
for laws in [('periodic','periodic'),('periodic',(0.,0.)),((0.,0.),'periodic'),((0.,0.),(0.,0.))]:
 S,_=m._forward(s,laws); rows=[]
 for ridge,zridge in [(s['source_ridge'],s['nuisance_ridge']),(1e-10,1e-10)]:
  ev=[]
  for mask in [np.zeros(len(y),bool)]+masks:
   tr=~mask; a=m._project(Z[tr],zridge,np.column_stack([S[tr],y[tr]])); w,_=m._qp(a[:,:-1],a[:,-1],ridge,cap,budget)
   g=np.linalg.solve(Z[tr].T@Z[tr]+zridge*np.eye(Z.shape[1]),Z[tr].T@(y[tr]-S[tr]@w)); val=mask if mask.any() else tr
   ev.append(np.mean((S[val]@w+Z[val]@g-y[val])**2)**.5)
  rows.append(np.round(ev,7))
 print(laws,*rows)
