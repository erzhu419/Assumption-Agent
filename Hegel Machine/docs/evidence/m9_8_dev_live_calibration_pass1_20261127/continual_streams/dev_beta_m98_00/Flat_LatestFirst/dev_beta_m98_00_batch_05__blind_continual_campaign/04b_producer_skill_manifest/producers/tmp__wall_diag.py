import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares,lsq_linear
p=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_beta_m98_00_batch_04_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibration(s['sensor_calibration_sweeps'])(s['y_obs_raw']); z=s['Z_nuisance']; src=s['source_modes']; cap=s['u_box']; total=s['total_budget']; n=len(src); nz=z.shape[1]; allrows=np.ones(len(y),bool); folds=np.arange(len(y))//4%(len(s['sensor_ij'])//4)%4
print('pulses',type(s['reference_tracer_pulses']), 'shifts',s['mu_shifts'],'obs',len(y),'nuisance',z.shape,'noise context',s['nuisance_ridge'])
bounds=(np.r_[np.zeros(n),np.full(nz,-np.inf)],np.r_[np.full(n,cap),np.full(nz,np.inf)])
def fit(groups, mask=allrows):
 def fun(x,full=False):
  wall=np.ones(8)
  for g,v in zip(groups,x): wall[list(g)]=np.exp(v)
  a=np.c_[m._response(s,wall,src)[0],z]; b=a[mask]; goal=y[mask]
  q=lsq_linear(np.vstack((b,10*np.r_[np.ones(n),np.zeros(nz)])),np.r_[goal,10*total],bounds=bounds,method='bvls',tol=1e-11).x
  e=a@q-y
  return (wall,e,q) if full else e[mask]
 x=np.zeros(len(groups));
 if groups:x=least_squares(fun,x,bounds=(-4,3),max_nfev=70,gtol=1e-10,ftol=1e-9).x
 return fun(x,True)
F={'base':(), 'all_robin':((0,1,4,5),),'all_diff':((2,3,6,7),),'axis_robin':((0,1),(4,5)),'axis_diff':((2,3),(6,7)),'each_robin':((0,),(1,),(4,),(5,)),'each_diff':((2,),(3,),(6,),(7,)),'axis_both':((0,1),(2,3),(4,5),(6,7)),'left_right':((0,4),(1,5),(2,6),(3,7)),'all':tuple((j,) for j in range(8))}
for name,g in F.items():
 w,r,q=fit(g); cv=[np.mean(fit(g,folds!=k)[1][folds==k]**2)**.5 for k in range(4)];print(name,'train',round(np.mean(r*r)**.5,7),'cv',np.round(cv,7),'wall',np.round(w,4),'source',np.round(q[:n],3),flush=True)
