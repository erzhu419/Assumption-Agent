import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fix=m._calibration(s['sensor_calibration_sweeps']); R,F,a0=m._chamber(s); w0=m.solve_batch(s); cert=w0['kkt_certificate']; a=m._fit_walls(s,fix,R,F,a0); S,U=R(a); T=U.transpose(1,2,0).reshape(s['nx']*s['ny'],-1,order='F'); assert np.allclose(T[s['sensor_ij'][:,0]+s['nx']*s['sensor_ij'][:,1]], S[-len(s['sensor_ij']):],atol=1e-14); print('public certificate gap',cert['primal_objective']-cert['dual_lower_bound'],'weights',w0['weights'])
rng=np.random.default_rng(4); worst=np.zeros(4)
for i in range(600):
 n=int(rng.integers(2,32)); D=rng.normal(size=(n,n)); H=D.T@D+10**rng.uniform(-4,0)*np.eye(n); c=rng.normal(size=n)*10**rng.uniform(-2,2); u=rng.uniform(.05,1,n); u[rng.random(n)<.15]=0
 if i%7==0: u[:]=float(rng.choice([.25,.5,1.]));
 b=[0.,u.sum(),u[:n//2].sum(),rng.uniform(0,u.sum())][i%4]; w,nu,l,h=m._box_simplex(H,c,u,b); e=H@w+c+nu-l+h; worst=np.maximum(worst,[max(abs(w.sum()-b),max(-w),max(w-u)),max(abs(e)),max(max(abs(w*l)),max(abs((u-w)*h))),abs(nu*(w.sum()-b)-l@w+h@(w-u)-.5*e@np.linalg.solve(H,e))]); assert min(l)>=0 and min(h)>=0
print('stress worst feasibility/stationarity/complementarity/gap',worst); assert max(worst)<1e-8
Z=s['Z_nuisance']; y=fix(s['y_obs_raw']); u=np.broadcast_to(s['u_box'],(F.shape[1],)); b=s['total_budget']; ridge=s['source_ridge']; ns=len(s['sensor_ij']); pulses=s['reference_tracer_pulses']; B=F@np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([fix(p['y_pulse_raw']) for p in pulses]); I=np.eye(len(u)); G=np.repeat(np.eye(2),2,axis=0)
def profile(walls,mask):
 sm=R(walls)[0][mask]; z=Z[mask]; yz=np.column_stack((sm,y[mask])); zz=z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]); x,r=(yz-z@np.linalg.solve(zz,z.T@yz))[:,:-1],(yz-z@np.linalg.solve(zz,z.T@yz))[:,-1]; w=m._box_simplex(x.T@x+ridge*I,-x.T@r,u,b)[0]; g=np.linalg.solve(zz,z.T@(y[mask]-sm@w)); return w,g
rows=np.arange(len(y)); tests=[(rows%ns//4)%4==i for i in range(4)]+[rows//ns==i for i in range(len(s['mu_shifts']))]; allrows=np.ones(len(y),bool)
for name,g,prior in [('nominal',G,0),('axis',G,.1),('axis',G,.01),('split',np.eye(4),.01),('axis',G,0.)]:
 def wallfit(mask):
  if name=='nominal': return a0
  def f(delta):
   walls=a0+g@delta; w,gamma=profile(walls,mask); return np.r_[(R(walls)[0]@w+Z@gamma-y)[mask],prior*delta]
  return a0+g@least_squares(f,np.zeros(g.shape[1]),bounds=(-1.5,1.5),max_nfev=32).x
 aq=wallfit(allrows); out=[]
 for mask in tests:
  aa=wallfit(~mask); w,gamma=profile(aa,~mask); out.append(np.sqrt(np.mean((R(aa)[0]@w+Z@gamma-y)[mask]**2)))
 e=(R(aq,B)[0]-Y).reshape(len(s['mu_shifts']),ns,-1); print('blind',name,prior,'walls',np.round(aq,4),'worst routine CV',max(out),'worst independent tracer shift',max(np.sqrt(np.mean(e**2,axis=(1,2)))))
