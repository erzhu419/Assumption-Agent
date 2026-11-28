import importlib.util, pickle, itertools, numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
for item in items:
 s=item['study']; out=m.solve_batch(s); w=out['weights']; c=out['kkt_certificate']; nx,ny=s['nx'],s['ny']; modes=s['source_modes']; Z=s['Z_nuisance']; cap=s['u_box']; budget=s['total_budget']
 def axis(n,L,k,v):
  h=L/(n-1); A=np.zeros((n,n))
  for i in range(n):
   for step in [-1,1]:
    j=(i+step)%n; rate=(k[i]+k[j])/(2*h*h)+max(-step*v[i],0)/h
    A[i,j]=-rate; A[i,i]+=rate
  return A
 A=np.kron(np.eye(ny),axis(nx,s['Lx'],s['kappa_x'],s['vel_x']))+np.kron(axis(ny,s['Ly'],s['kappa_y'],s['vel_y']),np.eye(nx))+np.diag(s['V_reaction_2d'].flatten(order='F'))
 idx=s['sensor_ij']; fields=[np.linalg.solve(A+mu*np.eye(nx*ny),np.column_stack([u.flatten(order='F') for u in modes])) for mu in s['mu_shifts']]
 obs=np.vstack([u[idx[:,0]+nx*idx[:,1],:] for u in fields]); expected=(fields[-1]@w).reshape(nx,ny,order='F'); ferr=np.max(abs(expected-out['terminal_field']))
 y=m.calibrate(s); raw=m.raw_map(y.reshape(-1,4),m.calibration_map(s['sensor_calibration_sweeps'])).ravel(); ierr=np.max(abs(raw-s['y_obs_raw'])); ridge=s['source_ridge']; P=np.eye(len(y))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); S=P@obs; t=P@y; r=S@w-t; g=S.T@r+ridge*w
 lo,hi=np.array(c['lam_lower']),np.array(c['lam_upper']); J=.5*(r@r+ridge*(w@w)); gap=J-c['dual_lower_bound']; stat=np.max(abs(g-c['nu_sum']-lo+hi)); comp=max(np.max(abs(w*lo)),np.max(abs((cap-w)*hi)))
 assert np.allclose(c['weights'],w,rtol=0,atol=1e-13) and min(w)>=-1e-12 and max(w)<=cap+1e-12 and abs(sum(w)-budget)<1e-12
 assert min(lo)>=0 and min(hi)>=0 and abs(c['primal_objective']-J)<1e-12 and max(ferr,ierr,abs(gap),stat,comp)<1e-9
 print('field / calibration inversion / dual gap / stationarity / complementarity:',ferr,ierr,gap,stat,comp)
 # Refit on each training shift/rosette block to compare prediction errors away from fitted sensors.
 groups=[np.repeat(np.arange(len(fields)),len(idx)),np.tile(np.arange(len(idx))//4,len(fields))]
 for group in groups:
  errors=[]
  for k in np.unique(group):
   ix=group!=k; z=Z[ix]; a=obs[ix]; tt=y[ix]; q=np.linalg.qr(z,mode='reduced')[0]; b=a-q@(q.T@a); d=tt-q@(q.T@tt); ww,_=m.box_simplex(b.T@b+1e-10*np.eye(len(w)),b.T@d,cap,budget); gamma=np.linalg.lstsq(z,tt-a@ww,rcond=None)[0]; errors.append(np.sqrt(np.mean((obs[~ix]@ww+Z[~ix]@gamma-y[~ix])**2)))
  print('chamber holdout pooled/worst RMSE:',np.sqrt(np.mean(np.square(errors))),max(errors))
 # Exercise the active set at vertices and at both budget endpoints.
 rng=np.random.default_rng(13); worst=0.
 for b in [0.,.01,cap,3*cap,budget,(len(w)-.01)*cap,len(w)*cap]:
  for _ in range(15):
   R=rng.normal(size=(len(w),len(w))); H=R.T@R+.04*np.eye(len(w)); c0=3*rng.normal(size=len(w)); ww,nu=m.box_simplex(H,c0,cap,b); gg=H@ww-c0; dd=gg-nu; eg=dd@ww-cap*np.minimum(dd,0).sum(); worst=max(worst,abs(eg)); assert abs(eg)<1e-9 and abs(sum(ww)-b)<1e-10 and min(ww)>=0 and max(ww)<=cap
 print('active-set stress worst dual gap:',worst)
