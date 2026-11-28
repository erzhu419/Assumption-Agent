import importlib.util,pickle,numpy as np
from scipy.linalg import solve
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=pickle.load(open(p+'public_batch.pkl','rb'))[0];s=d['study'];nx,ny=s['nx'],s['ny'];n=nx*ny
# Assemble the independently indexed forward model and compare the whole field.
L=np.diag(s['V_reaction_2d'].ravel(order='F'))
for ix in range(nx):
 for iy in range(ny):
  r=ix+nx*iy
  for j,key,stride in [(ix,'x',1),(iy,'y',nx)]:
   k=s['kappa_'+key];v=s['vel_'+key][j];h=s['L'+key]/(len(k)-1)
   for dj in [-1,1]:
    if 0<=j+dj<len(k):
     a=k[j]/h**2 if j in (0,len(k)-1) else (k[j]+k[j+dj])/(2*h**2)+max(-dj*v,0)/h
     L[r,r]+=a;L[r,r+dj*stride]-=a
rhs=np.column_stack([b.ravel(order='F') for b in s['source_modes']]);ij=s['sensor_ij'];rows=ij[:,0]+nx*ij[:,1];fields=[solve(L+mu*np.eye(n),rhs) for mu in s['mu_shifts']];S=np.vstack([f[rows] for f in fields]);Z=s['Z_nuisance'];proj=np.eye(len(Z))-Z@solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
cal=s['sensor_calibration_sweeps'];X=cal['ref_rosette'];Y=cal['raw_rosette'];bank=cal['single_bank_sweeps'];D=np.c_[X,np.ones(len(X))]
def fit(train,fold):
 normal=D[train].T@D[train];rhs=D[train].T@Y[train];B=np.empty((5,4))
 for k in range(4):
  rows=np.flatnonzero(bank['bank_id']==k); rows=rows[np.arange(len(rows))%5!=fold];x=bank['ref_signal'][rows];y=bank['raw_readout'][rows];ix=np.array([k,4]);G=normal.copy();r=rhs[:,k].copy();G[np.ix_(ix,ix)]+=np.array([[x@x,x.sum()],[x.sum(),len(x)]]);r[ix]+=[x@y,y.sum()];B[:,k]=solve(G,r,assume_a='pos')
 return B
worst=np.zeros(3)
for fold,test in enumerate(np.array_split(np.arange(len(X)),5)):
 train=np.setdiff1d(np.arange(len(X)),test); beta=fit(train,fold)
 for pulse in d['campaign_diagnostics']:
  a=solve(beta[:4].T,(pulse['y_pulse_raw'].reshape(-1,4)-beta[4]).T).T.ravel()
  worst=np.maximum(worst,np.sqrt(np.mean((S@pulse['w_pulse']-a).reshape(len(worst),-1)**2,axis=1)))
print('Worst joint-calibration fold and pulse RMSE by shift:',worst);assert max(worst)<1.6e-6
b=fit(np.arange(len(X)),5)
for pulse in d['campaign_diagnostics']:
 obs=pulse['y_pulse_raw'];pred=m._calibrate(dict(s,y_obs_raw=obs));ref=solve(b[:4].T,(obs.reshape(-1,4)-b[4]).T).T.ravel();assert np.max(abs(pred-ref))<1e-12
rng=np.random.default_rng(150);u=s['u_box']; cases=[s]
for total in [0.,u,3*u,4*u,8*u,1e-8,8*u-1e-8,s['total_budget']]:
 for noise in [0.,0.02,0.4]:
  y=S@rng.uniform(0,u,S.shape[1])+Z@rng.normal(size=Z.shape[1])+rng.normal(0,noise,len(Z));raw=(y.reshape(-1,4)@b[:4]+b[4]).ravel();cases.append(dict(s,total_budget=total,y_obs_raw=raw))
worst=np.zeros(6)
for st in cases:
 out=m.solve_batch(st);w=out['weights'];cert=out['kkt_certificate'];A=proj@S;y=proj@solve(b[:4].T,(st['y_obs_raw'].reshape(-1,4)-b[4]).T).T.ravel();res=A@w-y;g=A.T@res+s['source_ridge']*w;nu=cert['nu_sum'];lo=np.array(cert['lam_lower']);hi=np.array(cert['lam_upper']);obj=.5*(res@res+s['source_ridge']*(w@w));expected_field=(fields[-1]@w).reshape(nx,ny,order='F')
 assert np.array_equal(cert['weights'],w) and np.all(lo>=0) and np.all(hi>=0) and np.all(w>=-1e-13) and np.all(w<=u+1e-13)
 worst=np.maximum(worst,[abs(w.sum()-st['total_budget']),np.max(abs(g-nu-lo+hi)),max(np.max(abs(lo*w)),np.max(abs(hi*(u-w)))),abs(obj-cert['dual_lower_bound']),abs(obj-cert['primal_objective']),np.max(abs(expected_field-out['terminal_field']))])
print('Worst budget, stationarity, complementarity, gap, objective, field over',len(cases),'cases:',worst);assert max(worst)<1e-10
