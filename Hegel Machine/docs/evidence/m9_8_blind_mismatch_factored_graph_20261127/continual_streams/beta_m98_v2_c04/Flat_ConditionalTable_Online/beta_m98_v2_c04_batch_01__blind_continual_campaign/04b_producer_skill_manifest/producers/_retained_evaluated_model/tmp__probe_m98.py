import sys,pickle,importlib.util,numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/beta_m98_v2_c04_s00_solver.py';spec=importlib.util.spec_from_file_location('old',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; ds=it['campaign_diagnostics']; Y=np.column_stack([old._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds]); W=np.column_stack([d['w_pulse'] for d in ds]); nx,ny=s['nx'],s['ny']; N=nx*ny; dims=(nx,ny); axes=[old._axis(s['kappa_'+a],s['vel_'+a],s['L'+a],s['robin_'+a]) for a in 'xy']; I=eye(N,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=s['source_modes'].transpose(0,2,1).reshape(-1,N).T@W; si,sj=s['sensor_ij'].T; idx=si+nx*sj
z=s['Z_nuisance']; Q=np.linalg.qr(z)[0];
def forward(theta):
 aa=[]
 for a,n,t in zip(axes,dims,np.asarray(theta).reshape(2,4)):
  a=a.tolil(); a[0,0]+=t[0]; a[0,1]+=t[1]; a[-1,-1]+=t[2]; a[-1,-2]+=t[3]; aa.append(a.tocsc())
 L=kron(eye(ny),aa[0],format='csc')+kron(aa[1],eye(nx),format='csc')+V
 return np.vstack([splu(L+m*I).solve(rhs)[idx] for m in s['mu_shifts']]);
def stats(r): return np.sqrt(np.mean(r**2,axis=0)), np.max(abs(r),axis=0)
zero=np.zeros(8); r0=forward(zero)-Y; print('baseline pulse rms, max',stats(r0),'projected',stats(r0-Q@(Q.T@r0))); print('baseline axes',[(a[0,:2].toarray().ravel(),a[-1,-2:].toarray().ravel()) for a in axes]); print('h', [s['L'+a]/(s['n'+a]-1) for a in 'xy']); print('endpoints',[(s['kappa_'+a][[0,-1]],s['vel_'+a][[0,-1]],s['robin_'+a]) for a in 'xy']); print('archive forward parity',np.max(abs(old._forward(s)[0]@W-forward(zero))));
for proj in (False,True):
 def err(t):
  e=forward(t)-Y;return e-Q@(Q.T@e) if proj else e
 for tr,te in [(slice(None),slice(None)),(0,1),(1,0)]:
  fit=least_squares(lambda t:err(t)[:,tr].ravel(),zero,xtol=1e-12,ftol=1e-12,gtol=1e-12); print('project, train, test',proj,tr,te,'delta',np.round(fit.x,5),'fit',stats(err(fit.x)),'test raw',stats((forward(fit.x)-Y)[:,te]));
