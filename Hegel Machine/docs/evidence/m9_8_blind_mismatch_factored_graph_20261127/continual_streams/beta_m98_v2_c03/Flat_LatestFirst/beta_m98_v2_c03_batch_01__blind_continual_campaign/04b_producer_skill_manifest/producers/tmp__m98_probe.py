import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import eye,kron,diags,csc_matrix
from scipy.sparse.linalg import splu
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/beta_m98_v2_c03_s00_solver.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; nx,ny=s['nx'],s['ny']; X=[]; scales=[]
for k,n in [('x',nx),('y',ny)]:
 h=s['L'+k]/(n-1); kap=s['kappa_'+k]; X.append(old._axis(kap,s['vel_'+k],h,s['robin_'+k]).toarray()); scales.extend([(kap[0]/h**2,kap[-1]/h**2,s['robin_'+k]/h)])
print('boundary scales kap/h2, rob/h',scales); print('boundary velocity',s['vel_x'][[0,-1]],s['vel_y'][[0,-1]])
w=np.array([p['w_pulse'] for p in a['campaign_diagnostics']]); obs=np.array([old._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in a['campaign_diagnostics']]).T; rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']])@w.T
ix,iy=s['sensor_ij'].T; sensor=ix+nx*iy; I=eye(nx*ny,format='csc'); react=diags(s['V_reaction_2d'].ravel(order='F')); Z=s['Z_nuisance']; Q=np.linalg.qr(Z,mode='reduced')[0];
def predict(p):
 A=[]
 for i,(M,b) in enumerate(zip(X,scales)):
  M=M.copy(); kl,kr,r=b; bl,br,rl,rr=p[4*i:4*i+4]; M[0,0]=kl*bl+rl*r; M[-1,-1]=kr*br+rr*r; M[0,1]=-kl*bl; M[-1,-2]=-kr*br; A.append(csc_matrix(M))
 L=kron(eye(ny),A[0],format='csc')+kron(A[1],eye(nx),format='csc')+react
 return np.vstack([splu(L+mu*I).solve(rhs)[sensor] for mu in s['mu_shifts']])
def resid(p,nuis=False):
 d=predict(p)-obs
 return d-Q@(Q.T@d) if nuis else d
def rms(d): return np.sqrt(np.mean(d*d,axis=0))
for name,p in [('open',[1]*8),('reflect',[1,1,0,0]*2),('double diffusion',[2,2,1,1]*2),('double wall',[1,1,2,2]*2)]: print(name,'rms',rms(resid(p)),'profiled',rms(resid(p,True)))
for nuis in (False,True):
 for cols,test in [(slice(None),slice(None)),(slice(0,1),slice(1,2)),(slice(1,2),slice(0,1))]:
  r=least_squares(lambda p:resid(p,nuis)[:,cols].ravel(),np.ones(8),bounds=(0,np.inf),gtol=1e-13,xtol=1e-13,ftol=1e-13); print('fit',nuis,cols,np.round(r.x,7),'train/held',rms(resid(r.x,nuis)[:,cols]),rms(resid(r.x,nuis)[:,test]),'sv',np.round(np.linalg.svd(r.jac,compute_uv=False),6))
