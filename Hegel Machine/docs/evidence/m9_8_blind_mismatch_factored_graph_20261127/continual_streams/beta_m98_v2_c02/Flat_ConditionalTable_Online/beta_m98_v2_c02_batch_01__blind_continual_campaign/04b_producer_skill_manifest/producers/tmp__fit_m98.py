import glob,importlib.util,pickle,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p=glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]
spec=importlib.util.spec_from_file_location('prev',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=d['study']; nx,ny=s['nx'],s['ny']; Z=s['Z_nuisance']; I=sparse.eye(nx*ny,format='csc')
base=[]; boundary=[]
for axis,n in [('x',nx),('y',ny)]:
 h=s['L'+axis]/(n-1); k=s['kappa_'+axis]; rob=s['robin_'+axis]; a=m._axis(k,s['vel_'+axis],h,rob).toarray(); a[[0,-1],:]=0; base.append(a); boundary.append((k/h**2,rob/h))
react=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); ii,jj=s['sensor_ij'].T; rows=ii+nx*jj
Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in d['campaign_diagnostics']]); W=np.array([p['w_pulse'] for p in d['campaign_diagnostics']]); F=(W@s['source_modes'].reshape(-1,nx*ny,order='F')).T

def predict(p):
 axes=[]
 for a,(k,r),q in zip(base,boundary,p.reshape(2,2,3)):
  a=a.copy()
  for i,j,l,(diff,leak,wrap) in zip([0,-1],[1,-2],[-1,0],q):
   a[i,i]=k[i]*(diff+wrap)+leak*r; a[i,j]=-k[i]*diff; a[i,l]=-k[i]*wrap
  axes.append(sparse.csc_matrix(a))
 L=sparse.kron(sparse.eye(ny),axes[0],format='csc')+sparse.kron(axes[1],sparse.eye(nx),format='csc')+react
 return np.vstack([splu(L+mu*I).solve(F)[rows] for mu in s['mu_shifts']])

def fit(p,cols):
 def residual(q):
  e=(predict(q)-Y)[:,cols]; return e.ravel()
 sol=least_squares(residual,p,bounds=(0,np.inf),gtol=1e-12,xtol=1e-12,ftol=1e-12,max_nfev=200)
 print('fit cols',cols,'params',sol.x.reshape(2,2,3),'pulse rms',np.sqrt(np.mean((predict(sol.x)-Y)**2,axis=0)), 'optimality',sol.optimality)
 return sol.x
p=np.tile([1.,1.,0.],4)
for name,q in [('open',p),('reflect',np.tile([1.,0.,0.],4)),('halfcell-reflect',np.tile([2.,0.,0.],4)),('periodic',np.tile([1.,0.,1.],4))]:
 e=predict(q)-Y; print(name,'rms',np.sqrt(np.mean(e*e,axis=0)),'nuisance-adjusted',np.sqrt(np.mean((e-Z@np.linalg.lstsq(Z,e,rcond=None)[0])**2,axis=0)))
for cols in ([0],[1],[0,1]):fit(p,cols)
