import importlib.util,pickle,numpy as np
from scipy.sparse import diags,eye,kron,lil_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/beta_m100_c01_s00_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=i['study']; nx,ny=st['nx'],st['ny']; ij=st['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]; ds=i['campaign_diagnostics']
w=np.array([d['w_pulse'] for d in ds]).T
Y=np.column_stack([prior._calibrate(dict(st,y_obs_raw=d['y_pulse_raw'])) for d in ds]); Z=st['Z_nuisance']; Q=np.linalg.qr(Z)[0]
rhs=np.asfortranarray(st['source_modes'].reshape(len(w),-1,order='F').T@w)
axes=[prior._axis(st['kappa_'+a],st['vel_'+a],st['L'+a]/(n-1),st['robin_'+a]) for a,n in [('x',nx),('y',ny)]]
reaction=diags(st['V_reaction_2d'].ravel(order='F'),format='csc'); identity=eye(nx*ny,format='csc')
def pred(rows=None):
 out=[]
 for j,axis in enumerate(axes):
  a=axis.copy().tolil()
  if rows is not None:
   for k,row in enumerate([0,a.shape[0]-1]):
    step=1-2*k
    sm,adj,wrap,second=rows[2*j+k]; a[row,row]=sm+adj+wrap+second; a[row,row+step]=-adj; a[row,a.shape[0]-1-row]=-wrap; a[row,row+2*step]=-second
  out.append(a.tocsc())
 ax,ay=out
 L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+reaction
 return np.vstack([splu(L+mu*identity).solve(rhs)[slots] for mu in st['mu_shifts']])
def errs(d): return np.sqrt(np.mean(d*d,axis=0))
base=np.array([[st['robin_'+a]/h,k/h**2,0,0] for a in ('x','y') for h in [st['L'+a]/(st['n'+a]-1)] for k in st['kappa_'+a][[0,-1]]]); print('base coefficients',base)
def resid(rows,prof=False):
 d=pred(rows)-Y
 return d-Q@(Q.T@d) if prof else d
print('baseline RMS',errs(resid(base)), 'profiled',errs(resid(base,True)))
for sm,adj,wrap in [(0,1,0),(1,1,1)]:
 a=base.copy(); a[:,0]*=sm; a[:,1]*=adj; a[:,2]=base[:,1]*wrap
 print('candidate',sm,adj,wrap,errs(resid(a)))
for prof in (False,True):
 for cols in ([0],[0,1],[0,1,2],[0,1,2,3]):
  def unpack(p):
   a=base.copy(); a[:,cols]=p.reshape(4,-1); return a
  fit=least_squares(lambda p:resid(unpack(p),prof).ravel(),base[:,cols].ravel(),max_nfev=150)
  a=unpack(fit.x); print('fit',prof,cols,errs(resid(a,prof)),np.round(a,5), 'cost',fit.cost)
print('pulse raw calibrated minima',Y.min(axis=0)); print('baseline minima',pred().min(axis=0)); print('sensor coverage',ij.tolist())
