import importlib.util,pickle,numpy as np
from scipy.sparse import diags,eye,kron,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('prev','/tmp/hegel_continual_archive/dev_beta_m100_02_s00_solver.py'); prev=importlib.util.module_from_spec(spec); spec.loader.exec_module(prev)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; ix,iy=eye(nx,format='csc'),eye(ny,format='csc'); I=eye(nx*ny,format='csc'); base=prev._operator(s); D=np.array([p['w_pulse'] for p in pulses]); rhs=(D@np.array(s['source_modes']).transpose(0,2,1).reshape(len(D[0]),-1)).T
sw=s['sensor_calibration_sweeps']; R=sw['ref_rosette']; C=np.linalg.lstsq(np.c_[R,np.ones(len(R))],sw['raw_rosette'],rcond=None)[0]; obs=np.array([np.linalg.solve(C[:4].T,(p['y_pulse_raw'].reshape(-1,4)-C[4]).T).T.ravel() for p in pulses]); ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]
ops=[]; leak=[]
for a in 'xy':
 n=s['n'+a]; h=s['L'+a]/(n-1)
 for i,j,k in [(0,1,n-1),(n-1,n-2,0)]:
  def mk(cols,vals):
   A=csc_matrix((vals,([i]*len(cols),cols)),shape=(n,n)); return kron(iy,A,format='csc') if a=='x' else kron(A,ix,format='csc')
  v=s['robin_'+a]/h; leak.append(v); ops.extend([mk([i],[v]),mk([i,j],[s['kappa_'+a][i]/h**2,-s['kappa_'+a][i]/h**2]),mk([i,k],[1/h**2,-1/h**2])]);
p0=np.tile([1.,1.,0.],(4,1))
def forward(p):
 A=base.copy()
 for B,v in zip(ops,(p-p0).ravel()):
  if v: A=A+v*B
 return np.concatenate([splu(A+t*I).solve(rhs)[rows] for t in s['mu_shifts']]).T

def report(name,p):
 e=forward(p)-obs; print(name,'p=',np.round(p,6).tolist(),'pulse rms',np.round(np.mean(e**2,axis=1)**.5,8),'shift rms',np.round(np.mean(e.reshape(3,3,-1)**2,axis=(0,2))**.5,8),'max',abs(e).max(),flush=True)
if __name__=='__main__':
 for name,p in [('Robin',p0),('Neumann',np.tile([0.,1.,0.],(4,1))),('periodic',np.tile([0.,1.,1.],(4,1)))]: report(name,p)
 for cols in ([0],[0,1],[0,2],[0,1,2]):
  shape=(4,len(cols)); lb=np.tile({0:-.95,1:.05,2:0}[cols[0]],shape).astype(float)
  for j,c in enumerate(cols):lb[:,j]={0:-.95,1:.05,2:0}[c]
  def unpack(x):
   p=p0.copy(); p[:,cols]=x.reshape(shape); return p
  for held in (-1,0,1,2):
   use=np.arange(len(obs))!=held
   sol=least_squares(lambda x:(forward(unpack(x))-obs)[use].ravel(),np.maximum(p0[:,cols],lb+1e-5).ravel(),bounds=(lb.ravel(),np.inf),xtol=1e-11,gtol=1e-12)
   report((cols,held),unpack(sol.x))
