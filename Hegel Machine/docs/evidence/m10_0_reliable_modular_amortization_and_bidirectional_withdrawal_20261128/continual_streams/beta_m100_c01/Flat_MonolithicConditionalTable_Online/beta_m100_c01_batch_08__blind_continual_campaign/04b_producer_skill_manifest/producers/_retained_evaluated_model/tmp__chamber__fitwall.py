import json,pickle,runpy
import numpy as np
from scipy.sparse import csc_matrix,diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/beta_m100_c01_batch_03_solver.py')
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; ps=it['campaign_diagnostics']; w=np.array([p['w_pulse'] for p in ps]); raw=np.array([p['y_pulse_raw'] for p in ps]); obs=np.array([m['_calibrate'](dict(s,y_obs_raw=r)) for r in raw]); nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; I=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=(w@s['source_modes'].transpose(0,2,1).reshape(len(w[0]),-1)).T

def axis(a,p):
 k=s['kappa_'+a]; v=s['vel_'+a]; n=len(k); h=s['L'+a]/(n-1); A=m['_axis'](k,v,h,0.).toarray();
 for j,(i,b,o) in enumerate([(0,1,n-1),(n-1,n-2,0)]):
  inward,wrap,leak=p[j]; A[i,b]=-inward*k[i]/h**2; A[i,o]=-wrap*.5*(k[i]+k[o])/h**2; A[i,i]=-A[i,b]-A[i,o]+leak/h
 return csc_matrix(A)

def predict(p):
 A=kron(eye(ny),axis('x',p[:2]),format='csc')+kron(axis('y',p[2:]),eye(nx),format='csc')+V
 return np.concatenate([splu(A+mu*I).solve(rhs)[idx].T for mu in s['mu_shifts']],axis=1)

if __name__=='__main__':
 for mode in ['separate','paired','wraponly','inwardonly']:
  base=np.tile([1.,0.,0.],(4,1));
  def expand(v):
   p=base.copy()
   if mode=='separate': return v.reshape(4,3)
   if mode=='paired': return np.repeat(v.reshape(2,3),2,axis=0)
   p[:,1 if mode=='wraponly' else 0]=v
   return p
  for hold in [-1,0,1,2]:
   use=np.arange(3)!=hold; x0={'separate':base.ravel(),'paired':base[::2].ravel(),'wraponly':base[:,1],'inwardonly':base[:,0]}[mode]
   fit=least_squares(lambda v:(predict(expand(v))-obs)[use].ravel(),x0,ftol=1e-12,gtol=1e-12,xtol=1e-12,max_nfev=80); r=predict(expand(fit.x))-obs
   print(mode,hold,'p',np.round(expand(fit.x).ravel(),5),'rmse',np.sqrt(np.mean(r*r,axis=1)).round(8),'jac rank',np.linalg.matrix_rank(fit.jac))
 d=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('telemetry metadata',[(k,str(v)[:400]) for k,v in d.items() if not isinstance(v,list)] if isinstance(d,dict) else type(d)); print('gate',open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json').read()[:3000])
