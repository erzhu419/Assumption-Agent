import importlib.util, pickle
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('prev','/tmp/hegel_continual_archive/dev_beta_m100_01_s00_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; ps=item['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in ps]); y=np.stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in ps]); nx,ny=s['nx'],s['ny']; L=m._operator(s); ix=np.arange(len(ps));
# Boundary corrections are in units of local wall conductance for conservative
# couplings and local Robin impedance for the row sum.
def corrections(grammar):
 mats=[]
 for a,n in [('x',nx),('y',ny)]:
  h=s['L'+a]/(n-1); k=s['kappa_'+a]; r=s['robin_'+a]
  for i,step in [(0,1),(n-1,-1)]:
   for j in [None,i+step]+([i+2*step] if 'second' in grammar else [])+([n-1-i] if 'wrap' in grammar else []):
    C=sparse.lil_matrix((n,n)); C[i,i]=r/h if j is None else k[i]/h**2
    if j is not None: C[i,j]=-C[i,i]
    mats.append(sparse.kron(sparse.eye(ny),C,format='csc') if a=='x' else sparse.kron(C,sparse.eye(nx),format='csc'))
 return mats

def run(grammar,nuisance=False,cv=False):
 mats=corrections(grammar)
 def pred(p):
  B=L+sum((v*a for v,a in zip(p,mats)),sparse.csc_matrix(L.shape)); return W@m._forward(s,B)[0].T
 def resid(p):
  E=pred(p)-y
  if nuisance:
   Z=s['Z_nuisance']; E=E-np.linalg.lstsq(Z,E.T,rcond=None)[0].T@Z.T
  return E
 def summary(e): return np.round(np.sqrt(np.mean(e.reshape(3,3,-1)**2,axis=2)),8)
 p0=np.zeros(len(mats)); p=least_squares(lambda p:resid(p).ravel(),p0,gtol=1e-13,xtol=1e-13,ftol=1e-13,max_nfev=100,bounds=(-.995,15.)).x
 print(grammar,nuisance,'p',np.round(p,7),'RMSE by pulse/shift',summary(resid(p)).tolist())
 if cv:
  for i in ix:
   ph=least_squares(lambda p:resid(p)[ix!=i].ravel(),p0,gtol=1e-13,xtol=1e-13,ftol=1e-13,max_nfev=100,bounds=(-.995,15.)).x
   print('held pulse',i,'p',np.round(ph,6),'RMSE',summary(resid(ph))[i])
 return p
if __name__=='__main__':
 E=W@m._forward(s)[0].T-y
 print('baseline RMSE by pulse/shift',np.sqrt(np.mean(E.reshape(3,3,-1)**2,axis=2)))
 run('nearest',False,True); run('nearest',True); run('second',False); run('wrap',False); run('second+wrap',False)
