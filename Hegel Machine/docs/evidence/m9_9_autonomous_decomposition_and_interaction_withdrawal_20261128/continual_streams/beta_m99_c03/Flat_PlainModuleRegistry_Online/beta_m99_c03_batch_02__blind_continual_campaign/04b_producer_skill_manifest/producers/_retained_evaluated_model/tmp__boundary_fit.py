import json,pickle,sys
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
I=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; S=I['study']; P=I['campaign_diagnostics']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
Y=np.array([calibrate_upstream(dict(S,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in P]).T
W=np.array([p['w_pulse'] for p in P]).T
nx,ny=S['nx'],S['ny']; ix,iy=S['sensor_ij'].T; idx=ix+nx*iy
Z=S['Z_nuisance']; Q=np.linalg.qr(Z)[0]; rhs=np.column_stack([v.ravel(order='F') for v in S['source_modes']])@W

def axis(a,p):
 k,v=S['kappa_'+a],S['vel_'+a]; h=S['L'+a]/(len(k)-1); n=len(k); face=(k[:-1]+k[1:])/(2*h*h)
 lo=np.r_[-face[:-1]-np.maximum(v[1:-1],0)/h,-k[-1]/h**2*p[3]]; hi=np.r_[-k[0]/h**2*p[2],-face[1:]-np.maximum(-v[1:-1],0)/h]; d=np.r_[-hi[0]+p[0]*S['robin_'+a]/h+p[4]*face[0],-lo[:-1]-hi[1:],-lo[-1]+p[1]*S['robin_'+a]/h+p[5]*face[-1]]
 mat=sp.diags([lo,d,hi],[-1,0,1],format='lil');mat[0,n-1]=-p[4]*face[0];mat[n-1,0]=-p[5]*face[-1];return mat.tocsc()
def predict(p):
 ax,ay=axis('x',p[:6]),axis('y',p[6:]); e=sp.eye(nx*ny,format='csc'); l=sp.kron(sp.eye(ny),ax,format='csc')+sp.kron(ay,sp.eye(nx),format='csc')+sp.diags(S['V_reaction_2d'].ravel(order='F'),format='csc');return np.concatenate([splu(l+mu*e).solve(rhs)[idx] for mu in S['mu_shifts']])
def res(p,proj=False):
 d=predict(p)-Y
 return d-Q@(Q.T@d) if proj else d
if __name__=='__main__':
 print('observed rms by pulse',np.mean(Y**2,axis=0)**.5)
 start=np.tile([1,1,1,1,0,0],2)
 for a in ([1,1,1,1,0,0],[0,0,1,1,0,0],[0,0,2,2,0,0],[0,0,1,1,1,1]):
  r=res(np.tile(a,2));print('candidate',a,np.mean(r*r,axis=0)**.5)
 for proj in (False,True):
  for train in ([0,1,2],[0,1],[0,2],[1,2]):
   fit=least_squares(lambda p:res(p,proj)[:,train].ravel(),start,ftol=1e-12,gtol=1e-12,xtol=1e-12,max_nfev=150);print('fit',proj,train,np.round(fit.x,6),'rms',np.mean(res(fit.x,proj)**2,axis=0)**.5,'sv min',np.linalg.svd(fit.jac,compute_uv=False)[-1])
