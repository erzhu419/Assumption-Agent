import sys,json,pickle
sys.path.insert(0,'/holdout_task')
import _verified_upstream as up
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
P=item['campaign_diagnostics']; W=np.stack([p['w_pulse'] for p in P],axis=1); Y=np.stack([up.calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in P],axis=1)
Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; I=eye(nx*ny,format='csc'); modes=s['source_modes'].reshape(-1,nx*ny,order='F').T; B=modes@W

def axis(k,v,h,rob,loss,link):
 face=(k[:-1]+k[1:])/(2*h*h)
 l=np.r_[-face[:-1]-np.maximum(v[1:-1],0)/h,-link[1]*k[-1]/h**2]
 r=np.r_[-link[0]*k[0]/h**2,-face[1:]-np.maximum(-v[1:-1],0)/h]
 d=-np.r_[0,l]-np.r_[r,0]; d[[0,-1]]+=np.array(loss)*rob/h
 return diags([l,d,r],[-1,0,1],format='csc')

def pred(p,rhs=B):
 A=[axis(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1),s['robin_'+a],p[2*j:2*j+2],p[4+2*j:6+2*j]) for j,a in enumerate('xy')]
 L=(kron(eye(ny),A[0])+kron(A[1],eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.concatenate([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])

def report(label,p):
 r=pred(p)-Y; rr=r-Q@(Q.T@r); print(label,'parameters',np.round(p,7),'raw pulse RMS',np.sqrt(np.mean(r*r,axis=0)),'projected pulse RMS',np.sqrt(np.mean(rr*rr,axis=0)))

if __name__=='__main__':
 for name,p in [('Robin',np.ones(8)),('reflecting',np.r_[np.zeros(4),np.ones(4)])]:report(name,p)
 for nuisance in (False,True):
  def resid(p):
   r=pred(p)-Y
   return r-Q@(Q.T@r) if nuisance else r
  for n in (4,8):
   fit=least_squares(lambda x:resid(np.r_[x,np.ones(8-n)]).ravel(),np.ones(n),bounds=(0,10),gtol=1e-12)
   report(f'fit {n}, nuisance={nuisance}',np.r_[fit.x,np.ones(8-n)])
   print('Jacobian singular values',np.round(np.linalg.svd(fit.jac,compute_uv=False),6))
 print('pulses',W.T)
