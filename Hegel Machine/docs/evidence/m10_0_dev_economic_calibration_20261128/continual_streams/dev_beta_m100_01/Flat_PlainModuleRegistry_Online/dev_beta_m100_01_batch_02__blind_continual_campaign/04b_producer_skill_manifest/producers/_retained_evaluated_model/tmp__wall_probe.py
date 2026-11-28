import sys,pickle,json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=b['study'];pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
pulses=b['campaign_diagnostics']; W=np.array([d['w_pulse'] for d in pulses]); Y=np.array([calibrate_upstream(dict(s,y_obs_raw=d['y_pulse_raw']),pay)['y_obs_raw'] for d in pulses]).T
nx,ny=s['nx'],s['ny']; N=nx*ny
I=sparse.eye(N,format='csc'); ix,iy=s['sensor_ij'].T; at=ix+nx*iy
rhs=s['source_modes'].transpose(0,2,1).reshape(-1,N).T @ W.T

def axis(k,v,h,r,p):
 f=(k[:-1]+k[1:])/(2*h*h); lo=-f-np.maximum(v[1:],0)/h; hi=-f-np.maximum(-v[:-1],0)/h
 hi[0]=-p[2]*k[0]/h**2;lo[-1]=-p[3]*k[-1]/h**2
 d=-np.r_[0.,lo]-np.r_[hi,0.];d[[0,-1]]+=r/h*np.array(p[:2]);return sparse.diags([lo,d,hi],[-1,0,1],format='csc')

def pred(p):
 ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],p[:4]);ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],p[4:]);L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc');return np.vstack([splu(L+mu*I).solve(rhs)[at] for mu in s['mu_shifts']])

if __name__=='__main__':
 for kind,p in [('open',np.ones(8)),('sealed',np.tile([0,0,1,1],2)),('x open',np.array([1,1,1,1,0,0,1,1])),('y open',np.array([0,0,1,1,1,1,1,1]))]:
  e=pred(p)-Y; print(kind,'pulse RMSE',np.sqrt(np.mean(e**2,axis=0)),'max',abs(e).max())
 for hold in [-1,0,1,2]:
  train=np.arange(3)!=hold
  fit=least_squares(lambda p:(pred(p)-Y)[:,train].ravel(),np.ones(8),bounds=([-0.1,-0.1,0.05,0.05]*2,[6,6,8,8]*2),gtol=1e-13,xtol=1e-13,ftol=1e-13)
  e=pred(fit.x)-Y;print('hold',hold,'wall',np.round(fit.x,7),'pulse RMSE',np.sqrt(np.mean(e**2,axis=0)), 'Jacobian sigma min',np.linalg.svd(fit.jac,compute_uv=False)[-1]);
  if hold==-1:
   for j,idx in enumerate([ix==0,iy==0,(ix>0)&(iy>0)]): print('slice',j,np.sqrt(np.mean(e[np.tile(idx,len(s['mu_shifts']))]**2)))
