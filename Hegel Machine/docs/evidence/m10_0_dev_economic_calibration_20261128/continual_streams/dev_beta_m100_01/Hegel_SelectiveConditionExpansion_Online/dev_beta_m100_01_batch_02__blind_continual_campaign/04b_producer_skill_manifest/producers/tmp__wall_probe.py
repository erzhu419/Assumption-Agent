import sys,pickle,json
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; pul=it['campaign_diagnostics']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); y=np.array([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pul]); w=np.array([p['w_pulse'] for p in pul]); nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T; sensor=i+nx*j; nsen=len(i); z=s['Z_nuisance']; I=sparse.eye(nx*ny,format='csc'); src=(w@s['source_modes'].reshape(len(w[0]),-1,order='F')).T

def axis(k,v,h,r,b):
 f=(k[1:]+k[:-1])/(2*h*h); l=-f-np.maximum(v[1:],0)/h; u=-f-np.maximum(-v[:-1],0)/h
 l[-1]=-k[-1]/h**2*b[1,0]; u[0]=-k[0]/h**2*b[0,0]; d=-np.r_[0,l]-np.r_[u,0]; d[[0,-1]]+=r/h*b[:,1]; return sparse.diags([l,d,u],[-1,0,1],format='csc')

def predict(p):
 p=np.array(p).reshape(4,2); ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],p[:2]); ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],p[2:]); L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); return np.vstack([splu(L+mu*I).solve(src)[sensor] for mu in s['mu_shifts']]).T

def resid(p, pi=slice(None),si=slice(None),nuis=False):
 r=(predict(p)-y)[pi][:,si]; q=z[si]; return r-(q@np.linalg.lstsq(q,r.T,rcond=None)[0]).T if nuis else r

def fit(decode,p0,pi=slice(None),si=slice(None),nuis=False):
 return least_squares(lambda p:resid(decode(p),pi,si,nuis).ravel(),p0,bounds=(-.08,12),xtol=1e-13,gtol=1e-13,ftol=1e-13)

def rms(a,axis=None): return np.sqrt(np.mean(a*a,axis=axis))
if __name__=='__main__':
 for x in [0,1,2]:
  for a in [0,1,2]: print('uniform end leak',x,a,'rms',rms(resid([1,x,1,x,1,a,1,a]),1))
 configs=[('leak-all',lambda p: np.c_[np.ones(4),p].ravel(),np.ones(4)),('free-wall',lambda p:p,np.ones(8)),('axis-symmetric',lambda p:np.repeat(np.reshape(p,(2,2)),2,axis=0).ravel(),np.ones(4))]
 for name,dec,start in configs:
  for nuis in (False,True):
   f=fit(dec,start,nuis=nuis); p=dec(f.x); print(name,nuis,'p',p,'res',rms(resid(p,nuis=nuis),1),'sing',np.linalg.svd(f.jac,compute_uv=False))
   for k in range(len(y)):
    f=fit(dec,start,np.arange(len(y))!=k,nuis=nuis); print('loo',k,dec(f.x),rms(resid(dec(f.x),np.array([k]),nuis=nuis)))
   print('mean/res by shift',resid(p,nuis=nuis).reshape(3,3,-1).mean(2),rms(resid(p,nuis=nuis).reshape(3,3,-1),2))
 print('nuisance basis range',z.min(0),z.max(0))
