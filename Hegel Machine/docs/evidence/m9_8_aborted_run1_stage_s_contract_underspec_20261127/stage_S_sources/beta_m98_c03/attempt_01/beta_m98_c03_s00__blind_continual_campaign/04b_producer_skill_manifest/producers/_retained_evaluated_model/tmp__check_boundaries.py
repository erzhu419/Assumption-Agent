import pickle,itertools
import numpy as np
from scipy.sparse import diags,eye,kron,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
d=s['sensor_calibration_sweeps']; x=d['ref_rosette']; y=d['raw_rosette']; c=np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]
p=s['reference_tracer_pulses']; Y=np.stack([np.linalg.solve(c[1:].T,(q['y_pulse_raw'].reshape(-1,4)-c[0]).T).T.ravel() for q in p],axis=1)
W=np.stack([q['w_pulse'] for q in p],axis=1); nx,ny=s['nx'],s['ny']; modes=np.array([m.ravel(order='F') for m in s['source_modes']]).T@W
ix,iy=s['sensor_ij'].T; ids=ix+nx*iy; N=nx*ny; I=eye(N,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
def axis(k,v,h,avg,adv,wall,sign):
 n=len(k); kf=(k[:-1]+k[1:])/2 if avg=='arith' else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]); kf/=h*h
 m=np.zeros((n,n)); j=np.arange(n-1); m[j,j]+=kf;m[j+1,j+1]+=kf;m[j,j+1]-=kf;m[j+1,j]-=kf
 v=sign*v/h; vf=(v[:-1]+v[1:])/2
 if adv in ('flux','flux_edge','flux_left','flux_right'):
  if adv=='flux_left':vf=v[:-1]
  if adv=='flux_right':vf=v[1:]
  vp=np.maximum(vf,0);vn=np.minimum(vf,0);m[j,j]+=vp;m[j+1,j]-=vp;m[j,j+1]+=vn;m[j+1,j+1]-=vn
  if adv=='flux_edge':m[0,0]+=max(-v[0],0);m[-1,-1]+=max(v[-1],0)
 elif adv in ('row','row_edge'):
  m[j+1,j]-=np.maximum(v[1:],0);m[j+1,j+1]+=np.maximum(v[1:],0);m[j,j+1]+=np.minimum(v[:-1],0);m[j,j]-=np.minimum(v[:-1],0)
  if adv=='row_edge':m[0,0]+=max(v[0],0);m[-1,-1]+=max(-v[-1],0)
 elif adv=='row_face':
  m[j+1,j]-=np.maximum(vf,0);m[j+1,j+1]+=np.maximum(vf,0);m[j,j+1]+=np.minimum(vf,0);m[j,j]-=np.minimum(vf,0)
 return csc_matrix(m)
def setup(params):
 grid,avg,adv,sign=params
 Ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx+grid),avg,adv,0,sign); Ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny+grid),avg,adv,0,sign)
 return (kron(eye(ny),Ax)+kron(Ay,eye(nx))+V).tocsc()
X=np.tile(np.arange(nx),ny); J=np.repeat(np.arange(ny),nx); mask=np.array([X==0,X==nx-1,J==0,J==ny-1],float).T

def predict(A,r):
 if len(r)==2:r=np.repeat(r,2)
 A=A+diags(mask@r,format='csc'); return np.concatenate([splu(A+mu*I).solve(modes)[ids] for mu in s['mu_shifts']])
def fit(A,num,sel=None):
 if sel is None:sel=np.ones_like(Y,dtype=bool)
 def fun(r):return (predict(A,r)-Y)[sel]
 return least_squares(fun,[1.]*num,bounds=(0,200),gtol=1e-11)
from scipy.linalg import null_space
A=setup((-1,'arith','row',1)); nominal=np.array([s['robin_x']*(nx-1)/s['Lx'],s['robin_y']*(ny-1)/s['Ly']]); print('nominal wall',nominal,'pulse RMS', np.sqrt(np.mean((predict(A,nominal)-Y)**2)))
for num in (2,4):
 e=[]
 for j in range(2):
  train=np.ones_like(Y,dtype=bool);train[:,j]=False
  f=fit(A,num,train); e.append(np.sqrt(np.mean((predict(A,f.x)[:,j]-Y[:,j])**2))); print('held pulse',j,'walls',f.x,'RMSE',e[-1])
 print('worst',num,max(e))
f=fit(A,2); r=predict(A,f.x)-Y; print('sensor/slice residuals shift, pulse',np.sqrt(np.mean(r.reshape(3,-1,2)**2,axis=1))); print('residual means',r.mean(axis=0)); print('rms after removing nuisance',np.mean((r-Q@(Q.T@r))**2)**.5)
B=np.array([m.ravel(order='F') for m in s['source_modes']]).T
obs=np.linalg.solve(c[1:].T,(s['y_obs_raw'].reshape(-1,4)-c[0]).T).T.ravel()
budget=s['total_budget']; U=s['u_box']; E=null_space(np.ones((1,len(B.T)))); center=np.full(len(B.T),budget/len(B.T)); P=np.eye(len(Z))-Q@Q.T
def design(wall):
 if len(wall)==2: wall=np.repeat(wall,2)
 matrix=A+diags(mask@wall,format='csc'); return np.concatenate([splu(matrix+mu*I).solve(B)[ids] for mu in s['mu_shifts']])
for num in (2,4):
 def loss(wall,ret=False):
  S=design(wall); C=P@S; a=C@E; b=P@obs-C@center; z=np.linalg.lstsq(a,b,rcond=None)[0]
  return (center+E@z, np.linalg.svd(a,compute_uv=False)) if ret else a@z-b
 x0=nominal if num==2 else np.repeat(nominal,2); z=least_squares(loss,x0,bounds=(0,np.inf),gtol=1e-13); w, sv=loss(z.x,True)
 print('blind obs',num,'prior rms',np.mean(loss(x0)**2)**.5,'fitted rms',np.mean(z.fun**2)**.5,'wall',z.x,'weights',w,'sv',sv,'wall profiled SV',np.linalg.svd(z.jac,compute_uv=False))
