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
res=[]
for par in itertools.product((0,-1,1),('arith','harm'),('flux','flux_edge','flux_left','flux_right','row','row_edge','row_face','none'),(1,-1)):
 A=setup(par)
 for num in (2,4):
  f=fit(A,num); res.append((np.sqrt(np.mean(f.fun**2)),par,num,f.x))
for e,par,num,r in sorted(res,key=lambda row:row[0])[:22]:print('RMSE',round(e,9),par,num,np.round(r,6))
print('raw signal RMS',np.mean(Y**2)**.5,'Y mean/min/max',Y.mean(),Y.min(),Y.max()); print('source mode ranges',[(m.min(),m.max()) for m in s['source_modes']]);print('nominal',s['robin_x'],s['robin_y'])
