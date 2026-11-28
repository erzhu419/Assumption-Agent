import itertools
import numpy as np
from scipy.optimize import least_squares
import bound08 as b
s,m=b.s,b.m
f=b.fit(b.X,b.Y,'sep'); y=b.calibrate(f)
Z=s['Z_nuisance']; n=s['source_modes'].shape[0]; cap=s['u_box']; budget=s['total_budget']
Q=np.linalg.qr(Z,mode='reduced')[0]; P=np.eye(len(y))-Q@Q.T
# Endpoints: normal diffusion scale, wrapped diffusion scale, wall loss.
def axis(k,v,h,p):
 A=b.orig(k,v,h,[0,0]).tolil(); N=len(k)
 for r,i,j,t in [(0,0,1,N-1),(1,N-1,N-2,0)]:
  normal,wrap,loss=p[3*r:3*r+3]; A[i,j]=-normal*k[i]/h**2; A[i,t]=-wrap*(k[i]+k[t])/(2*h*h); A[i,i]=-A[i,j]-A[i,t]+loss/h
 return A.tocsc()
def response(p):
 def op(k,v,h,loss):return axis(k,v,h,p[:6] if len(k)==s['nx'] else p[6:])
 m.axis_operator=op
 return m.chamber_response(s,[0]*4)
def prof(S,sel=None):
 if sel is None: sel=np.ones(len(y),bool)
 z=Z[sel]; q=np.linalg.qr(z,mode='reduced')[0]; a=S[sel]-q@(q.T@S[sel]); t=y[sel]-q@(q.T@y[sel]); w,_=m.box_simplex(a.T@a+1e-12*np.eye(n),a.T@t,cap,budget); g=np.linalg.lstsq(z,y[sel]-S[sel]@w,rcond=None)[0]; return S@w+Z@g-y,w
for i,j in itertools.product(range(3),repeat=2):
 m.axis_operator=b.per
 S,U=m.chamber_response(s,[[s['robin_x'],0,np.nan][i]]*2+[[s['robin_y'],0,np.nan][j]]*2); e,w=prof(S); print('discrete',i,j,np.linalg.norm(e),np.linalg.norm(e.reshape(3,-1),axis=1).round(5))
for name,base,B in [('robin',np.tile([1.,0,0],4),np.eye(12)[:,[2,5,8,11]]),('sealed seam',np.tile([1.,0,0],4),np.eye(12)[:,[1,4,7,10]]),('generic',np.tile([1.,0,0],4),np.eye(12))]:
 def fit(sel):
  return least_squares(lambda q:prof(response(base+B@q)[0],sel)[0][sel],np.zeros(B.shape[1]),max_nfev=90,ftol=1e-10)
 allr=fit(np.ones(len(y),bool)); print('parametric',name,'norm',np.linalg.norm(allr.fun),'params',(base+B@allr.x).round(5),flush=True)
 if name!='generic':
  for k in range(len(s['mu_shifts'])):
   sel=np.arange(len(y))//len(s['sensor_ij'])!=k; out=fit(sel); e,_=prof(response(base+B@out.x)[0],sel); print('shift holdout',name,k,np.linalg.norm(e[~sel]),(base+B@out.x).round(4),flush=True)
print('aux first rows',s['aux_manifold_telemetry'][:8].round(3)); print('prior stencil summaries')
