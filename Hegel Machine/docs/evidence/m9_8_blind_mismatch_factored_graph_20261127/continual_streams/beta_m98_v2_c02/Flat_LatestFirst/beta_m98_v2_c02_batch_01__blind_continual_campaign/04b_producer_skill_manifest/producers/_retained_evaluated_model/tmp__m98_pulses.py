import pickle
import numpy as np
from scipy import sparse
from scipy.linalg import null_space
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
import m98_base as base

item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s = item['study']
pulses = item['campaign_diagnostics']
nx,ny = s['nx'],s['ny']
Z = s['Z_nuisance']
Q = null_space(Z.T).T
W = np.stack([p['w_pulse'] for p in pulses],axis=1)
Y = np.stack([base._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses],axis=1)
S,last = base._forward(s)
R = S@W-Y
print('pulse full RMS', np.sqrt(np.mean(R**2,axis=0)), 'projected RMS',np.linalg.norm(Q@R,axis=0)/np.sqrt(len(R)))
print('baseline relative l2 by shift and pulse',np.linalg.norm(R.reshape(3,-1,2),axis=1)/np.linalg.norm(Y.reshape(3,-1,2),axis=1))
print('nuisance fits',np.linalg.lstsq(Z,-R,rcond=None)[0])
items=[]
for a,n in [('x',nx),('y',ny)]:
 h=s['L'+a]/(n-1); k=s['kappa_'+a]; v=s['vel_'+a]; r=s['robin_'+a]; A=base._axis(k,v,h,r)
 items.append((A,h,k[[0,-1]]/h**2,r/h,v[[0,-1]]/h))
print('baseline wall scales',[(k,r,v) for _,_,k,r,v in items])
I = sparse.eye(nx*ny,format='csc')
ix,iy=s['sensor_ij'].T
rows=ix+nx*iy
src=s['source_modes'].reshape(-1,nx*ny,order='F').T@W

def forward(p, kind):
 axes=[]
 for j,(A,h,k,r,v) in enumerate(items):
  B=A.tolil(); n=B.shape[0]
  for i,(b,f) in enumerate(((0,1),(n-1,n-2))):
   a=j*2+i
   if kind=='robin': B[b,b]+=r*p[a]
   if kind=='coupling':
    d=k[i]*p[a]; B[b,b]+=d; B[b,f]-=d
   if kind=='both':
    d=k[i]*p[a]; B[b,b]+=d+r*p[4+a]; B[b,f]-=d
   if kind=='wrap':
    d=k[i]*p[8+a]; B[b,b]+=k[i]*p[a]+r*p[4+a]+d; B[b,f]-=k[i]*p[a]; B[b,n-1-b]-=d
  axes.append(B.tocsc())
 L=sparse.kron(sparse.eye(ny),axes[0],format='csc')+sparse.kron(axes[1],sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
 return np.concatenate([splu(L+shift*I).solve(src)[rows] for shift in s['mu_shifts']])

for proj in (False,True):
 T=Q if proj else np.eye(len(Y))
 for kind,n in [('robin',4),('coupling',4),('both',8),('wrap',12)]:
  for train in ([0,1],[0],[1]):
   def residual(p): return (T@(forward(p,kind)-Y))[:,train].ravel()
   out=least_squares(residual,np.zeros(n),max_nfev=140,ftol=1e-11,gtol=1e-11,xtol=1e-11)
   err=T@(forward(out.x,kind)-Y)
   print(proj,kind,train,'rms',np.round(np.sqrt(np.mean(err**2,axis=0)),9),'parameters',np.round(out.x,6))
