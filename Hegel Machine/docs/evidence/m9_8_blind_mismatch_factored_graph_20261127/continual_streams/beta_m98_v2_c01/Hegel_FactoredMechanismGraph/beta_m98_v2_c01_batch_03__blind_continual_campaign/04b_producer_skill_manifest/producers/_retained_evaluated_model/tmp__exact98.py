import pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=item['study']; sw=st['sensor_calibration_sweeps']; ref=sw['ref_rosette']; raw=sw['raw_rosette']; B=np.linalg.lstsq(np.c_[np.ones(len(ref)),ref],raw,rcond=None)[0]
Y=np.stack([np.linalg.solve(B[1:].T,(p['y_pulse_raw'].reshape(-1,4)-B[0]).T).T.ravel() for p in item['campaign_diagnostics']],axis=-1)
W=np.stack([p['w_pulse'] for p in item['campaign_diagnostics']],axis=1); Z=st['Z_nuisance']; nx,ny=st['nx'],st['ny']; ij=st['sensor_ij']; T=ij[:,0]+nx*ij[:,1]; V=diags(st['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); rhs=np.column_stack([u.ravel(order='F') for u in st['source_modes']])@W
def axis(d,loss,mult):
 k=st['kappa_'+d]; u=st['vel_'+d]; h=st['L'+d]/(len(k)-1); f=(k[:-1]+k[1:])*.5/h**2; l=f[:-1]+np.maximum(u[1:-1],0)/h; r=f[1:]+np.maximum(-u[1:-1],0)/h; e=k[[0,-1]]/h**2*mult; b=e+loss*st['robin_'+d]/h
 return diags((-np.r_[l,e[1]],np.r_[b[0],l+r,b[1]],-np.r_[e[0],r]),(-1,0,1),format='csc')
def predict(q):
 ax=axis('x',q[:2],q[4:6]); ay=axis('y',q[2:4],q[6:]); L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[T] for mu in st['mu_shifts']])
from itertools import product
from scipy.sparse import csc_matrix
def wrap_axis(d,loss,inward,wrap,averaged=False,adv=False):
 k=st['kappa_'+d]; v=st['vel_'+d]; h=st['L'+d]/(len(k)-1); e=k[[0,-1]]/h**2
 A=axis(d,loss,inward).toarray(); ew=np.repeat(e.mean(),2) if averaged else e
 for side,i,j,near in [(0,0,len(k)-1,1),(1,len(k)-1,0,len(k)-2)]:
  z=wrap[side]*ew[side]; A[i,j]-=z; A[i,i]+=z
  if adv:
   z=abs(v[i])/h; col=j if ((side==0 and v[i]>0) or (side==1 and v[i]<0)) else near
   A[i,col]-=z; A[i,i]+=z
 return csc_matrix(A)
def pred(q,avg=False,adv=False):
 ax=wrap_axis('x',*q[:,:2],avg,adv); ay=wrap_axis('y',*q[:,2:],avg,adv)
 L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[T] for mu in st['mu_shifts']])
def metrics(r):
 return np.sqrt(np.mean(r*r)),np.max(np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1)))
q=np.zeros((3,4))
for a,d in enumerate('xy'):
 k=st['kappa_'+d]; v=st['vel_'+d]; h=st['L'+d]/(len(k)-1); e=k[[0,-1]]
 q[1,2*a:2*a+2]=0.5*(e+k[[1,-2]])/e
 q[2,2*a:2*a+2]=e.mean()/e
print('face-average ratios',q)
r=pred(q,False,True)-Y
print('exact stencil per shift and pulse',np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1)), 'max abs',np.max(np.abs(r)))
# Compare the best candidate selected independently on each pulse.
from itertools import product
candidates=[]
for leftavg,seamavg,adv in product([False,True],repeat=3):
 t=q.copy()
 if not leftavg:t[1]=1
 if not seamavg:t[2]=1
 r=pred(t,False,adv)-Y
 candidates.append((np.mean(r*r,axis=0),np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1)),(leftavg,seamavg,adv)))
for j in (0,1):
 c=min(candidates,key=lambda a:a[0][j]); print('trained on',j,'stencil',c[2],'heldout worst shift',np.max(c[1][:,1-j]))
