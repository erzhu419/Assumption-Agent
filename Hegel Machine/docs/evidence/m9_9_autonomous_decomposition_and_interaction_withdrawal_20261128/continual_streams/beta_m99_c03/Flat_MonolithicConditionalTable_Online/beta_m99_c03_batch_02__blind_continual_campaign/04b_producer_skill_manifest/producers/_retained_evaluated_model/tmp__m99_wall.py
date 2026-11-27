import pickle, importlib.util, numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; sw=s['sensor_calibration_sweeps']; x,y=sw['ref_rosette'],sw['raw_rosette']; w=np.array([p['w_pulse'] for p in item['campaign_diagnostics']]).T
raw=np.array([p['y_pulse_raw'] for p in item['campaign_diagnostics']]).T
from itertools import combinations_with_replacement
for d in (1,2,3):
 terms=[np.ones(len(x))]+[np.prod(x[:,ix],axis=1) for k in range(1,d+1) for ix in combinations_with_replacement(range(4),k)]; X=np.array(terms).T
 pred=X@np.linalg.lstsq(X,y,rcond=None)[0]; errs=[]
 for i in range(len(x)):
  t=np.arange(len(x))!=i; errs.append(X[i]@np.linalg.lstsq(X[t],y[t],rcond=None)[0]-y[i])
 print('cal degree',d,'LOO RMSE',np.mean(np.square(errs),axis=0)**.5,'max',np.max(np.abs(errs)))
 if d==1: B=np.linalg.lstsq(X,y,rcond=None)[0]; print('affine',B)
cal=np.linalg.solve(B[1:].T,(raw.T.reshape(-1,4)-B[0]).T).T.reshape(raw.shape[1],-1).T
nx,ny=s['nx'],s['ny']; ix,iy=s['sensor_ij'].T; inds=ix+nx*iy
rhs=np.column_stack([a.ravel(order='F') for a in s['source_modes']])@w
I=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'))
def axis(a,p):
 k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1); rob=s['robin_'+a]; f=(k[1:]+k[:-1])/(2*h*h)
 lo=np.r_[-f[:-1]-np.maximum(v[1:-1],0)/h,-p[1]*k[-1]/h**2]; hi=np.r_[-p[0]*k[0]/h**2,-f[1:]-np.maximum(-v[1:-1],0)/h]; diag=np.r_[-hi[0]+p[2]*rob/h,-lo[:-1]-hi[1:],-lo[-1]+p[3]*rob/h]
 return sparse.diags([lo,diag,hi],[-1,0,1],format='csc')
def forward(p):
 A=sparse.kron(sparse.eye(ny),axis('x',p[:4]),format='csc')+sparse.kron(axis('y',p[4:]),sparse.eye(nx),format='csc')+V
 return np.concatenate([splu(A+I*t).solve(rhs)[inds] for t in s['mu_shifts']])
from itertools import product
Z=s['Z_nuisance']
Q=np.linalg.qr(Z)[0]
print('pulse keys',list(item['campaign_diagnostics'][0]), 'Z norm',np.linalg.norm(Z,axis=0))
def axop(a,mode):
 k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1); n=len(k)
 A=axis(a,[1,1,0,0]).tolil(); rob=s['robin_'+a]
 for i,j,wrap in [(0,1,n-1),(n-1,n-2,0)]:
  if mode=='open': A[i,i]+=rob/h
  if mode in ('wrap','periodic','periodic_diff'):
   if mode=='wrap': A[i,wrap]=A[i,j]; A[i,j]=0; continue
   km=(k[i]+k[(i-1)%n])/2/h**2; kp=(k[i]+k[(i+1)%n])/2/h**2
   vl=v[i] if mode=='periodic' else 0.
   A[i,(i-1)%n]=-km-max(vl,0)/h; A[i,(i+1)%n]=-kp-max(-vl,0)/h; A[i,i]=-A[i,(i-1)%n]-A[i,(i+1)%n]
  if mode=='dirichlet': A[i,i]+=10000/h
 return A.tocsc()
def pred(ax,ay):
 A=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+V
 return np.concatenate([splu(A+I*t).solve(rhs)[inds] for t in s['mu_shifts']])
rows=[]
for a,b in product(('reflect','open','wrap','periodic','periodic_diff','dirichlet'),repeat=2):
 E=pred(axop('x',a),axop('y',b))-cal; E0=E-Q@(Q.T@E)
 rows.append((np.max(np.mean(E**2,axis=0)**.5),np.max(np.mean(E0**2,axis=0)**.5),a,b))
print('raw best',sorted(rows)[:8]); print('nuisance-orthogonal best',sorted(rows,key=lambda r:r[1])[:8])
E=pred(axop('x','reflect'),axop('y','reflect'))-cal
for k,e in enumerate(np.split(E,len(s['mu_shifts']))):
 print('reflect residual shift',k,'mean',e.mean(0).round(5),'rmse',np.mean(e**2,axis=0)**.5,'first pulse',e[:,0].round(4))
