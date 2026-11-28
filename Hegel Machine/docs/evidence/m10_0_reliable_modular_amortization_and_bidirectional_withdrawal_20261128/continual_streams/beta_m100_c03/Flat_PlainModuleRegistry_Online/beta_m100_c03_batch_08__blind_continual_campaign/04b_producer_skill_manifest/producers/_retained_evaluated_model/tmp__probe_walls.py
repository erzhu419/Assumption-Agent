import sys,pickle,itertools
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
nx,ny=s['nx'],s['ny']; n=nx*ny
ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]
z=s['Z_nuisance']; Q=np.linalg.qr(z)[0]; weights=np.array([p['w_pulse'] for p in s['reference_tracer_pulses']]).T
y=np.array([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]).T
rhs=s['source_modes'].transpose(0,2,1).reshape(-1,n).T @ weights
I=eye(n,format='csc'); reaction=diags(s['V_reaction_2d'].ravel(order='F'))
def axis(d,p):
 k=s['kappa_'+d];v=s['vel_'+d];h=s['L'+d]/(len(k)-1)
 f=(k[1:]+k[:-1])/(2*h*h); left=f[:-1]+np.maximum(v[1:-1],0)/h; right=f[1:]+np.maximum(-v[1:-1],0)/h
 a=diags((-np.r_[left,p[3]],np.r_[sum(p[:3]),left+right,sum(p[3:])],-np.r_[p[0],right]),[-1,0,1],format='lil')
 a[0,-1]=-p[1];a[-1,0]=-p[4]
 return a.tocsc()
def states(d):
 k=s['kappa_'+d];v=s['vel_'+d];h=s['L'+d]/(len(k)-1);rob=s['robin_'+d]/h
 f=(k[1:]+k[:-1])/(2*h*h); seam=(k[0]+k[-1])/(2*h*h)
 return [(k[0]/h**2,0,rob),(k[-1]/h**2,0,rob)],[(k[0]/h**2,0,0),(k[-1]/h**2,0,0)],[(f[0]+max(-v[0],0)/h,seam+max(v[0],0)/h,0),(f[-1]+max(v[-1],0)/h,seam+max(-v[-1],0)/h,0)]
def pred(p):
 L=(kron(eye(ny),axis('x',p[:6]))+kron(axis('y',p[6:]),eye(nx))+reaction).tocsc()
 return np.concatenate([splu(L+mu*I).solve(rhs)[idx,:] for mu in s['mu_shifts']])
def residual(p,profile=False):
 r=pred(p)-y
 return r-Q@(Q.T@r) if profile else r
sx,sy=[states(d) for d in 'xy']; lookup=[[a[j] for a in t] for t in (sx,sy) for j in range(2)]
print('pulse rms',np.mean(y*y,axis=0)**.5,'nuisance correlations',np.linalg.norm(Q.T@y,axis=0)); print('states',lookup)
sc=[]
for keys in itertools.product(range(3),repeat=4):
 p=np.array([lookup[i][k] for i,k in enumerate(keys)]).ravel();r=residual(p)
 sc.append((np.linalg.norm(r),keys,p,np.sqrt(np.mean(r*r,axis=0))))
for norm,keys,p,r in sorted(sc,key=lambda x:x[0])[:12]: print('discrete',keys,norm,r)
for prof in (False,True):
 for hold in (None,0,1,2):
  cols=[j for j in range(3) if j!=hold];p0=sorted(sc,key=lambda x:x[0])[0][2]
  opt=least_squares(lambda p:residual(p,prof)[:,cols].ravel(),p0,max_nfev=200,gtol=1e-12,ftol=1e-12,xtol=1e-12)
  print('free',prof,hold,np.round(opt.x.reshape(-1,3),6).tolist(),'pulse rms',np.sqrt(np.mean(residual(opt.x,prof)**2,axis=0)))
