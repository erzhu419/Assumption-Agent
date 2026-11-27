import sys
sys.path.insert(0,'/holdout_task')
import numpy as np, pickle
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from itertools import product
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
W=np.stack([p['w_pulse'] for p in s['reference_tracer_pulses']],axis=1)
Y=np.stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']],axis=1)
nx,ny=s['nx'],s['ny']; pos=s['sensor_ij']@np.array([1,nx]); V=diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc'); B=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W
base=[]; variants=[]
for q,n in zip('xy',(nx,ny)):
 k=s['kappa_'+q]; v=s['vel_'+q]; h=s['L'+q]/(n-1); r=s['robin_'+q]/h
 A=np.zeros((n,n)); j=np.arange(1,n-1); l=(k[j]+k[j-1])/(2*h*h)+np.maximum(v[j],0)/h; u=(k[j]+k[j+1])/(2*h*h)+np.maximum(-v[j],0)/h
 A[j,j-1]=-l; A[j,j+1]=-u; A[j,j]=l+u; base.append(A)
 b=k[[0,-1]]/h**2; seal=np.column_stack([b,[0,0],[0,0]]); robin=seal.copy(); robin[:,2]=r
 wrap=(k[0]+k[-1])/(2*h*h); per=np.array([[(k[0]+k[1])/(2*h*h)+max(-v[0],0)/h,wrap+max(v[0],0)/h,0],[(k[-1]+k[-2])/(2*h*h)+max(v[-1],0)/h,wrap+max(-v[-1],0)/h,0]])
 variants.append((seal,robin,per)); print(q,'sealed/Robin/periodic',variants[-1])
def pred(p):
 axes=[]
 for a,rates in zip(base,p.reshape(2,2,3)):
  a=a.copy(); n=len(a)
  for (j,near,far),(d,w,r) in zip([(0,1,n-1),(n-1,n-2,0)],rates): a[j,j],a[j,near],a[j,far]=d+w+r,-d,-w
  axes.append(csc_matrix(a))
 L=(kron(eye(ny),axes[0])+kron(axes[1],eye(nx))+V).tocsc()
 return np.vstack([splu(L+z*I).solve(B)[pos] for z in s['mu_shifts']])
def rms(x): return np.sqrt(np.mean(x*x,axis=0))
combos=[]
for x,y in product(range(3),repeat=2):
 p=np.stack([variants[0][x],variants[1][y]]).ravel(); e=pred(p)-Y; combos.append((np.mean(e**2),p)); print('stencil',x,y,rms(e))
p0=min(combos,key=lambda t:t[0])[1]
for i in [-1,0,1,2]:
 use=np.arange(3)!=i
 fit=least_squares(lambda p:(pred(p)-Y)[:,use].ravel(),p0,bounds=(-30,250),jac='3-point',ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=220)
 print('omit',i,'pulse RMS',rms(pred(fit.x)-Y),'rates',fit.x.reshape(2,2,3).round(6),'jac sv',np.linalg.svd(fit.jac,compute_uv=False).round(6))
