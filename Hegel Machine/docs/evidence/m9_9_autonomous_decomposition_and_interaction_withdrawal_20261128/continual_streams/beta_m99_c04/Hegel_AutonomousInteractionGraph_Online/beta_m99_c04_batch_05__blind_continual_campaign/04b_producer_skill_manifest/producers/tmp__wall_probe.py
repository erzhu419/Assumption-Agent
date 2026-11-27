import sys, itertools, pickle
import numpy as np
from scipy.sparse import csc_matrix, diags, kron, eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(p['study'],reference_tracer_pulses=p['campaign_diagnostics']))
W=np.array([d['w_pulse'] for d in s['reference_tracer_pulses']]).T
Y=np.array([d['y_pulse_raw'] for d in s['reference_tracer_pulses']]).T
nx,ny=s['nx'],s['ny']; ix,iy=s['sensor_ij'].T; inds=ix+nx*iy
rhs=np.stack([m.ravel(order='F') for m in s['source_modes']],axis=1)@W
axes=[]
for q in 'xy':
 k,v=s['kappa_'+q],s['vel_'+q]; n=len(k); h=s['L'+q]/(n-1); i=np.arange(n); left=(i-1)%n; right=(i+1)%n
 dn=(k+k[left])/(2*h*h)+np.maximum(v,0)/h; up=(k+k[right])/(2*h*h)+np.maximum(-v,0)/h
 a=np.zeros((n,n)); a[i,i]=dn+up; a[i,left]=-dn; a[i,right]=-up
 inward=np.array([k[0],k[-1]])/h**2; rob=s['robin_'+q]/h
 cases={'periodic':np.array([[up[0],dn[0],0],[dn[-1],up[-1],0]]),'sealed':np.column_stack([inward,[0,0],[0,0]]),'robin':np.column_stack([inward,[0,0],[rob,rob]])}
 axes.append((a,cases))
ident=eye(nx*ny,format='csc'); reaction=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); shifts=s['mu_shifts']
def residual(param):
 mats=[]
 for (a,_),b in zip(axes,np.asarray(param).reshape(2,2,3)):
  a=a.copy(); a[[0,-1],:]=0; a[0,[1,-1,0]]=[-b[0,0],-b[0,1],sum(b[0])]; a[-1,[-2,0,-1]]=[-b[1,0],-b[1,1],sum(b[1])]; mats.append(csc_matrix(a))
 L=kron(eye(ny),mats[0],format='csc')+kron(mats[1],eye(nx),format='csc')+reaction
 return np.stack([splu(L+m*ident).solve(rhs)[inds] for m in shifts])-Y.reshape(len(shifts),len(inds),-1)
rank=[]
for a,b in itertools.product(*[cases for _,cases in axes]):
 par=np.stack([axes[0][1][a],axes[1][1][b]]); r=residual(par); rank.append((np.linalg.norm(r),a,b,par)); print(a,b,'rmse',np.sqrt(np.mean(r*r)), 'max',np.max(np.abs(r)))
best=min(rank,key=lambda v:v[0]); print('best discrete',best[1:]);
for _,cases in axes: print('axis cases',cases)
if best[0]>0.004:
 for hold in [-1,0,1,2]:
  train=np.arange(W.shape[1])!=hold
  fit=least_squares(lambda par:residual(par)[:,:,train].ravel(), best[3].ravel(),diff_step=1e-4,max_nfev=250,ftol=1e-12,gtol=1e-12)
  res=residual(fit.x); print('held pulse',hold,'fit',fit.x.reshape(2,2,3).round(6),'RMSE per shift and pulse',np.sqrt(np.mean(res**2,axis=1)).round(8),'max',np.max(np.abs(res)))
