import pickle, numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from itertools import product
import sys
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m98_00_batch_01_solver as sealed, dev_beta_m98_00_batch_02_solver as robin, dev_beta_m98_00_batch_03_solver as periodic
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; nxt=(np.arange(4)+1)%4; prev=(np.arange(4)-1)%4
A=np.c_[np.ones(len(x)),x]; err=np.linalg.norm(y-A@np.linalg.lstsq(A,y,rcond=None)[0],axis=1); good=err<6*np.median(err)
fit=np.array([np.linalg.lstsq(np.c_[np.ones(len(x)),x[:,i],x[:,nxt[i]],x[:,prev[i]],x[:,i]*x[:,nxt[i]],x[:,i]**3][good],y[good,i],rcond=None)[0] for i in range(4)]).T
b,d,p,m,q,t=fit; raw=s['y_obs_raw'].reshape(-1,4); x=np.zeros_like(raw); J=np.zeros((len(x),4,4)); i=np.arange(4)
for _ in range(10):
 r=b+d*x+p*x[:,nxt]+m*x[:,prev]+q*x*x[:,nxt]+t*x**3-raw
 J[:,i,i]=d+q*x[:,nxt]+3*t*x*x; J[:,i,nxt]=p+q*x; J[:,i,prev]=m; x-=np.linalg.solve(J,r[...,None])[...,0]
print('inverse residual',abs(r).max(),'calibrated range',x.min(),x.max()); y=x.ravel()
print('scalar fields',[(j,s[j]) for j in s if np.isscalar(s[j])]); print('telemetry means',np.mean(s['aux_manifold_telemetry'],axis=0))
nx,ny=s['nx'],s['ny']; dx=s['Lx']/(nx-1); dy=s['Ly']/(ny-1)
def ax(kind, c, h):
 k,v=s['kappa_'+c],s['vel_'+c]; rob=s['robin_'+c] if 'robin_'+c in s else next(s[j] for j in s if 'rob' in j and j.endswith(c));
 if kind=='S': return sealed._axis(k,v,h)
 if kind=='P': return periodic._axis(k,v,h)
 return robin._axis(k,v,h,rob)
n=nx*ny; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; rhs=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); I=sparse.eye(n,format='csc'); v=sparse.diags(s['V_reaction_2d'].ravel(order='F')); z=s['Z_nuisance']; zr=s['nuisance_ridge']; sr=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']; groups=np.arange(len(y)).reshape(len(s['mu_shifts']),-1,4).transpose(1,0,2).reshape(-1,4*len(s['mu_shifts']));
def solve(D,y,z):
 p=np.linalg.solve(z.T@z+zr*np.eye(z.shape[1]),z.T); a=D-z@(p@D); b=y-z@(p@y); w,_=sealed._box_qp(a.T@a+sr*np.eye(a.shape[1]),-a.T@b,cap,budget); return w,p,0.5*(np.linalg.norm(a@w-b)**2+sr*(w@w))
for a,b in product('RSP',repeat=2):
 L=sparse.kron(sparse.eye(ny),ax(a,'x',dx),format='csc')+sparse.kron(ax(b,'y',dy),sparse.eye(nx),format='csc')+v
 D=np.vstack([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']]); w,p,j=solve(D,y,z); rms=[]
 for g in groups:
  mask=np.ones(len(y),bool); mask[g]=False; u,p,_=solve(D[mask],y[mask],z[mask]); rms.append(np.sqrt(np.mean((D[g]@u+z[g]@(p@(y[mask]-D[mask]@u))-y[g])**2)))
 print(a+b,'J',round(j,9),'CV pooled/max',np.sqrt(np.mean(np.square(rms))),max(rms),'w',w.round(5))
