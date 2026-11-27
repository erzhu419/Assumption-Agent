import pickle, sys
import numpy as np
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m99_00_s00_solver as old
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; ds=item['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; pos=ij[:,0]+nx*ij[:,1]; shifts=s['mu_shifts']; Z=s['Z_nuisance']; W=np.array([d['w_pulse'] for d in ds]).T
x=s['sensor_calibration_sweeps']['ref_rosette']; y=s['sensor_calibration_sweeps']['raw_rosette']; a=np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]
Y=np.array([np.linalg.solve(a[1:].T,(d['y_pulse_raw'].reshape(-1,4)-a[0]).T).T.ravel() for d in ds]).T
rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W
axes=[]
for q,n in zip('xy',(nx,ny)):
 h=s['L'+q]/(n-1); axes.append((old._axis(s['kappa_'+q],s['vel_'+q],h,s['robin_'+q]).toarray(), h, s['robin_'+q]))
I=eye(nx*ny,format='csc'); reaction=diags(s['V_reaction_2d'].ravel(order='F'))
def pred(p):
 mats=[]
 for k,(A,h,r) in enumerate(axes):
  A=A.copy(); n=len(A)
  for j,(i,l,m) in enumerate([(0,1,n-1),(n-1,n-2,0)]):
   dr,dd,wrap,extra=p[4*(2*k+j):4*(2*k+j+1)]
   A[i,i]+= (dr*r+extra)/h - dd*A[i,l]+wrap/h**2
   A[i,l]*=1+dd; A[i,m]-=wrap/h**2
  mats.append(csc_matrix(A))
 L=kron(eye(ny),mats[0])+kron(mats[1],eye(nx))+reaction
 return np.vstack([splu((L+mu*I).tocsc()).solve(rhs)[pos] for mu in shifts])
def rms(x): return np.sqrt(np.mean(x*x,axis=0)).round(8)
def proj(x): return x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]
p0=np.zeros(16); err=pred(p0)-Y
print('baseline RMS',rms(err),'projected',rms(proj(err)),'Z fit',np.linalg.lstsq(Z,err,rcond=None)[0].round(5)); print('shift RMS',rms(err.reshape(3,-1,3).transpose(1,0,2)))
for nuisance in (False,True):
 for components in ([0],[0,1],[0,1,2]):
  inds=[4*j+i for j in range(4) for i in components]
  def loss(t,train):
   p=p0.copy(); p[inds]=t; E=pred(p)-Y
   return (proj(E) if nuisance else E)[:,train].ravel()
  fits=[least_squares(loss,np.zeros(len(inds)),args=(train,),bounds=([-1]*len(inds),[5]*len(inds)),max_nfev=150) for train in ([0,1,2],[1,2],[0,2],[0,1])]
  print('project',nuisance,'components',components,'parameters',fits[0].x.reshape(4,-1).round(5).tolist(),'all RMS',rms(loss(fits[0].x,[0,1,2]).reshape(-1,3)),'LOPO RMS',[float(rms(loss(f.x,[j]))) for j,f in enumerate(fits[1:])])
