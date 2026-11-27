import sys,pickle,numpy as np
from scipy.sparse import eye,kron,diags
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m98_v2_c03_batch_02_solver as m
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; nx,ny=s['nx'],s['ny']; ns=len(s['sensor_ij']); ix,iy=s['sensor_ij'].T; sensor=ix+nx*iy
W=np.column_stack([p['w_pulse'] for p in item['campaign_diagnostics']]); Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in item['campaign_diagnostics']]); F=np.column_stack([a.ravel(order='F') for a in np.tensordot(W.T,s['source_modes'],axes=1)]); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc')
axes=[]; base=[]
for k,n in [('x',nx),('y',ny)]:
 h=s['L'+k]/(n-1); kap=s['kappa_'+k]; axes.append(m._axis(kap,s['vel_'+k],h,0)); base.append((kap[[0,-1]]/h**2,s['robin_'+k]/h))
def predict(p):
 A=[]
 for ax,(q,r),v in zip(axes,base,p.reshape(2,2,2)):
  a=ax.tolil(); a[0,0]=q[0]*v[0,0]+r*v[0,1]; a[0,1]=-q[0]*v[0,0]; a[-1,-1]=q[1]*v[1,0]+r*v[1,1]; a[-1,-2]=-q[1]*v[1,0]; A.append(a.tocsc())
 L=kron(eye(ny),A[0],format='csc')+kron(A[1],eye(nx),format='csc')+V
 return np.vstack([splu(L+t*I).solve(F)[sensor] for t in s['mu_shifts']])
if __name__=='__main__':
 print('scales',base,'unique sensor locations',len(set(sensor)))
 rows=np.arange(len(Y)); groups=np.tile(np.arange(ns)//4,len(s['mu_shifts']))
 masks=[('all',np.ones_like(Y,dtype=bool))]+[(f'pulse{j}',np.broadcast_to(np.arange(Y.shape[1])==j,Y.shape)) for j in range(Y.shape[1])]+[(f'rosette{g}',np.broadcast_to((groups!=g)[:,None],Y.shape)) for g in np.unique(groups)]
 rmse=lambda x: np.sqrt(np.mean(x*x))
 for name,mask in masks:
  out=least_squares(lambda p:(predict(p)-Y)[mask],np.tile([1.,0.],4),bounds=(np.tile([0.,-.1],4),np.tile([8.,3.],4)),gtol=1e-13,ftol=1e-12,xtol=1e-12); e=predict(out.x)-Y
  print(name,out.x.round(7),'train',rmse(e[mask]),'held',rmse(e[~mask]) if not mask.all() else None,'sv',np.linalg.svd(out.jac,compute_uv=False)[[0,-1]],flush=True)
