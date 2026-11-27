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
one=np.ones(4); zero=np.zeros(4)
models={'common robin':(lambda a:np.r_[np.repeat(a[0],4),one],[.1]),'common edge':(lambda a:np.r_[zero,np.repeat(a[0],4)],[1.]),'axis robin':(lambda a:np.r_[np.repeat(a,2),one],[.1,.1]),'wall robin':(lambda a:np.r_[a,one],[.1]*4),'axis edge':(lambda a:np.r_[zero,np.repeat(a,2)],[1.,1.]),'wall edge':(lambda a:np.r_[zero,a],[1.]*4),'joint':(lambda a:a,[.1]*4+[1.]*4)}
for name,(pack,a0) in models.items():
 for mask in [None,0,1]:
  def f(a):
   r=predict(pack(a))-Y
   return (r if mask is None else r[:,[mask]]).ravel()
  opt=least_squares(f,a0,bounds=(0,5),gtol=1e-14,ftol=1e-12,xtol=1e-12); q=pack(opt.x); r=predict(q)-Y
  print(name,mask,np.round(q,6), 'pulse rms',np.sqrt(np.mean(r*r,axis=0)), 'worst shift',np.max(np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1))))
r=predict(np.r_[zero,one])-Y;print('reflect residual signed',r.min(axis=0),r.max(axis=0),'means by shift',r.reshape(3,32,2).mean(axis=1)); print('axes inputs',[(d,st['robin_'+d],st['kappa_'+d][[0,-1]],st['vel_'+d][[0,-1]]) for d in 'xy'])
