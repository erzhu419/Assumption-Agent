import pickle, numpy as np
from scipy.sparse import diags, eye, kron, lil_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; raw=c['raw_rosette']; B=np.linalg.lstsq(np.c_[x,np.ones(len(x))],raw,rcond=None)[0]
def cal(y): return np.linalg.solve(B[:4].T,(y.reshape(-1,4)-B[4]).T).T.ravel()
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; mus=s['mu_shifts']; pulses=item['campaign_diagnostics']; obs=np.column_stack([cal(p['y_pulse_raw']) for p in pulses]); W=np.column_stack([p['w_pulse'] for p in pulses]); rhs=np.stack([u.ravel(order='F') for u in s['source_modes']],axis=1)@W
I=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); Z=s['Z_nuisance']; P=np.linalg.solve(Z.T@Z,Z.T)
def axis(a,pars,face_edge=False):
 k,v=s['kappa_'+a],s['vel_'+a]; n=len(k); h=s['L'+a]/(n-1); f=(k[:-1]+k[1:])/(2*h*h); l=f[:-1]+np.maximum(v[1:-1],0)/h; r=f[1:]+np.maximum(-v[1:-1],0)/h; e=(f[[0,-1]] if face_edge else k[[0,-1]]/h**2)*pars[:,0]; q=s['robin_'+a]*pars[:,1]/h; p=pars[:,2]*(k[0]+k[-1])/(2*h*h)
 A=diags((-np.r_[l,e[1]],np.r_[e[0]+q[0]+p[0],l+r,e[1]+q[1]+p[1]],-np.r_[e[0],r]),(-1,0,1),format='lil'); A[0,-1]=-p[0]; A[-1,0]=-p[1]; return A.tocsc()
def forward(pars,face_edge=False):
 a=kron(eye(ny),axis('x',pars[:2],face_edge),format='csc')+kron(axis('y',pars[2:],face_edge),eye(nx),format='csc')+V
 return np.vstack([splu(a+u*I).solve(rhs)[rows] for u in mus])
def stats(e): return ' '.join(f'{v:.7g}' for v in [np.sqrt(np.mean(e**2)),np.max(np.sqrt(np.mean(e.reshape(len(mus),len(rows),-1)**2,axis=1))),np.max(abs(e))])
print('banks',s['row_bank_id'][:36]); print('pulse weights',W.T); print('mu',mus); print('boundary kappa and velocity',[(a,s['kappa_'+a][[0,1,-2,-1]],s['vel_'+a][[0,-1]]) for a in 'xy']); print('sweep residual rms',np.sqrt(np.mean((np.c_[x,np.ones(len(x))]@B-raw)**2,axis=0)))
for name,p,face in [('reflecting',[1,0,0],False),('reflecting face',[1,0,0],True),('double',[2,0,0],False),('double face',[2,0,0],True),('periodic',[1,0,1],False),('Robin',[1,1,0],False)]:
 e=forward(np.tile(p,(4,1)),face)-obs; print(name,stats(e),stats(e-Z@(P@e)))
models=[('scale',1,[0]),('Robin factor',1,[1]),('wrap',1,[2]),('scale+leak',1,[0,1]),('scale+wrap',1,[0,2]),('axis wrap',2,[2]),('axis scale+leak',2,[0,1]),('wall scale+leak',4,[0,1]),('axis full',2,[0,1,2]),('wall full',4,[0,1,2])]
for face in (False,True):
 for name,n,cols in models:
  def expand(v):
   p=np.tile([1.,0.,0.],(n,1)); p[:,cols]=v.reshape(n,len(cols)); return np.repeat(p,4//n,axis=0)
  start=np.tile([[1.,.01,.01][j] for j in cols],n)
  for train in (0,1):
   def loss(v): return (forward(expand(v),face)-obs)[:,train]
   fit=least_squares(loss,start,bounds=(0.,10.),max_nfev=150,gtol=1e-11,ftol=1e-10,xtol=1e-10); e=forward(expand(fit.x),face)-obs
   print('face',face,name,'train',train,'params',np.round(fit.x,5),'fit',stats(e[:,train]),'holdout',stats(e[:,1-train]))
