import pickle,sys,itertools
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c04_batch_03_solver as arc
from scipy import sparse
from scipy.sparse.linalg import splu
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; pulses=a['campaign_diagnostics']; cal=s['sensor_calibration_sweeps']; ref=cal['ref_rosette']; raw=cal['raw_rosette']; good=np.max(np.abs(raw),axis=1)<8
bank=cal['single_bank_sweeps']; ids=bank['bank_id']; bx=np.eye(4)[ids]*bank['ref_signal'][:,None]; by=bank['raw_readout']
def feat(x,k,kind):
 v=x[...,k]; p=x[...,(k-1)%4]; n=x[...,(k+1)%4]; o=x[...,(k+2)%4]; f=[np.ones_like(v),v,p,n,v*n,v**3]
 if kind>=1:f += [o,v*p,v*o,v*v,p*n,p*o,n*o,p*p,n*n,o*o]
 if kind>=2:f += [x[...,i]*x[...,j]*x[...,l] for i,j,l in itertools.combinations_with_replacement(range(4),3) if (i,j,l)!=(k,k,k)]
 return np.stack(f,axis=-1)
def fit(kind,mask):
 return [np.linalg.lstsq(np.vstack((feat(ref[mask],k,kind),feat(bx[(ids==k)&(np.abs(by)<8)],k,kind))),np.r_[raw[mask,k],by[(ids==k)&(np.abs(by)<8)]],rcond=None)[0] for k in range(4)]
def pred(x,kind,c):return np.stack([feat(x,k,kind)@c[k] for k in range(4)],axis=-1)
for kind in range(3):
 errs=[]
 for t in np.array_split(np.random.default_rng(2).permutation(np.flatnonzero(good)),5):
  mask=good.copy(); mask[t]=False; errs.append(np.sqrt(np.mean((pred(ref[t],kind,fit(kind,mask))-raw[t])**2)))
 print('sensor',kind,'CV',np.round(errs,7),'coeff base',np.array(fit(kind,good))[:,:6].round(5).tolist())
nx,ny=s['nx'],s['ny']; dx=s['Lx']/(nx-1); dy=s['Ly']/(ny-1); rob=np.repeat([s['robin_x'],s['robin_y']],2); ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; eye=sparse.eye(nx*ny,format='csc'); vdiag=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); w=np.array([p['w_pulse'] for p in pulses]); yy=np.array([p['y_pulse_raw'] for p in pulses]).reshape(3,-1,4)
def forward(z,q=(1,1,1,1)):
 ax=arc._axis(s['kappa_x'],s['vel_x'],dx,z[:2]*rob[:2]); ay=arc._axis(s['kappa_y'],s['vel_y'],dy,z[2:]*rob[2:]);
 def rescale(M,f):
  M=M.tolil(); n=M.shape[0]
  for a,b,t in [(0,1,f[0]),(n-1,n-2,f[1])]:
   d=M[a,b]*(t-1); M[a,b]+=d; M[a,a]-=d
  return M.tocsc()
 L=sparse.kron(sparse.eye(ny),rescale(ax,q[:2]),format='csc')+sparse.kron(rescale(ay,q[2:]),sparse.eye(nx),format='csc')+vdiag
 return np.vstack([splu(L+u*eye).solve(rhs)[rows] for u in s['mu_shifts']])
def residual(z,kind,c):return pred((w@forward(z[:4],z[4:] if len(z)>4 else (1,)*4).T).reshape(3,-1,4),kind,c)-yy
if __name__=='__main__':
 for kind in (0,1,2):
  c=fit(kind,good); ranks=[]
  for z in itertools.product((0.,1.),repeat=4):
   e=residual(np.array(z),kind,c); ranks.append((np.sqrt(np.mean(e**2)),z,np.sqrt(np.mean(e**2,axis=(1,2))).round(7)))
  print('walls',kind,sorted(ranks)[:4]);
  for n in (4,8):
   for test in (-1,0,1,2):
    take=np.arange(3)!=test; x0=np.r_[sorted(ranks)[0][1],np.ones(n-4)]; sol=least_squares(lambda z:residual(z,kind,c)[take].ravel(),x0,bounds=(np.r_[[-.2]*4,[.2]*(n-4)],np.r_[[4.]*4,[4.]*(n-4)]),diff_step=1e-4)
    e=residual(sol.x,kind,c); print('fitted',kind,n,test,sol.x.round(6),np.sqrt(np.mean(e**2,axis=(1,2))).round(7))
