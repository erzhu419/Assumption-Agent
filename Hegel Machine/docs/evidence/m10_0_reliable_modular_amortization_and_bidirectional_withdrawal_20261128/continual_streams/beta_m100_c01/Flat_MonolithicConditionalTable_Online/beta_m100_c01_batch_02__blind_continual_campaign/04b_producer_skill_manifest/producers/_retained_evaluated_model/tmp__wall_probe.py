import numpy as np,pickle,itertools,importlib.util
from scipy.sparse import diags,eye,kron,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; p=item['campaign_diagnostics']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; singles=c['single_bank_sweeps']; bank=singles['bank_id']
coef=np.array([np.linalg.lstsq(np.c_[np.ones(len(x)+sum(bank==j)),np.r_[x[:,j],singles['ref_signal'][bank==j]]],np.r_[y[:,j],singles['raw_readout'][bank==j]],rcond=None)[0] for j in range(4)])
Y=np.column_stack([((q['y_pulse_raw'].reshape(-1,4)-coef[:,0])/coef[:,1]).ravel() for q in p]); W=np.column_stack([q['w_pulse'] for q in p]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z,mode='reduced')[0]; nx,ny=s['nx'],s['ny']; slots=s['sensor_ij']@np.array([1,nx]); rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']])@W
print('pulse RMS by shift/pulse',np.sqrt(np.mean(Y.reshape(3,-1,len(p))**2,axis=1))); print('sensor diagonal sweep',coef)

def axis(a,n,bc):
 k=s['kappa_'+a]; h=s['L'+a]/(n-1); v=s['vel_'+a][1:-1]/h; f=(k[:-1]+k[1:])/(2*h*h); A=diags((np.r_[-f[:-1]-np.maximum(v,0),0],np.r_[0,f[:-1]+f[1:]+abs(v),0],np.r_[0,-f[1:]+np.minimum(v,0)]),(-1,0,1)).toarray(); r=s['robin_'+a]/h
 for i,j,t in [(0,1,bc[0]),(n-1,n-2,bc[1])]:
  d=k[i]/h**2
  if isinstance(t,str) and t=='P': A[i,i]=d+k[i]/h**2;A[i,j]=-d;A[i,n-1-i]=-k[i]/h**2
  elif isinstance(t,str) and t=='D': A[i,i]=d+r
  else:
   alpha,beta={'R':(1,1),'N':(1,0),'H':(2,0),'B':(2,1),'W':(1,-1)}[t] if isinstance(t,str) else t
   A[i,i]=alpha*d+beta*r; A[i,j]=-alpha*d
 return csc_matrix(A)

def pred(bc):
 A=kron(eye(ny),axis('x',nx,bc[:2]),format='csc')+kron(axis('y',ny,bc[2:]),eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); return np.vstack([splu(A+mu*I).solve(rhs)[slots] for mu in s['mu_shifts']])

def residual(bc,proj=False):
 E=pred(bc)-Y
 return E-Q@(Q.T@E) if proj else E

def rms(e): return np.sqrt(np.mean(e*e,axis=0))
for proj in (False,True):
 bcs=list(itertools.product('RNPHBD',repeat=4)); scores=[]
 for b in bcs:
  E=residual(b,proj); scores.append((np.linalg.norm(E),''.join(b),rms(E)))
 for score in sorted(scores)[:8]: print('projected' if proj else 'direct',score)
 for b in ('RRRR','NNNN','PPPP'):
  print(b,rms(residual(b,proj)))
 for held in (-1,0,1,2):
  tr=np.arange(len(p))!=held
  fit=least_squares(lambda t:residual(t.reshape(4,2),proj)[:,tr].ravel(),np.tile([1.,0.5],4),bounds=(np.tile([0.,0.],4),np.tile([20.,20.],4)),gtol=1e-12,ftol=1e-12,xtol=1e-12)
  print('continuous',proj,held,fit.x.round(5),rms(residual(fit.x.reshape(4,2),proj)))
