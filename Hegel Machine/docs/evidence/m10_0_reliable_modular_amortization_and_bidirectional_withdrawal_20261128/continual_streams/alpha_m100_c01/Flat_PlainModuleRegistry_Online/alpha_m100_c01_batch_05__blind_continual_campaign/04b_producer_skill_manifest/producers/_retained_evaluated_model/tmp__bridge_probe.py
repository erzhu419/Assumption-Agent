import pickle,numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; S=C['single_channel_sweeps']; P=np.array([2,3,0,1]); ok=np.max(abs(Y),axis=1)<5; axes=np.count_nonzero(X,axis=1)==1
families={'sin':np.sin,'erf':erf,'tanh':np.tanh,'atan':np.arctan,'gauss':lambda u:u*np.exp(-u*u),'lorentz':lambda u:u/(1+u*u),'softsign':lambda u:u/np.sqrt(1+u*u),'sinx2':lambda u:u*np.sinc(u*u/np.pi),'cos':lambda u:np.cos(u)-1,'gauss_even':lambda u:np.expm1(-u*u),'rational_even':lambda u:-u*u/(1+u*u)}

def profile(blocks, family, powers, bounds):
 f=families[family]
 def inner(k, ret=False):
  rs=[]; bs=[]
  for x,y in blocks:
   a=np.column_stack([x**p for p in powers]+[f(k*x)]); b=np.linalg.lstsq(a,y,rcond=None)[0]; rs.extend(y-a@b); bs.append(b)
  return np.asarray(bs) if ret else np.mean(np.square(rs))
 grid=np.linspace(*bounds,50); ds=[inner(k) for k in grid]; i=np.argmin(ds); sol=minimize_scalar(inner,bounds=(grid[max(0,i-1)],grid[min(i+1,len(grid)-1)]),method='bounded'); return np.sqrt(sol.fun),family,powers,sol.x,inner(sol.x,True)
blocks=[]
for j in range(4):
 a=(S['lab_id']==j)&(abs(S['cmd_stress'])<5); b=ok&axes&(X[:,j]!=0); blocks.append((np.r_[S['true_stress'][a],X[b,j]],np.r_[S['cmd_stress'][a],Y[b,j]]))
results=[profile(blocks,f,p,(.12,5)) for f in list(families)[:8] for p in [(0,1),(0,1,3)]]
for r in sorted(results,key=lambda r:r[0])[:10]: print('diag',r[:4],np.round(r[4],6).tolist())
_,f,p,k,b=min(results,key=lambda r:r[0]); diag=np.column_stack([np.column_stack([X[:,j]**a for a in p]+[families[f](k*X[:,j])])@b[j] for j in range(4)]); R=Y-diag
blocks=[(X[ok,P[j]],R[ok,j]) for j in range(4)]
for f in ['cos','gauss_even','rational_even']:
 for p in [(1,),(0,1),(1,2),(1,3)]:
  r=profile(blocks,f,p,(.15,5)); print('cross',r[:4],np.round(r[4],6).tolist())
print('dataset keys',list(D),'chamber type',type(D.get('chamber_verification_trials')).__name__)
