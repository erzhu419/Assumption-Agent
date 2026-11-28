import sys,pickle,itertools
import numpy as np
from scipy import sparse
from scipy.optimize import root
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m100_01_batch_03_solver as prev
import dev_beta_m100_01_batch_02_solver as periodic
import dev_beta_m100_01_batch_01_solver as reflecting
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; raw=c['raw_rosette']; bank=c['single_bank_sweeps']
exps=[p for p in itertools.product(range(4),repeat=4) if sum(p)<=3]; exps.sort(key=lambda p:(sum(p),p))
old=[(0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,0,1),(1,1,0,0),(3,0,0,0)]
def design(x,j,p): return np.prod(np.roll(x,-j,axis=1)[:,None,:]**np.array(p)[None,:,:],axis=2)
def fit(A,b):
 f=np.linalg.lstsq(A,b,rcond=None)[0]
 for _ in range(15):
  e=b-A@f; scale=max(1e-6,1.4826*np.median(abs(e-np.median(e)))); q=1/np.hypot(1,e/(2.5*scale)); f=np.linalg.lstsq(A*q[:,None],b*q,rcond=None)[0]
 e=b-A@f; keep=abs(e-np.median(e))<max(2e-4,8*scale)
 return np.linalg.lstsq(A[keep],b[keep],rcond=None)[0],keep
coefs={}
for name,p in [('archived',old),('full',exps)]:
 fs=[]
 for j in range(4):
  m=bank['bank_id']==j; bx=np.zeros((sum(m),4)); bx[:,j]=bank['ref_signal'][m]; A=design(np.vstack((x,bx)),j,p); b=np.r_[raw[:,j],bank['raw_readout'][m]]
  f,k=fit(A,b); pred=np.empty(len(b));
  for fold in range(5):
   train=np.arange(len(b))%5!=fold; ff,_=fit(A[train],b[train]); pred[~train]=A[~train]@ff
  print(name,j,'inliers',sum(k),'OOF clean RMS/max',np.sqrt(np.mean((b-pred)[k]**2)),max(abs((b-pred)[k])), 'terms',[(a,round(v,6)) for a,v in zip(p,f) if abs(v)>0.001]); fs.append(f)
 coefs[name]=(fs,p)

def model(i,j):
 n=s['n'+j]; h=s['L'+j]/(n-1); a=(s['kappa_'+j],s['vel_'+j],h)
 return prev._axis(*a,s['robin_'+j]) if i=='O' else reflecting._axis(*a) if i=='R' else periodic._axis(*a)
Z=s['Z_nuisance']; nr=s['nuisance_ridge']; n=len(Z); ij=s['sensor_ij']; wall=np.tile((ij[:,0]==0)|(ij[:,0]==s['nx']-1)|(ij[:,1]==0)|(ij[:,1]==s['ny']-1),len(s['mu_shifts'])); groups=np.arange(n)//4
for name,(fs,p) in coefs.items():
 def fwd(t): return np.array([(design(t[None,:],j,p)@fs[j])[0] for j in range(4)])
 y=[]; errs=[]
 for obs in s['y_obs_raw'].reshape(-1,4):
  r=root(lambda t:fwd(t)-obs,obs,tol=1e-11); y.extend(r.x); errs.append(max(abs(r.fun)))
 y=np.array(y); print('inverse',name,max(errs)); results=[]
 for a,b in itertools.product('ORP',repeat=2):
  L=sparse.kron(sparse.eye(s['ny']),model(a,'x'),format='csc')+sparse.kron(model(b,'y'),sparse.eye(s['nx']),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); S,U=prev._forward(s,L)
  pred=np.empty(n)
  for group in np.unique(groups):
   m=groups!=group; z=Z[m]; v=np.c_[S[m],y[m]]; v=v-z@np.linalg.solve(z.T@z+nr*np.eye(z.shape[1]),z.T@v); w,nu=prev._qp(v[:,:-1],v[:,-1],s['source_ridge'],s['u_box'],s['total_budget']); gam=np.linalg.solve(z.T@z+nr*np.eye(z.shape[1]),z.T@(y[m]-S[m]@w)); pred[~m]=S[~m]@w+Z[~m]@gam
  e=y-pred; results.append((np.sqrt(np.mean(e**2)),a+b,np.sqrt(np.mean(e[wall]**2)),np.sqrt(np.mean(e[~wall]**2)),[round(np.sqrt(np.mean(z*z)),5) for z in np.split(e,3)]))
 print('LO rosette RMS, candidate, wall, interior, shifts:',sorted(results))
