import importlib.util,pickle,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; d=a['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; N=nx*ny
w=np.column_stack([p['w_pulse'] for p in d]); y=np.column_stack([m._calibrated_observations({**s,'y_obs_raw':p['y_pulse_raw']}) for p in d]); rhs=np.column_stack([q.ravel(order='F') for q in s['source_modes']])@w
i,j=s['sensor_ij'].T; pick=i+nx*j; I=sparse.eye(N,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
axes=[]
for axis in 'xy':
 k,v,r,L=[s[p+axis] for p in ('kappa_','vel_','robin_','L')]; h=L/(len(k)-1); A=m._axis_operator(k,v,r,L).tolil(); axes.append((A,k,h,r))
def pred(p):
 mats=[]
 for ia,(base,k,h,r) in enumerate(axes):
  A=base.copy()
  for side,(i,j,opp) in enumerate([(0,1,-1),(-1,-2,0)]):
   diff,leak,wrap=np.asarray(p).reshape(2,2,3)[ia,side]; v=k[i]/h**2
   A[i,i]=v*(diff+wrap)+r/h*leak; A[i,j]=-v*diff; A[i,opp]=-v*wrap
  mats.append(A.tocsc())
 A=sparse.kron(sparse.eye(ny),mats[0],format='csc')+sparse.kron(mats[1],sparse.eye(nx),format='csc')+V
 return np.vstack([splu(A+shift*I).solve(rhs)[pick] for shift in s['mu_shifts']])
def rms(e):return np.sqrt(np.mean(e**2,axis=0))
def report(name,p):
 e=pred(p)-y; print(name,'params',np.round(np.array(p).reshape(4,3),6).tolist(),'rms',np.round(rms(e),8),'shift rms',np.round(rms(e.reshape(3,-1).T),8),flush=True)
for name,p in [('robin',[1,1,0]*4),('reflect',[1,0,0]*4),('mirror',[2,0,0]*4),('periodic',[1,0,1]*4),('clamped-decoupled',[0,0,0]*4)]: report(name,p)
models={
 'leak-only':(np.tile([1.,0.,0.],4),[1,4,7,10]),
 'reflect-diff':(np.tile([1.,0.,0.],4),[0,3,6,9]),
 'diff-leak':(np.tile([1.,0.,0.],4),[0,1,3,4,6,7,9,10]),
 'diff-leak-wrap':(np.tile([1.,0.,0.],4),list(range(12))),
}
def fit(base,ind,cols,profile=False):
 p=base.copy()
 def fun(z):
  p[ind]=z; e=(pred(p)-y)[:,cols]; e=e-Q@(Q.T@e) if profile else e
  return e.ravel()
 opt=least_squares(fun,p[ind],max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-11)
 p[ind]=opt.x
 return p
for name,(base,ind) in models.items():
 p=fit(base,ind,[0,1,2]); report(name,p); print('leave-one-pulse-out rms', [round(float(rms((pred(fit(base,ind,[i for i in range(3) if i!=j]))-y)[:,j])),8) for j in range(3)])
 if name=='diff-leak-wrap': report('profiled all',fit(base,ind,[0,1,2],True))
