import importlib.util,itertools,pickle
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/beta_m100_c03_batch_06_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; y=m._calibrate(s,s['y_obs_raw']); z=s['Z_nuisance']; nx,ny=s['nx'],s['ny']; I=eye(nx*ny,format='csc'); sn=s['sensor_ij']; inds=sn[:,0]+nx*sn[:,1]; rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T; sizes=[nx,ny]; hs=[s['L'+d]/(n-1) for d,n in zip('xy',sizes)]; axes=[m._axis(s['kappa_'+d],s['vel_'+d],h) for d,h in zip('xy',hs)]; diag=diags(s['V_reaction_2d'].ravel(order='F'))
def response(p):
 mats=[]
 for k,(d,n,h,a) in enumerate(zip('xy',sizes,hs,axes)):
  v=np.zeros(n); v[[0,-1]]=np.asarray(p[2*k:2*k+2])*s['robin_'+d]/h; mats.append(a+diags(v))
 a,b=mats; op=(kron(eye(ny),a)+kron(b,eye(nx))+diag).tocsc(); return np.vstack([splu(op+v*I).solve(rhs)[inds] for v in s['mu_shifts']])
def fit(R,train):
 A=R[train]; Z=z[train]; Y=y[train]; V=np.c_[A,Y]; P=V-Z@np.linalg.lstsq(Z,V,rcond=None)[0]; B,q=P[:,:-1],P[:,-1]; w,_=m._box_qp(B.T@B+1e-12*np.eye(B.shape[1]),-B.T@q,s['u_box'],s['total_budget']); g=np.linalg.lstsq(Z,Y-A@w,rcond=None)[0]; return w,g
n=len(y); allrows=np.arange(n); groups=[np.flatnonzero(np.tile(np.arange(len(inds))//4,len(s['mu_shifts']))==k) for k in range(len(inds)//4)]
for p in itertools.product((0,1),repeat=4):
 R=response(p); w,g=fit(R,allrows); r=R@w+z@g-y; cv=[]
 for te in groups:
  tr=np.setdiff1d(allrows,te); u,v=fit(R,tr); cv.append(np.sqrt(np.mean((R[te]@u+z[te]@v-y[te])**2)))
 print(p,'rmse',round(np.sqrt(np.mean(r*r)),7),'worst grouped CV',round(max(cv),7),'w min/max',w.min().round(4),w.max().round(4))
def resid(p):
 R=response(p); w,g=fit(R,allrows); return R@w+z@g-y
p=least_squares(resid,np.full(4,.1),bounds=(-.5,2),diff_step=1e-4,gtol=1e-12); print('continuous wall scales',p.x,'rmse',np.sqrt(np.mean(p.fun**2)),'diagnostics',type(i.get('campaign_diagnostics')).__name__)
