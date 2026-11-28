import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];p=s['panel_c'];u=p['u'];v=p['v1']; I=lambda x:np.c_[np.ones(len(x)),x]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
funs={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'rat':lambda a:a/(1+a*a),'erf':erf,'atan':np.arctan,'cos':np.cos}
for name,f in funs.items():
 for extra in ['none','individual','cube','pair','square','cos','sin','tanh']:
  def D(a,q):
   g=f(a*q[:4]) if extra=='individual' else f(a*q[0]); b=I(np.c_[a,g]); i,j=np.triu_indices(4,1)
   if extra=='cube':return np.c_[b,a**3]
   if extra=='pair':return np.c_[b,a[:,i]*a[:,j]]
   if extra=='square':return np.c_[b,a*a]
   if extra in ['cos','sin','tanh']:return np.c_[b,funs[extra](a*q[-1])]
   return b
  def E(q):
   a=D(u,q); h=np.linalg.pinv(a); rr=a@(h@v)-v; lev=np.sum(a*h.T,axis=1); return (rr/np.maximum(1-lev[:,None],.02)).ravel()
  dim=4 if extra=='individual' else (2 if extra in ['cos','sin','tanh'] else 1)
  z=min([least_squares(E,np.full(dim,q),bounds=(.1,7),max_nfev=90) for q in [.7,1.5,2.15,3.4]],key=lambda a:a.cost)
  print(name,extra,z.x.round(3),np.linalg.norm(z.fun)/np.sqrt(v.size))
print('telemetry',s['telemetry_vec']); print('summary',s['train']['summary']); print('raw stats',s['train']['y'].mean(),s['train']['y'].std(),s['train']['traj'][:,-1].mean(),s['train']['traj'][:,-1].std()); print('train/test ranges',s['train']['x'].min(0),s['train']['x'].max(0),s['test']['x'].min(0),s['test']['x'].max(0))
