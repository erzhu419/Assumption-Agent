import numpy as np,pickle,glob,runpy,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*01_solver.py')[0]); z=a['_inverse'](s['train']['x'],a['_calibration'](s['panel_a'])); y=s['train']['y']; Y=s['train']['traj']
def fit(A,Y): return np.linalg.lstsq(A,Y,rcond=1e-10)[0]
def one(u):return np.ones((len(u),1))
forms={'sin':lambda u,f:np.sin(u*f),'tanh':lambda u,f:np.tanh(u*f),'atan':lambda u,f:np.arctan(u*f),'rat':lambda u,f:u/(1+(f*u)**2),'erf':lambda u,f:__import__('scipy').special.erf(u*f),'sin+cos':lambda u,f:np.column_stack([np.sin(u*f),np.cos(u*f)]),'shifted_sin':lambda u,f:np.sin(u*f[:4]+f[4:]),'shifted_tanh':lambda u,f:np.tanh(u*f[:4]+f[4:]),'shifted_rat':lambda u,f:(u+f[4:])/(1+(f[:4]*(u+f[4:]))**2),'sq':lambda u,f:u*u}
results=[]
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u']; v=s[p][k]; candidates=[]
 for name,fun in forms.items():
  for lin in [False,True]:
   def A(u,f):return np.column_stack([one(u),*( [u] if lin else []),fun(u,f)])
   def residual(f):
    m=A(u,f);return (m@fit(m,v)-v).ravel()
   opts=[]
   for j in [.4,1,1.7,2.4,3.2,4.]:
    f=np.repeat(j,4); shift=name.startswith('shifted'); f=np.r_[f,np.zeros(4)] if shift else f
    lo=np.r_[np.full(4,.03),np.full(4,-3)] if shift else .03; hi=np.r_[np.full(4,6.),np.full(4,3)] if shift else 6.
    r=least_squares(residual,f,bounds=(lo,hi),max_nfev=90);opts.append((np.sqrt(np.mean(r.fun**2)),r.x))
   e,f=min(opts,key=lambda q:q[0]); candidates.append((e,name,lin,f.copy()))
 print(p,k,[(round(e,6),name,lin,np.round(f,3)) for e,name,lin,f in sorted(candidates,key=lambda q:q[0])[:10]],flush=True)
 results.append(sorted(candidates,key=lambda q:q[0])[:4])
np.save('/tmp/separable.npy',np.array(results,dtype=object),allow_pickle=True)
