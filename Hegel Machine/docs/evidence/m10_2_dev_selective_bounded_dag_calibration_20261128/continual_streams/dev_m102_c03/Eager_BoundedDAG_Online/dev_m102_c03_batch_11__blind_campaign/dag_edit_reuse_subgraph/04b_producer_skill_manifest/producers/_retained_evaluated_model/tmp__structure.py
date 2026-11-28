exec(open('/tmp/probe.py').read().split('for name,p')[0])
from scipy.optimize import least_squares, minimize_scalar
np.set_printoptions(precision=5,suppress=True)
def bias(v):return np.column_stack([np.ones(len(v)),v])
def ls(f,y):return np.linalg.lstsq(f,y,rcond=1e-10)[0]
u,v=s['panel_a'].values();
for j in range(4):
 p,q=u[:,j],u[:,(j+1)%4]; f=bias(np.c_[p,q,p*q,p**3]); keep=np.ones(len(u),bool)
 for k in range(10):
  c=ls(f[keep],v[keep,j]); r=abs(f@c-v[:,j]); keep=r<max(.001,4*np.median(r))
 print('A',j,c,'resid',np.quantile(r,[.5,.75,.9,.98,1]),'inlier',sum(keep))
u,v1,v2=s['panel_c'].values();
z=a._coordinates(s,x)
from itertools import product
forms={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'exp':lambda w:np.expm1(-w*w),'atan':np.arctan,'soft':lambda w:w/(1+abs(w)),'sq':lambda w:w*w,'cubic':lambda w:w**3,'lin':lambda w:w}
for j,v in enumerate([v1,v2]):
 res=[]
 for nm,g in forms.items():
  for nm2,h in [(None,None),*forms.items()]:
   if nm2==nm:continue
   def mat(U,par):return bias(g(par[0]*U) if h is None else np.c_[g(par[0]*U),h(par[1]*U)])
   def err(par):
    f=mat(u,par);return (f@ls(f,v)-v).ravel()
   fit=least_squares(err,[1.5] if h is None else [1.5,1.3],bounds=(.06,5.5),max_nfev=100)
   res.append((np.mean(abs(fit.fun)),nm,nm2,np.round(fit.x,4)))
 print('C',j,sorted(res,key=lambda a:a[0])[:14]); print('affine coeff',ls(bias(u),v))
t=s['eval_times']; traj=tr['traj']; print('traj spec',np.linalg.svd(traj,compute_uv=False))
for form in ['exp','exp_plus','expdecay']:
 def q(p):
  q = -np.expm1(-t[:,None]*p) if form!='expdecay' else np.exp(-t[:,None]*p)
  return bias(q) if form=='exp_plus' else q
 def err(p):
  f=q(p);return (f@ls(f,traj.T)-traj.T).ravel()
 r=least_squares(err,[.4,1.8],bounds=(.001,12)); print(form,r.x,np.mean(abs(r.fun)))
