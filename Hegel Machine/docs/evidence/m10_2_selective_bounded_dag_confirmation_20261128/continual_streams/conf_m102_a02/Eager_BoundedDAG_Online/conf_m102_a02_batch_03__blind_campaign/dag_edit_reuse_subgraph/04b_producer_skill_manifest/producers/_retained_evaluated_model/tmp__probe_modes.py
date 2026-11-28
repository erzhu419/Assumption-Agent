exec(open('/tmp/probe_dag.py').read().split('for k in (')[0])
from scipy.optimize import least_squares
from scipy.special import erf
U,S,V=np.linalg.svd(tr,full_matrices=False); modes=V[:2]; w=tr@modes.T
for fun in [lambda r:-np.expm1(-r*t),lambda r:np.exp(-r*t),lambda r: t*np.exp(-r*t)]:
 z=minimize_scalar(lambda r:np.linalg.norm(fun(r)-(fun(r)@modes.T)@modes)/np.linalg.norm(fun(r)),bounds=(.02,10),method='bounded'); print('time family',z.x,z.fun)
 print('rate profile',[(r,round(np.linalg.norm(fun(r)-(fun(r)@modes.T)@modes)/np.linalg.norm(fun(r)),5)) for r in [.2,.4,.6,.8,1,1.2,1.4,1.6,1.8,2,2.5,3]])
print('modes',modes.round(4)); print('reconstruction MAE',np.mean(abs(w@modes-tr)))
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; a=D(u,1)
 for name,v in p.items():
  if name=='u':continue
  b=np.linalg.lstsq(a,v,rcond=None)[0]; e=v-a@b; print('panel',k,name,'matrix singular',np.linalg.svd(b[1:],compute_uv=False),'residual singular',np.linalg.svd(e,compute_uv=False))
rng=np.random.default_rng(7)
A=D(x,1); Q=D(x,2); Y=np.c_[y,w]; n=52
fs={'tanh':np.tanh,'sin':np.sin,'erf':erf,'rat':lambda z:z/(1+z*z),'cubic':lambda z:z**3,'tanh2':lambda z:np.tanh(z)**2}
for i in range(3):
 z=Y[:,i]; direction=np.linalg.lstsq(A[:n],z[:n],rcond=None)[0][1:]; direction/=np.linalg.norm(direction)
 for name,f in fs.items():
  for d,B in ((1,A),(2,Q)):
   def design(p):return np.column_stack([B, f(A@p)])
   def fitp(p):
    H=design(p); c=np.linalg.lstsq(H[:n],z[:n],rcond=1e-10)[0]; return H@c-z,c
   starts=[np.r_[0.,direction],np.r_[0.,2*direction]]+list(rng.normal(size=(12,5)))
   sol=min((least_squares(lambda p:fitp(p)[0][:n],p,bounds=(-7,7),max_nfev=180) for p in starts),key=lambda o:o.cost)
   e,c=fitp(sol.x); print(i,name,d,'train/test',round(np.mean(abs(e[:n])),5),round(np.mean(abs(e[n:])),5),'p',sol.x.round(3),'nonlincoef',round(c[-1],3))
