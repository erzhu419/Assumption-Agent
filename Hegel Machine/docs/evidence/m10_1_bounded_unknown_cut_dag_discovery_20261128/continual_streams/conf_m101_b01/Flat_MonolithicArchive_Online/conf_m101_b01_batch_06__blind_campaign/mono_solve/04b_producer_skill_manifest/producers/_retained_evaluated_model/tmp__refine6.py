from panel6 import *
from scipy.optimize import least_squares
from scipy.special import hyp2f1
u,v=b['u'],b['v']
forms={'erf':erf,'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'exp_gate':lambda z:z*np.exp(-z*z),'rat_gate':lambda z:z/(1+z*z),'alg_gate':lambda z:z/np.sqrt(1+z*z),'asinh':np.arcsinh,'log_gate':lambda z:z*np.log1p(z*z)}
for name,f in forms.items():
 for sep in (False,True):
  def design(x,p):return np.c_[af(x),f(x*np.exp(p))]
  def resid(p):
   d=design(u,p);return (d@ls(d,v)-v).ravel()
  opts=[least_squares(resid,np.full(4 if sep else 1,np.log(k)),bounds=(-2,1.5)) for k in (.6,1.3,2.5)]
  opt=min(opts,key=lambda o:o.cost);d=design(u,opt.x); w=ls(d,v);h=np.sum(d*np.linalg.pinv(d).T,axis=1)
  print(name,sep,'scale',np.round(np.exp(opt.x),4),'rmse',np.sqrt(np.mean(resid(opt.x)**2)), 'loo',np.sqrt(np.mean((resid(opt.x).reshape(v.shape)/(1-h[:,None]))**2)), 'ratio',np.round((w[1:5]*w[5:]).sum(1)/(w[5:]**2).sum(1),3))
for f in ['/tmp/hegel_continual_archive/conf_m101_b01_batch_%02d_solver.py'%k for k in (1,2,3,4)]:
 print('\n',f); print(''.join(line for line in open(f) if any(k in line for k in ('hyp2f1','exp(', 'tanh','cos(', 'erf(', 'sin(', ' / ', 'def _', '**'))))
print('time grid',s['eval_times']); print('traj centered singular',np.linalg.svd(Y[:52,1:]-Y[:52,1:].mean(0),compute_uv=False));print('B and C ranges',[(p['u'].min(),p['u'].max()) for p in (b,c)])
