import runpy,pickle,numpy as np
from scipy.optimize import minimize_scalar
m=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_c']; x=a['u']; ls=m['_ls']; af=m['_affine']; p,r=m['_panel_operators'](a)
for name,f in [('v1',p),('v2',r)]: print('archive',name,np.sqrt(np.mean((f(x)-a[name])**2)))
func={'satabs':lambda z,k:z/(1+k*abs(z)), 'satsq':lambda z,k:z/(1+k*z*z), 'sq':lambda z,k:z*z/(1+k*z*z), 'tanh':lambda z,k:np.tanh(k*z), 'sin':lambda z,k:np.sin(k*z), 'cos':lambda z,k:np.cos(k*z), 'cub':lambda z,k:z**3/(1+k*z*z), 'expabs':lambda z,k:np.exp(-k*abs(z)), 'abs':lambda z,k:abs(z), 'softsign2':lambda z,k:z*z/(1+k*abs(z))}
for name in ('v1','v2'):
 y=a[name]; print(name)
 for kn,fun in func.items():
  def err(k):
   F=np.column_stack((af(x),fun(x,k))); return np.mean((F@ls(F,y)-y)**2)
  opts=[minimize_scalar(err,bounds=(lo,hi),method='bounded') for lo,hi in [(0.001,1),(1,3),(3,8)]]; o=min(opts,key=lambda o:o.fun); print(kn,round(o.x,5),round(o.fun**.5,6))
 q=np.array([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]).T
 F=np.column_stack((af(x),q)); print('quadratic',np.sqrt(np.mean((F@ls(F,y)-y)**2))); print('coefs',np.round(ls(F,y),3))
