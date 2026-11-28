import copy,glob,importlib.util,pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52
D=lambda x:np.c_[np.ones(len(x)),x]
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
print('time',s['eval_times'],'telemetry',s['telemetry_vec'])
u,v=s['panel_a']['u'],s['panel_a']['v']; print('A affine',ls(D(u),v));print('A abs resid',np.quantile(abs(D(u)@ls(D(u),v)-v),[0,.5,.8,.9,1],axis=0)); print('traj singular values',np.linalg.svd(tr['traj'][:52],compute_uv=False))
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  sp=importlib.util.spec_from_file_location('m',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
  ss=copy.deepcopy(s);ss['train']={k:(v[:n] if k!='summary' else v) for k,v in tr.items()};ss['test']={'x':tr['x'][n:]};o=m.solve_batch(ss)
  print('archive',f.split('_')[-2], np.mean(abs(o['y_hat']-tr['y'][n:])),np.mean(abs(o['state_pred']-tr['traj'][n:])))
 except Exception as e: print(f,e)
u=s['panel_c']['u']; quad=np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]).T
fs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'softplus':lambda z:np.logaddexp(0,z),'sat':lambda z:z/(1+abs(z)),'exp':np.exp,'rational':lambda z:1/(1+z*z),'sq':np.square,'abs':abs,'cube':lambda z:z**3}
for vk in ('v1','v2'):
 v=s['panel_c'][vk]; fits=[]
 def rms(a):return np.sqrt(np.mean((a@ls(a,v)-v)**2))
 for nm,g in fs.items():
  for linear in (False,True):
   def design(k):return D(np.c_[u,g(k*u)] if linear else g(k*u))
   opt=minimize_scalar(lambda k:rms(design(k)),bounds=(.08,5),method='bounded'); fits.append((opt.fun,nm,linear,opt.x))
 for j in range(1,4):
  p=u*np.roll(u,j,axis=1); fits.append((rms(D(np.c_[u,p])),'pair'+str(j),True,0));fits.append((rms(D(p)),'pair'+str(j),False,0))
 fits.append((rms(D(np.c_[u,quad])),'fullquad',True,0))
 print(vk,sorted(fits)[:12])
