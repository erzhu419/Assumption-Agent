import pickle,numpy as np
from itertools import combinations
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-9)[0]
aff=lambda x: np.column_stack([np.ones(len(x)),x])
for key,outs in [('panel_b',['v']),('panel_c',['v1','v2'])]:
 u=s[key]['u']; pairs=np.array(list(combinations(range(4),2))); p=u[:,pairs[:,0]]*u[:,pairs[:,1]]; c=u*np.roll(u,-1,axis=1)
 for out in outs:
  v=s[key][out]; results=[]
  def ev(x):
   x=aff(x); hat=x@np.linalg.pinv(x); r=v-x@fit(x,v); return np.sqrt(np.mean((r/(1-np.diag(hat))[:,None])**2)),np.sqrt(np.mean(r*r))
  for n,f in [('linear',u),('allpair',p),('cycle',c),('squares',u*u),('lin+sq',np.c_[u,u*u]),('lin+cube',np.c_[u,u**3]),('lin+cycle',np.c_[u,c]),('lin+pair',np.c_[u,p]),('quadratic',np.c_[u,u*u,p])]: results.append((*ev(f),n,0))
  for n,fn in [('sin',lambda w:np.sin(w*u)),('cos',lambda w:np.cos(w*u)),('tanh',lambda w:np.tanh(w*u)),('soft',lambda w:u/(1+w*abs(u))),('rat',lambda w:u/(1+w*u*u)),('gauss',lambda w:np.exp(-w*u*u)),('cos-cycle',lambda w:np.cos(w*c)),('sin-cycle',lambda w:np.sin(w*c))]:
   for lin in [False,True]:
    f=lambda w: np.c_[u,fn(w)] if lin else fn(w)
    opt=minimize_scalar(lambda w:ev(f(w))[0],bounds=(.05,5),method='bounded'); results.append((*ev(f(opt.x)),n+(' + lin' if lin else ''),opt.x))
  print(key,out,sorted(results)[:14])
for j in ['03','05']:
 import runpy
 g=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c03_batch_'+j+'_solver.py'); u=s['panel_a']['u']; z=g['_calibrate'](s['panel_a'],s['panel_a']['v']); print('cal',j,sorted(np.linalg.norm(z-u,axis=1)))
