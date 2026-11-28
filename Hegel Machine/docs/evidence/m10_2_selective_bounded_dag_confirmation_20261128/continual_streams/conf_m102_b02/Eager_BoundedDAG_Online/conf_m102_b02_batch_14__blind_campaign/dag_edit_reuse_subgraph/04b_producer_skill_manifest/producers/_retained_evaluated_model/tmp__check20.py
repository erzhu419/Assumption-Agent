import sys,pickle,numpy as np,runpy
sys.path.insert(0,'/holdout_task')
import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train']; m._panel_a(s,np.r_[tr['x'],s['test']['x']],True)
for key in ['v1','v2']:
 u=s['panel_c']['u'];v=s['panel_c'][key]; print('C',key)
 from scipy.optimize import minimize_scalar
 for label,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('erf',m.erf),('atan',np.arctan),('rat',lambda x:x/(1+x*x)),('soft',lambda x:x/np.sqrt(1+x*x))]:
  def err(q):
   F=m._design(np.c_[u,f(q*u)]);return np.mean((F@m._fit(F,v)-v)**2)
  r=min([minimize_scalar(err,bounds=(q,q+.45),method='bounded') for q in np.arange(.1,5,.45)],key=lambda r:r.fun)
  print(label, round(r.x,4),round(r.fun**.5,6))
