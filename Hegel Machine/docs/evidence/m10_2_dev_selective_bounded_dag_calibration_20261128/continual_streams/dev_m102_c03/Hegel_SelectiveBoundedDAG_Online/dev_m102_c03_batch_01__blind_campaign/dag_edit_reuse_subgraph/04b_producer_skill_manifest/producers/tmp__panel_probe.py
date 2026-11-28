import numpy as np,pickle,runpy,glob
from scipy.optimize import minimize_scalar
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=st['train']
arc=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); sparse=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; sink=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py')['run_node']
def fold(a,b):
 return dict(st,train={k:v[a] for k,v in tr.items() if k!='summary'},test={'x':tr['x'][b]})
def err(p,b): return np.mean(abs(p-np.column_stack([tr['y'],tr['traj']])[b]),axis=0)
for key in ('panel_a','panel_b','panel_c'):
 p=st[key]; u=p['u']; h=np.c_[np.ones(len(u)),u];print(key,'u range',u.min(0),u.max(0))
 for name,v in p.items():
  if name=='u':continue
  af=np.linalg.lstsq(h,v,rcond=None)[0]; r=np.linalg.norm(v-h@af,axis=1); print(name,'affine',af.round(3),'residual quantiles',np.quantile(r,[0,.5,.9,.95,1]).round(4))
  for lab,fn in [('negexp',lambda u,a:-np.expm1(-a*u*u)),('tanh',lambda u,a:np.tanh(a*u)),('atan',lambda u,a:np.arctan(a*u)),('rational',lambda u,a:u*u/(1+a*u*u)),('softsign',lambda u,a:u/(1+a*abs(u)))]:
   def obj(a):
    mat=np.c_[h,fn(u,a)]; e=v-mat@np.linalg.lstsq(mat,v,rcond=None)[0]; return np.sqrt(np.mean(e**2))
   o=minimize_scalar(obj,bounds=(.002,8),method='bounded');print(lab,round(o.x,4),round(o.fun,5),'fixed',[(a,round(obj(a),5)) for a in (.22,.2962,1.25)])
for b in [np.arange(52,72),np.arange(20),np.arange(20,40),np.arange(40,60)]:
 a=np.setdiff1d(np.arange(72),b);s=fold(a,b); d=sparse({'__input__':s}); dp=np.c_[d['y_hat'],d['state_pred']]; p=arc['_panel_pred'](s,1,.3,'sat');bl=sink({'node_00':d,'node_01':{'panel_pred':p}});print('fold',b[0], 'sparse',err(dp,b).round(4),'panel',err(p,b).round(4),'blend',err(np.c_[bl['y_hat'],bl['state_pred']],b).round(4))
print('target ranges',np.column_stack([tr['y'],tr['traj']]).min(0),np.column_stack([tr['y'],tr['traj']]).max(0));print('trajectory sv',np.linalg.svd(tr['traj'],compute_uv=False));print('x range',tr['x'].min(0),tr['x'].max(0),st['test']['x'].min(0),st['test']['x'].max(0))
