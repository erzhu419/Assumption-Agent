import sys;sys.path.insert(0,'/holdout_task')
import batch_solver as m
from laws12 import *
t=s['train'];N=len(t['x']);rng=np.random.default_rng(9873)
for i in range(12):
 order=np.arange(N) if not i else rng.permutation(N);a,b=order[:52],order[52:]; inp=dict(s,train={k:v[a] if k!='summary' else v for k,v in t.items()},test={'x':t['x'][b]}); o=m.solve_batch(inp); print(i,*[np.mean(abs(o[k]-t[j][b])) for k,j in [('y_hat','y'),('state_pred','traj')]], 'traj per horizon',np.round(np.mean(abs(o['state_pred']-t['traj'][b]),0),4))
G=m._features(s,t['x']);T=t['traj']; tt=s['eval_times'];print('times',tt,'telemetry',s['telemetry_vec']);print('coefs',np.round(ls(G[:52],np.c_[t['y'][:52],T[:52,2],T[:52,-1]]),4));
for mode in ['rise','decay','linear','rise_poly','rise_const']:
 def basis(r):
  g=np.exp(-tt[:,None]*r)
  return {'rise':1-g,'decay':g,'linear':np.c_[tt,1-g[:,0]],'rise_poly':np.c_[tt**2,1-g[:,0]],'rise_const':np.c_[1+tt*0,1-g[:,0]]}[mode]
 def res(r):
  g=basis(r);return (g@ls(g,T[:52].T)-T[:52].T).ravel()
 q=least_squares(res,[.45,2.1] if mode in ['rise','decay'] else [1.2],bounds=(.01,20)); print(mode,q.x,np.sqrt(np.mean(q.fun*q.fun)))
