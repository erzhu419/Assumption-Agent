import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
q=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(q);q.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; yy=a['y']; tt=a['traj']; t=s['eval_times']; p=dict(s,test={'x':a['x'][:0]}); simple,full=m._design(p)
tr=np.arange(52); va=np.arange(52,72)
y=m._ls(full[tr],yy[tr]); c=m._ls(full[tr],tt[tr]);print('y cross',y[9:].reshape(4,4)); print('state cross by time',np.round(c[9:,[0,3,7]].reshape(4,4,3),4))
def basis(r):return -np.expm1(-t[:,None]*r)
b=least_squares(lambda r:(basis(r)@m._ls(basis(r),tt[tr].T)-tt[tr].T).ravel(),[.4,1.8]).x
print('rates',b,'amp weights',np.round(m._ls(full[tr],m._ls(basis(b),tt[tr].T).T),4)); print('A',a['summary']);
z=m._calibration(s['panel_a'], a['x']); idxs=[va,np.argsort(abs(yy))[-20:],*[np.argsort(z[:,i])[-20:] for i in range(4)],*[np.argsort(z[:,i])[:20] for i in range(4)],*[np.random.default_rng(i).permutation(72)[:20] for i in range(16)]]
for j,va in enumerate(idxs):
 tr=np.setdiff1d(np.arange(72),va); o=m.solve_batch(dict(s,train={k:(v[tr] if k!='summary' else v) for k,v in a.items()},test={'x':a['x'][va]}));ey=abs(o['y_hat']-yy[va]); et=abs(o['state_pred']-tt[va]);print(j,np.round([ey.mean(),et.mean(),et.mean(0).max(),ey.max(),et.max()],6))
