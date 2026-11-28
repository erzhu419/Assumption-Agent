import sys, pickle, numpy as np
sys.path[:0]=['/tmp','/holdout_task']
from baseline13 import solve_batch as base
from challenger13 import solve_batch as trial
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y'])
r=np.random.default_rng(550); splits=[np.arange(n)]+[r.permutation(n) for _ in range(8)]
r=np.random.default_rng(20261128); splits += [r.permutation(n) for _ in range(40)]
def opt(solver):
 e=[]
 for p in splits:
  i,j=p[:52],p[52:]; q=dict(s,train={k:v[i] if k in ('x','y','traj') else v for k,v in t.items()},test={'x':t['x'][j]}); a=solver(q); y=a['y_hat']; st=a['state_pred']; assert np.isfinite(y).all() and np.isfinite(st).all(); np.testing.assert_allclose(a['diagnostics_vec'],[y.mean(),y.std(),st[:,-1].mean(),st[:,-1].std()]); e.append(np.r_[np.mean(abs(y-t['y'][j])),np.mean(abs(st-t['traj'][j]),axis=0)])
 return np.array(e)
a,b=opt(base),opt(trial)
def metrics(e): return [e[0,0],e[0,1:].mean(),e[:9,0].mean(),e[:9,0].max(),e[9:,0].mean(),np.quantile(e[9:,0],.90),e[9:,0].max(),e[:,1:].max()]
for name,e in [('baseline',a),('challenger',b)]: print(name,'prescribed y/state, prior avg/max, stress avg/p90/max, worst state horizon',np.round(metrics(e),6))
from pathlib import Path
import shutil
promote=(b[0,0]<=.038 and b[:9,0].max()<a[:9,0].max() and b[9:,0].mean()<a[9:,0].mean() and np.quantile(b[9:,0],.9)<np.quantile(a[9:,0],.9) and b[9:,0].max()<=1.02*a[9:,0].max())
if promote: shutil.copyfile('/tmp/challenger13.py','/holdout_task/batch_solver.py')
print('selected', 'challenger' if promote else 'baseline')
# Reproduce the missing-cubic witness with the production inverse on inliers.
from batch_solver import _cal, _lin, _ls
u,v=[s['panel_a'][k] for k in ('u','v')]; w=np.roll(u,-1,axis=1); d=np.c_[_lin(u),u*w,u**3]; keep=np.argsort(np.linalg.norm(v-d@_ls(d,v),axis=1))[:-3]
en=[]
for j in range(4):
 f=np.c_[np.ones(len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3][keep]; zz=v[keep,j]; errs=[]
 for h in (4,5):
  d=f[:,:h]; lev=(np.linalg.qr(d)[0]**2).sum(1); errs.append(np.sqrt(np.mean(((d@_ls(d,zz)-zz)/(1-lev))**2)))
 en.append(errs)
print('panel A inlier leave-one-out RMSE, quadratic vs cubic',np.array(en).T); assert np.all(np.array(en)[:,1]<.001)
print('production calibration inverse max inlier input error',np.max(abs(_cal(s['panel_a'],v[keep])-u[keep])))
