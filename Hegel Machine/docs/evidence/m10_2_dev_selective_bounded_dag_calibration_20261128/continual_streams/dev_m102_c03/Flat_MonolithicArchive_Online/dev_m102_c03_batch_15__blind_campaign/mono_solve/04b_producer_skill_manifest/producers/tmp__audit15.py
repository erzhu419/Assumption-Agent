import sys, pickle, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch,_aff,_calibrate,_branch,_design
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; train=s['train']; u=s['panel_c']['u']; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]
for j in [1,2]:
 v=s['panel_c'][f'v{j}'];
 def design(x,p):
  return np.c_[_aff(x),x/(1+abs(x*p))] if j==1 else np.c_[_aff(x),x*np.roll(x,-1,axis=1),np.tanh(x*p)]
 def res(p):
  d=design(u,p);return (d@ls(d,v)-v).ravel()
 p=least_squares(res,np.full(4,[.83,1.2][j-1]),bounds=(.1,4),gtol=1e-12).x; print('operator',j,p,'rmse',np.mean(res(p)**2)**.5, 'coef\n',np.round(ls(design(u,p),v),5))
def validate(a,b):
 r=solve_batch({**s,'train':{k:v[a] if k!='summary' else np.array([train['y'][a].mean(),train['y'][a].std(),train['traj'][a,-1].mean(),train['traj'][a,-1].std()]) for k,v in train.items()},'test':{'x':train['x'][b]}})
 ey=abs(r['y_hat']-train['y'][b]);et=abs(r['state_pred']-train['traj'][b]);return np.r_[ey.mean(),et.mean(),ey.max(),et.mean(0).max()]
ix=np.arange(72); print('52/20',validate(ix[:52],ix[52:])); rs=np.random.default_rng(15); print('repeated worst',np.max([validate(p[:52],p[52:]) for p in (rs.permutation(72) for _ in range(20))],axis=0)); z=_calibrate(s['panel_a'],train['x']);
for name,b in [('large-norm',np.argsort(np.linalg.norm(z,axis=1))[-20:]),*((f'coordinate-{j}-{sgn}',np.argsort(sgn*z[:,j])[-20:]) for j in range(4) for sgn in [-1,1])]:
 print(name,validate(np.setdiff1d(ix,b),b))
c=_branch(s['panel_c'],z,'v1','soft');d=_branch(s['panel_c'],z,'v2','cyclic');f=_design(z,c,d);print('readout scalar',np.round(ls(f[:52],train['y'][:52]),4))
