import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task');import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']['traj']; y=s['train']['y']; q=n._inverse_a(s['train']['x'],n._fit_panel_a(s['panel_a'])); p=s['panel_c']; u=p['u']; B=lambda a:np.column_stack((np.ones(len(a)),a));
def err(x,target=t):
 r=x[52:]@np.linalg.lstsq(x[:52],target[:52],rcond=None)[0]-target[52:];return np.round(np.mean(abs(r),axis=0),5)
print('A-inverse ranges',q.min(0),q.max(0));print('traj raw calibrated',err(B(q)));print('traj singular values',np.linalg.svd(t,compute_uv=False));
for name,fn in [('cos',np.cos),('sin',np.sin),('tanh',np.tanh),('square',lambda a:a*a),('cube',lambda a:a**3)]:
 for quad in [False,True]:
  def F(a,f):return np.column_stack((B(a),a*a,fn(f*a))) if quad else np.column_stack((B(a),fn(f*a)))
  for label in ['v1','v2']:
   v=p[label]; opt=minimize_scalar(lambda f:np.mean((F(u,f)@np.linalg.lstsq(F(u,f),v,rcond=None)[0]-v)**2),bounds=(.2,4),method='bounded'); f=opt.x; x=F(u,f); c=np.linalg.lstsq(x,v,rcond=None)[0]; print(name,quad,label,round(f,4),round(opt.fun**.5,7),'traj via C',err(B(F(q,f)@c)).mean(),'traj flexible',err(F(q,f)).mean());
   if opt.fun**.5<.004:print('coeff',np.round(c,4))
for f in (0.5,1.,1.5,1.7,2.,2.5,3.):
 for fn in (np.cos,np.sin,np.tanh):
  print('basis',f,fn.__name__,err(np.column_stack((B(q),fn(f*q)))).mean())
for name,fs in [('poly2',[q[:,i]*q[:,j] for i in range(4) for j in range(i,4)]),('squares/cubes',[*q.T**2,*q.T**3]),('channel2 cos2',[*np.cos(2*q).T]),('cos plus mix',[*np.cos(2*q).T,*q.T**2,*q.T**3])]:print(name,err(np.column_stack((B(q),*fs))))
