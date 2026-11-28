import numpy as np,pickle,runpy
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c03_batch_10_solver.py'); ls=old['_ls']; bias=old['_bias'];
a=s['panel_a']; u,v=a['u'],a['v']; j=np.arange(4); w=u[:,(j+2)%4]; keep=np.linalg.norm(v-u,axis=1)<3
for k in range(5):
 def af(x,p):
  w=np.roll(x,2,axis=1); g=np.roll(x,-1,axis=1); t=[x, np.tanh(p[0]*w), x**3];
  if k==1:t=[x,np.tanh(p[0]*w),np.tanh(p[1]*x)]
  if k==2:t=[x,np.tanh(p[0]*w),np.tanh(p[1]*x),x*g,g]
  if k==3:t += [x*g,g]
  if k==4:t=[x,np.sin(p[0]*w),x**3,x*g,g]
  return np.stack([np.ones_like(x)]+t,axis=2)
 def pars(x,p):
  f=af(x,p); return np.array([ls(f[keep,i],v[keep,i]) for i in j])
 def fun(p):return (np.einsum('njk,jk->nj',af(u,p),pars(u,p))-v)[keep].ravel()
 p=least_squares(fun,[1.5,1.3],bounds=(.05,6)).x; print('A',k,p,np.sqrt(np.mean(fun(p)**2)),np.round(pars(u,p),4))
 if k==2:
  b=pars(u,p); from scipy.optimize import root
  Z=np.array([root(lambda z: np.einsum('jk,jk->j',af(z[None],p)[0],b)-x,x).x for x in tr['x']]); print('coord range',Z.min(),Z.max()); np.save('/tmp/Z.npy',Z)
c=s['panel_c']; u,v=c['u'],c['v2']; print('B',ls(bias(s['panel_b']['u']),s['panel_b']['v'])); print('C2 lin',ls(bias(u),v));
D={'u2':u**2,'u3':u**3,'u4':u**4,'u5':u**5,'u6':u**6,'norm':u*np.sum(u*u,1,keepdims=True),'sumnorm':np.sum(u*u,1,keepdims=True),'u3/':u**3/(1+u*u),'abs':abs(u)}
for lag in (1,2,3):
 w=np.roll(u,lag,1); D['prod'+str(lag)]=u*w; D['dir'+str(lag)]=u*w*w; D['sqsum'+str(lag)]=(u+w)**2; D['cubsum'+str(lag)]=(u+w)**3
for k,d in D.items():
 f=bias(np.c_[u,d]); b=ls(f,v); e=f@b-v; lev=np.sum(f*np.linalg.pinv(f).T,axis=1); print('C2',k,np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-lev[:,None]))**2)),np.round(b[5:],2).tolist())
