import pickle,numpy as np,glob,importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  spec=importlib.util.spec_from_file_location('m',f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); o=m.solve_batch(dict(s,train={k:v[:n] if k!='summary' else v for k,v in tr.items()},test={'x':tr['x'][n:]})); print(f[-18:],np.mean(abs(o['y_hat']-tr['y'][n:])),np.mean(abs(o['state_pred']-tr['traj'][n:])))
 except Exception as e: print(f,e)
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; q=np.roll(u,1,axis=1); r=np.roll(u,-1,axis=1); h=np.roll(u,2,axis=1); pairs=np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]).T
 for ky,v in p.items():
  if ky=='u':continue
  print(k,ky)
  if k=='panel_a':
   for typ in range(6):
    er=[]; co=[]
    for j in range(4):
     x=u[:,j]; y=v[:,j]; z=[r,q,h][typ%3][:,j]; A=aff(np.c_[x,z,x*z,x**3] if typ<3 else np.c_[x,np.tanh(1.8*x),np.sin(2.4*z)]); ix=np.arange(len(x))
     for _ in range(12):
      w=ls(A[ix],y[ix]); e=abs(A@w-y); ix=np.argsort(e)[:39]
     er.append(np.sqrt(np.mean(e[ix]**2))); co.append(w)
    print(typ,np.round(er,6),np.round(co,3))
   continue
  D={'aff':u,'pair':np.c_[u,pairs],'poly2':np.c_[u,u*u,pairs],'diag3':np.c_[u,u*u,u**3], 'c_archive':np.tanh(1.75*u)+.3*np.sin(2.1*q),'dead':np.sign(u)*np.maximum(abs(u)-.35,0)**1.5,'cos':np.cos(2.6*u),'sin':np.sin(2.15*u),'tanhsin':np.c_[np.tanh(1.75*u),np.sin(2.1*u)],'tanhsinprod':np.c_[np.tanh(1.75*u),np.sin(2.1*u),pairs],'rat':np.c_[u/(1+1.25*u*u),u*u,pairs],'archive_aug':np.c_[u,np.cos(2.6*u),np.sin(2.15*u),pairs]}
  for nm,a in D.items():
   a=aff(a); e=a@ls(a,v)-v; lev=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1); print(nm, np.sqrt(np.mean(e*e)).round(5),np.sqrt(np.mean((e/(1-lev[:,None]))**2)).round(5))
t=s['eval_times']; Y=tr['traj'].T
for fun in [lambda t,r:-np.expm1(-t*r),lambda t,r:np.c_[-np.expm1(-t[:,0]*r[0]), t[:,0]*np.exp(-t[:,0]*r[1])],lambda t,r:np.c_[-np.expm1(-t[:,0]*r[0]),(1-np.exp(-t[:,0]*r[1]))**2]]:
 def err(p):
  a=fun(t[:,None],p); return ((a@ls(a,Y)-Y).ravel())
 p=least_squares(err,[2,.3],bounds=([.001,.001],[12,12])).x; a=fun(t[:,None],p); print('time',p,np.sqrt(np.mean(err(p)**2)),np.round(ls(a,Y)[:,:12],3))
