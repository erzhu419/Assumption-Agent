import runpy,pickle,numpy as np
m=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c03_batch_10_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=m['_ls']; bias=m['_bias']; tr=s['train']; z=m['_coordinates'](s,tr['x']); print('z range',z.min(),z.max());
a=s['panel_a']; u,v=a['u'],a['v'];
for kind in range(6):
 f=[np.column_stack((np.ones(len(u)),u[:,j],u[:,(j+1)%4],u[:,j]*u[:,(j+1)%4],u[:,j]**3)) for j in range(4)] if kind==0 else [bias(u)]*4 if kind==1 else [bias(np.c_[u,u**2,u**3,np.prod(u[:,[j,(j+1)%4]],axis=1)]) for j in range(4)] if kind==2 else [bias(u[:,j])] if False else [bias(u[:,j:j+1]) for j in range(4)]
 keep=np.arange(len(u));
 for _ in range(6):
  par=[ls(t[keep],v[keep,j]) for j,t in enumerate(f)]; r=np.linalg.norm(np.column_stack([t@par[j] for j,t in enumerate(f)])-v,axis=1); keep=np.argsort(r)[:39]
 print('a',kind,np.quantile(r,[.2,.5,.75,.85,.9,1]),np.round(par,3));
 if kind==3:break
from scipy.optimize import least_squares
for k in ('panel_b','panel_c'):
 p=s[k]; u=p['u'];
 for key,v in p.items():
  if key=='u':continue
  for name,fn in [('lin',lambda u,p:u),('tanh',lambda u,p:np.tanh(p[0]*u)),('cos',lambda u,p:np.cos(p[0]*u)),('sin',lambda u,p:np.sin(p[0]*u)),('gauss',lambda u,p:np.exp(-p[0]*u*u)),('lin+gauss',lambda u,p:np.c_[u,np.exp(-p[0]*u*u)]),('cyc',lambda u,p:np.tanh(p[0]*u)+p[2]*np.roll(np.sin(p[1]*u),1,axis=1)),('quad',lambda u,p:np.c_[u,u**2]),('cubic',lambda u,p:np.c_[u,u**2,u**3])]:
   def fun(p):
    f=bias(fn(u,p));return (f@ls(f,v)-v).ravel()
   p=least_squares(fun,[1.75,2.1,.3] if name=='cyc' else [1.7],bounds=([.05,.05,-2],[7,7,2]) if name=='cyc' else (.02,8)).x
   print(k,key,name,np.round(p,4),np.sqrt(np.mean(fun(p)**2)))
for kind,zz in [('raw',tr['x']),('inv',z)]:
 w=m['_panels'](s,zz); f=m['_readout_features'](w); y=np.c_[tr['y'],tr['traj']]; print(kind,'CV',np.abs(f[52:]@ls(f[:52],y[:52])-y[52:]).mean(0),'train',np.abs(f@ls(f,y)-y).mean(0))
