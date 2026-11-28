import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=f.run_node({'__input__':s}); tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times'];
def poly(x,p):
 terms=[i for d in range(p+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),d)];return np.stack([np.prod(x[:,i],axis=1) for i in terms],1),terms
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(a,b):return np.mean(abs(a-b))
u=s['panel_b']['u'];v=s['panel_b']['v'];a=f._affine(u); z=out['z']; zt=out['zt']; raw=np.linalg.solve(ls(a,v)[1:].T,(z-ls(a,v)[0]).T).T
from scipy.optimize import minimize_scalar
for pn,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 U=s[pn]['u']; V=s[pn][k]; cand=[]
 funs={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'atan':np.arctan,'soft':lambda x:x/np.sqrt(1+x*x),'exp':np.exp,'rational':lambda x:1/(1+x*x),'cube':lambda x:x**3}
 for nm,fn in funs.items():
  for extra in ('','pairs','roll'):
   def D(X,h):
    P={'':[],'pairs':[X[:,i]*X[:,j] for i in range(4) for j in range(i+1,4)],'roll':[X*np.roll(X,-1,axis=1)]}[extra];return np.column_stack((f._affine(X),fn(h*X),*P))
   def loss(h):
    A=D(U,h);return np.linalg.norm(A@ls(A,V)-V)
   grid=np.linspace(.15,5,80); i=np.argmin([loss(h) for h in grid]);h=minimize_scalar(loss,bounds=(max(.04,grid[i]-.1),grid[i]+.1),method='bounded',options={'xatol':1e-12}).x
   A=D(U,h); C=ls(A,V); e=A@C-V; lev=np.sum(np.linalg.qr(A)[0]**2,1);cv=np.mean(abs(e/np.maximum(1-lev,1e-9)[:,None])); cand.append((cv,loss(h),nm,extra,h,C,D(raw,h)))
 for cv,e,nm,ex,h,C,A in sorted(cand,key=lambda b:b[0])[:5]:
  print(pn,k,nm,ex,'freq',round(h,7),'press',round(cv,8),'res',round(e,8),'coef',C.round(4).tolist());
  if pn!='panel_b':continue
  X=A@C
  for d in range(1,4):
   B,names=poly(X,d);co=ls(B[:52],np.c_[y,T][:52]);E=abs((B@co-np.c_[y,T])[52:]);print('degree',d,'cv',E[:,0].mean(),E[:,1:].mean(),'y coeff',co[:,0].round(3) if d==2 else '')
