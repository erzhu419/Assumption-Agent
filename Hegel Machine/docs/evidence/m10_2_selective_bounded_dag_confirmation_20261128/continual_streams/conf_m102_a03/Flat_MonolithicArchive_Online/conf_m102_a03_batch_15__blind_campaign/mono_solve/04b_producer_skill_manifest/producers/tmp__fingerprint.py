import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
def affine(x):return np.c_[np.ones(len(x)),x]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
p=s['panel_a']; u=p['u'];v=p['v'];keep=np.linalg.norm(v-u,axis=1)<3;u=u[keep];v=v[keep];i=np.arange(4)
for shift in [1,2,3]:
 for kind in ['poly','fullpoly','ts','ss','tt','as','identity']:
  def design(a,p):
   b=np.roll(a,shift,axis=1); ns=len(a); A=np.tile(np.eye(4),(ns,1)); x=a.ravel();w=b.ravel(); f={'poly':[w,x*w,x**3], 'fullpoly':[w,x*w,x*x,w*w,x**3,w**3], 'ts':[np.tanh(p[0]*x),np.sin(p[1]*w)],'ss':[np.sin(p[0]*x),np.sin(p[1]*w)],'tt':[np.tanh(p[0]*x),np.tanh(p[1]*w)],'as':[np.arctan(p[0]*x),np.sin(p[1]*w)],'identity':[]}[kind];return np.column_stack([A,A*x[:,None]]+[A*t[:,None] for t in f])
  def err(p):
   a=design(u,p);return a@fit(a,v.ravel())-v.ravel()
  p=least_squares(err,[1.8,2.4],bounds=(.05,7)).x
  print('cal',shift,kind,np.linalg.norm(err(p))/np.sqrt(v.size),p)
for pk,vk in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pk]['u'];v=s[pk][vk]; res=[]
 funcs={'sin':(lambda x,p:np.sin(p[0]*x),1),'cos':(lambda x,p:np.cos(p[0]*x),1),'tanh':(lambda x,p:np.tanh(p[0]*x),1),'atan':(lambda x,p:np.arctan(p[0]*x),1),'rat':(lambda x,p:x/(1+p[0]*x*x),1),'threshold':(lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],2),'poly':(lambda x,p:np.c_[x*x,x**3],0),'atsin':(lambda x,p:np.c_[np.arctan(p[0]*x),np.sin(p[1]*x)],2),'sincos':(lambda x,p:np.c_[np.sin(p[0]*x),np.cos(p[1]*x)],2),'tansin':(lambda x,p:np.c_[np.tanh(p[0]*x),np.sin(p[1]*x)],2),'pow':(lambda x,p:np.sign(x)*abs(x)**p[0],1)}
 for k,(f,np_) in funcs.items():
  for linear in [False,True]:
   def err(p):
    a=affine(np.c_[u,f(u,p)] if linear else f(u,p)); return (a@fit(a,v)-v).ravel()
   starts=([[.3,1.5]] if k=='threshold' else [[j]*(np_) for j in [.8,1.8,2.8,4.]]) if np_ else [[]];opt=min([(np.linalg.norm(err(q)),q) for q in [least_squares(err,p,bounds=(.005,7)).x if np_ else [] for p in starts]],key=lambda r:r[0]);res.append((opt[0]/np.sqrt(v.size),k,linear,opt[1]))
 print(pk,vk,sorted(res,key=lambda r:r[0])[:10])
t=s['eval_times']; y=tr['traj'];
for typ in [0,1,2]:
 def basis(p):
  a=-np.expm1(-np.outer(t,p)) if typ==0 else np.exp(-np.outer(t,p));return np.c_[np.ones(len(t)),a] if typ==2 else a
 def err(p):
  a=basis(p);return (a@fit(a,y.T)-y.T).ravel()
 p=least_squares(err,[.4,1.8] if typ!=2 else [1.],bounds=(.001,15)).x
 print('time',typ,p,np.sqrt(np.mean(err(p)**2)))
