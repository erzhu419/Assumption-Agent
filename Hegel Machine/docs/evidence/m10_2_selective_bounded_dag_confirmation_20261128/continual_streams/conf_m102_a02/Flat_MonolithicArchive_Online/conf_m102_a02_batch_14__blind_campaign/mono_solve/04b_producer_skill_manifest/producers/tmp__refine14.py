from graph14 import *
from scipy.optimize import differential_evolution
from scipy.linalg import orthogonal_procrustes
B=S['panel_b'];u,v=B['u'],B['v']; tr=S['train'];z=m._inverse(tr['x'],calibrate(S));y=tr['y'];T=tr['traj'];n=52
H={'genrat':(lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[(.05,6),(.5,6),(.1,4)]),'gengauss':(lambda x,p:x*np.exp(-p[0]*abs(x)**p[1]),[(.05,8),(.5,6)]),'gensin':(lambda x,p:np.sin(p[0]*x)*np.exp(-p[1]*x*x),[(.05,6),(0,5)]),'gentanh':(lambda x,p:np.tanh(p[0]*x)*np.exp(-p[1]*x*x),[(.05,8),(0,5)]),'sat':(lambda x,p:x/(1+p[0]*abs(x)**p[1])+p[2]*x,[(.05,6),(.5,6),(-1,1)]),'poly':(lambda x,p:x**3+p[0]*x**5+p[1]*x**7,[(-10,10),(-10,10)]),'sin3':(lambda x,p:np.sin(p[0]*x**3),[(.02,6)]),'tanh3':(lambda x,p:np.tanh(p[0]*x**3),[(.02,6)]),'atan3':(lambda x,p:np.arctan(p[0]*x**3),[(.02,6)]),'log':(lambda x,p:x*np.log1p(p[0]*abs(x)**p[1]),[(.01,6),(.5,6)])}
models=[]
for k,(f,bounds) in H.items():
 for lin in [0,1]:
  def d(x,p):return np.c_[aff(x) if lin else np.ones(len(x)),f(x,p)]
  def obj(p):
   a=d(u,p);return np.mean((a@ls(a,v)-v)**2)
  o=differential_evolution(obj,bounds,seed=14,maxiter=100,tol=1e-7);w=ls(d(u,o.x),v);a=d(z,o.x)@w; h=f(z,o.x);models.append((o.fun**.5,k,lin,o.x,a,h));print('op',k,lin,o.x,o.fun**.5,np.linalg.svd(w[-4:],compute_uv=False),flush=True)
models.sort(key=lambda a:a[0]);np.savez('/tmp/ops14.npz',z=z,**{f'b{i}':q[-2] for i,q in enumerate(models)},**{f'h{i}':q[-1] for i,q in enumerate(models)}); print('order',[(q[1],q[2]) for q in models]);
for q in models[:9]:
 e,k,lin,p,b,h=q
 for typ,a in [('full',feats(np.c_[z,b])),('pre',feats(np.c_[z,h])),('sparse',np.c_[feats(z),feats(b)[:,1:],z*b,z*np.roll(b,-1,axis=1),b*np.roll(z,-1,axis=1)]),('sep',np.c_[feats(z),feats(b)[:,1:]]),('presep',np.c_[feats(z),feats(h)[:,1:],z*h])]:
  w=ls(a[:n],np.c_[y,T][:n]); err=abs(a@w-np.c_[y,T]);print(k,typ,np.round([err[:n,0].mean(),err[n:,0].mean(),err[:n,1:].mean(),err[n:,1:].mean()],5))
