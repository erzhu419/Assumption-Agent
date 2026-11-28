from explore import *
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

def cal(s,x,details=False):
 u,v=[s['panel_a'][k] for k in ('u','v')]; _,e=fit(one(np.c_[u,pair(u)]),v); g=e<1
 def basis(z):
  w=np.roll(z,-1,axis=1);return np.stack((np.ones_like(z),z,w,z*w,z**3),axis=2)
 b=np.stack([ls(basis(u)[g,j],v[g,j]) for j in range(4)])
 def f(z):return np.einsum('njk,jk->nj',basis(z),b)
 z0=(x-b[:,0])/b[:,1];z=least_squares(lambda Z:(f(Z.reshape(-1,4))-x).ravel(),z0.ravel(),jac_sparsity=np.kron(np.eye(len(x)),np.eye(4)+np.roll(np.eye(4),1,axis=1)),ftol=1e-12,gtol=1e-12).x.reshape(-1,4)
 if details:print('cal',b,'forward RMSE',np.mean((f(u)[g]-v[g])**2)**.5,'inverse max',np.max(abs(f(z)-x)))
 return z

def branch(p,z,key,details=False):
 u=p['u'];v=p[key]; best=[]
 for fun in [lambda z,a:z/(1+a*abs(z)),lambda z,a:1/(1+a*z*z),lambda z,a:np.cos(a*z),lambda z,a:np.tanh(a*z)]:
  for mode in [0,1]:
   def f(z,a):return one(np.c_[z,fun(z,a)]) if mode else one(fun(z,a))
   def err(a):
    F=f(u,a);return np.mean((F@ls(F,v)-v)**2)
   r=minimize_scalar(err,bounds=(.1,3),method='bounded');b=ls(f(u,r.x),v);best.append((r.fun,r.x,b,f,mode));
   if details:print(key,len(best),r.x,r.fun**.5)
 e,a,b,f,_=min(best,key=lambda x:x[0]);
 if details:print('best coefficients',b)
 return f(z,a)@b

def features(s,x,details=False):
 z=cal(s,x,details);b=s['panel_b']; L=one(z)@ls(one(b['u']),b['v']);C,D=(branch(s['panel_c'],z,k,details) for k in ('v1','v2'));return z,L,C,D

if __name__=='__main__':
 t=s['train'];z,L,C,D=features(s,t['x'],True);Y=np.c_[t['y'],t['traj']];ts=s['eval_times'];
 def h(p):return -np.expm1(-ts[:,None]*p)
 def err(p):H=h(p);return (H@ls(H,t['traj'].T)-t['traj'].T).ravel()
 pp=least_squares(err,[1.8,.4],bounds=(.01,10));print('time',pp.x,np.std(pp.fun));H=h(pp.x); a=ls(H,t['traj'].T).T;Y=np.c_[Y,a];
 cross=lambda A,B:(A[:,:,None]*B[:,None,:]).reshape(len(z),-1)
 F={'z':one(z),'L':one(L),'C':one(C),'D':one(D),'CD':one(np.c_[C,D]),'zCD':one(np.c_[z,C,D]),'zCLD':one(np.c_[z,C,L,D]),'dot':one(np.c_[C,D,(C*D).sum(1)]),'bilin':one(np.c_[C,D,cross(C,D)]),'zbilin':one(np.c_[z,C,D,cross(C,D)]),'zquad':one(np.c_[z,pair(z)]),'Cquad':one(np.c_[C,pair(C)]),'Dquad':one(np.c_[D,pair(D)]),'C+Dquad':one(np.c_[C,D,pair(C+D)]),'dotLD':one(np.c_[L,C,D,(L*D).sum(1)]),'dotLC':one(np.c_[L,C,D,(L*C).sum(1)]),'LC':one(np.c_[L,C,cross(L,C)]),'LD':one(np.c_[L,D,cross(L,D)])};
 for k,M in F.items():
  b=ls(M[:52],Y[:52]);E=Y-M@b;print(k,M.shape[1], 'MAE',np.round(abs(E[52:]).mean(0)[[0,1,8,9,10]],5),'fit',np.round(abs(E[:52]).mean(0)[[0,1,8,9,10]],5),'sv',np.round(np.linalg.svd(E[:52,:9],compute_uv=False)[:4],4));
  if k in ('dot','CD','bilin'): print('weights',np.round(b[:,[0,9,10]].T,4))
 np.savez('/tmp/features.npz',z=z,L=L,C=C,D=D,a=a,Y=Y,H=H)
