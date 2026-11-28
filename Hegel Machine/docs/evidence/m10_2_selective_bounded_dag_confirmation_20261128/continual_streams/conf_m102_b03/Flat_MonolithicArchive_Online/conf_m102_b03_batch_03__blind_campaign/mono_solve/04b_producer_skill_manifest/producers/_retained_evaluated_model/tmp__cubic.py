import numpy as np,pickle,importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=52
one=lambda x:np.column_stack((np.ones(len(x)),x)); ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
A=s['panel_a']; u,v=A['u'],A['v']; U=one(u); rng=np.random.default_rng(45)
sol=[]
for _ in range(200):
 i=rng.choice(len(u),8,False); q=ls(U[i],v[i]); sol.append((np.median(np.linalg.norm(U@q-v,axis=1)),q))
q=min(sol,key=lambda p:p[0])[1]; r=np.linalg.norm(U@q-v,axis=1); good=r<max(.005,5*np.median(r)); q=ls(U[good],v[good]); print('full cal',q,'error',np.sort(np.linalg.norm(U@q-v,axis=1))); z=np.linalg.solve(q[1:].T,(tr['x']-q[0]).T).T
def calfit(u,v,good):
 c=[]
 for j in range(4):
  a=u[:,j]; b=u[:,(j+1)%4]; D=one(np.c_[a,b,a*b,a**3]); c.append(ls(D[good],v[good,j]))
 return np.array(c).T
cf=calfit(u,v,good)
print('cyclic cubic calibration',cf)
def inverse(x):
 z=np.linalg.solve(q[1:].T,(x-q[0]).T).T
 for _ in range(18):
  zz=np.roll(z,-1,axis=1)
  r=cf[0]+cf[1]*z+cf[2]*zz+cf[3]*z*zz+cf[4]*z**3-x
  j=np.zeros((len(z),4,4)); k=np.arange(4)
  j[:,k,k]=cf[1]+cf[3]*zz+3*cf[4]*z*z
  j[:,k,(k+1)%4]=cf[2]+cf[3]*z
  z-=np.linalg.solve(j,r[...,None])[...,0]
 return z
z=inverse(tr['x']); print('inverse bounds',z.min(0),z.max(0)); print('in-sample inverse error',np.max(abs(inverse(v[good])-u[good])))
B=s['panel_b']; qb=ls(one(B['u']),B['v']); bfn=lambda z:one(z)@qb
C=s['panel_c']; fs=[]
for k,f,guess in [('v1',lambda w:w/(1+abs(w)),.84),('v2',lambda w:w/(1+np.exp(-w)),1.45)]:
 def fit(k,f,g):
  u,v=C['u'],C[k]; D=lambda z,h:one(np.c_[z,f(h[0]*z)]); res=lambda h:(D(u,h)@ls(D(u,h),v)-v).ravel(); o=least_squares(res,[g]); print(k,o.x,np.sqrt(np.mean(o.fun**2))); coef=ls(D(u,o.x),v); return lambda z:D(z,o.x)@coef
 fs.append(fit(k,f,guess))

def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def score(F):
 F=one(F);return ' '.join(str(tuple(round(w,5) for w in (np.mean(abs(F[N:]@ls(F[:N],tr[k][:N])-tr[k][N:])),np.mean(abs(F[:N]@ls(F[:N],tr[k][:N])-tr[k][:N]))))) for k in ['y','traj'])
c,d=[f(z) for f in fs]; b=bfn(z)
print('panel c fits',*[ls(one(np.c_[C['u'],f(C['u'])]),C[k]).shape for k,f in zip(['v1','v2'],fs)])
print('candidate scores: y (holdout, fit), trajectory (holdout, fit)')
for label,F in dict(cd=np.c_[c,d],dot=np.c_[c,d,(c*d).sum(1)],prod=np.c_[c,d,c*d],cross=np.c_[c,d,cross(c,d)],z_cross=np.c_[z,c,d,cross(c,d)],c_prods=np.c_[c,d,cross(c,c)],d_prods=np.c_[c,d,cross(d,d)],c_d_sq=np.c_[c,d,c*c,d*d],z_cd_prod=np.c_[z,c,d,c*d],cd_dot_prod=np.c_[c,d,(c*d).sum(1),c*d]).items():print(label,score(F))
t=s['eval_times']; Y=tr['traj'][:N]; H=lambda p:-np.expm1(-t[:,None]*p); res=lambda p:(H(p)@ls(H(p),Y.T)-Y.T).ravel(); fit=least_squares(res,[1.8,.4],bounds=([.7,.02],[6.,1.2])); print('time fit',fit.x,np.sqrt(np.mean(fit.fun**2))); h=H(fit.x); W=ls(h,Y.T).T
for F in [np.c_[c,d,(c*d).sum(1)], np.c_[c,d,c*d], np.c_[c,d,cross(c,d)]]:
 X=one(F); Q=ls(X[:N],W); print('coef',Q.T,'proj state MAE',np.mean(abs(X[N:]@Q@h.T-tr['traj'][N:])))
np.savez('/tmp/cubicops.npz',z=z,c=c,d=d,b=b)
