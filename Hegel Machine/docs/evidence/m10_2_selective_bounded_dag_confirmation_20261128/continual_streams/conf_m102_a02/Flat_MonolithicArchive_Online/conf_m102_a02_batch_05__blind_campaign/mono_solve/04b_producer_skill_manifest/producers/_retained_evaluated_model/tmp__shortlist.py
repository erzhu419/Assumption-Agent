from paths import *
import importlib.util
from scipy.optimize import minimize_scalar
P=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(P);P.loader.exec_module(m)
h=m._latents(S,T['x']); z=cal(T['x']); y=T['y']; tr=T['traj']; t=S['eval_times']; rng=np.random.default_rng(886); ps=[np.arange(72)]+[rng.permutation(72) for _ in range(25)]+[np.argsort(T['x'][:,j]*sgn) for j in range(4) for sgn in (-1,1)]; A=np.array(ps)[:,:52]; B=np.array(ps)[:,52:]; base=m._cross(h)
def stat(p):
 e=abs(p-y[B]).mean(1);return np.round([e[0],e[1:26].mean(),e[1:26].max(),e[26:].mean(),e[26:].max()],5)
def fit(F,pen):
 f,g=F[A],F[B]; ft=f.transpose(0,2,1);return (g@np.linalg.solve(ft@f+np.diag(pen),ft@y[A,:,None] if False else ft@y[A][...,None]))[:,:,0]
print('baseline',stat(fit(base,np.r_[np.zeros(5),np.full(6,.03)])))
for name, q in [('latent diagonal',h**2),('input diagonal',z**2),('response diagonal',(h @ np.linalg.inv(ls(af(z*(-.08972+z**2*(1+z**2*(-.46296+.09351*z**2)))),h)[1:]))**2),('projected state', (af(h)@ls(af(h),tr[:,-1]))[:,None]**2)][:3]:
 for l in (.2,.5,1.,2.):
  pred=fit(np.c_[base,q],np.r_[np.zeros(5),np.full(6,.03),np.full(q.shape[1],l)])
  for w in (.5,1.):
   b=fit(base,np.r_[np.zeros(5),np.full(6,.03)]); print(name,l,w,stat(w*pred+(1-w)*b))
# Joint diagonal correction. Leave each held-out observation out of the coefficient fit.
for lz,lh in [(1,2),(.6,6),(.8,8),(1,8),(1.2,6),(.8,20),(1.5,10),(1.2,20),(2,2)]:
 for lc in (.015,.03,.07):
  print('joint',lz,lh,lc,stat(fit(np.c_[base,z**2,h**2],np.r_[np.zeros(5),np.full(6,lc),np.full(4,lz),np.full(4,lh)])))
for l in [.4,.7,1,1.5]:
 for df in [None,.03,.06]:
  p=[];F=np.c_[base,z**2]; R=np.diag(np.r_[np.zeros(5),np.full(6,.03),np.full(4,l)])
  for a,b in zip(A,B):
   f=F[a]; Y=y[a]; w=np.ones(len(a))
   for _ in range(8 if df else 1):
    c=np.linalg.solve(f.T@(w[:,None]*f)+R,f.T@(w*Y));w=1/np.sqrt(1+((f@c-Y)/(df or 1))**2)
   p.append(F[b]@c)
  print('input diagonal robust',l,df,stat(np.array(p)))
