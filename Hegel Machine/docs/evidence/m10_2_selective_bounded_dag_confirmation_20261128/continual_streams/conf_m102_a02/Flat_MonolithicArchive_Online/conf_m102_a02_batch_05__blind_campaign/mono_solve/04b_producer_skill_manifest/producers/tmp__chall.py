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
for name,add in [('h2',h*h),('h3',h**3),('z',z),('z2',z*z),('z3',z**3),('raw interactions',quad(z,True)[:,5:]),('local corrections',[z,z*z,z**3]),('mixed interactions',(h[:,:,None]*z[:,None,:]).reshape(72,16)),('other panel',operators()['C'](z))]:
 if isinstance(add,list):add=np.column_stack(add)
 for lam in (.03,.3,1.):
  print(name,lam,stat(fit(np.c_[base,add],np.r_[np.zeros(5),np.full(6,.03),np.full(add.shape[1],lam)])))
for space,g in [('latent',h),('raw',z)]:
 for num in ('aff','cross'):
  for lam in (.02,.12):
   pred=[]; pars=[]
   for a,b in zip(A,B):
    F=af(h) if num=='aff' else base; R=np.diag(np.sqrt(np.r_[np.zeros(F.shape[1]),np.full(F.shape[1],lam)])); Y=np.r_[y[a],np.zeros(len(R))];
    def design(v,p):
     w=g[v]@p[1:5]+p[5]; s=p[0]*w*w;return np.c_[F[v],F[v]*(-s/(1+s))[:,None]]
    def res(p):
     U=np.r_[design(a,p),R];return U@ls(U,Y)-Y
    starts=[np.r_[1,np.eye(4)[j],0] for j in range(4)]; pp=min([least_squares(res,p,bounds=([.02]+[-2]*4+[-.6],[4.]+[2]*4+[.6]),max_nfev=44) for p in starts],key=lambda o:o.cost).x; pars.append(pp);pred.append(design(b,pp)@ls(np.r_[design(a,pp),R],Y))
   print('rational',space,num,lam,stat(np.array(pred)),np.round(pars[0],3),flush=True)
for name,F in [('aff',af(h)),('z',np.c_[af(h),z]),('q',quad(h)),('cross',base)]:
 for rank in (1,2,8):
  ee=[]
  for a,b in zip(A,B):
   _,_,V=np.linalg.svd(tr[a],full_matrices=False);V=V[:rank]; C=ls(F[a],tr[a]@V.T);ee.append(abs(F[b]@C@V-tr[b]).mean())
  print('state',name,rank,np.round([ee[0],np.mean(ee[1:26]),max(ee[1:26]),max(ee[26:])],5))
