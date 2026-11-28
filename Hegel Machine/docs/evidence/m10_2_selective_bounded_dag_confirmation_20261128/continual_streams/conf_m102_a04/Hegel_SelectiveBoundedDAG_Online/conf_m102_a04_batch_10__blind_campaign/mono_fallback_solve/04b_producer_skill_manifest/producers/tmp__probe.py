import numpy as np,pickle,sys
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_a04_batch_06_solver import _coords,_panels
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; T=tr['traj']; y=tr['y']; t=s['eval_times']; x=tr['x']; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=160)
def one(X): return np.c_[np.ones(len(X)),X]
def ls(X,Y): return np.linalg.lstsq(X,Y,rcond=1e-9)[0]
p=s['panel_a']; u,v=p['u'],p['v']; k=np.linalg.norm(v,axis=1)<4;u,v=u[k],v[k];print('A usable',len(u)); adj=v-.32*np.roll(u,-1,axis=1)*(u-1)-.16*u**3
for j in range(4):
 q=u[:,j]; w=v[:,j]; r=np.roll(u,-1,axis=1)[:,j]; F=one(np.c_[q,r,r*q,q**3]); b=ls(F,w);print('A archived residual',np.std(one(q)@ls(one(q),adj[:,j])-adj[:,j]),'fitted',b,np.std(F@b-w));
 Fs=one(np.column_stack([np.prod(u[:,I],axis=1) for d in range(1,4) for I in combinations_with_replacement(range(4),d)]));
p=s['panel_c']; u=p['u']; quad=np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]); cyc=u*np.roll(u,-1,axis=1)
for key in ['v1','v2']:
 V=p[key]; print('C',key)
 def ev(F):
  F=one(F); E=V-F@ls(F,V);h=np.sum(np.linalg.qr(F)[0]**2,1);return np.std(E),np.mean(abs(E/np.maximum(.001,1-h[:,None])))
 for label,F in [('lin',u),('quad',np.c_[u,quad]),('cyc',np.c_[u,cyc]),('cube',np.c_[u,u**3]),('diagquad',np.c_[u,u*u]),('cos2.4',np.cos(2.4*u)),('cos2.5',np.cos(2.5*u)),('lincos',np.c_[u,np.cos(2.4*u)])]: print(label,ev(F))
 for label,fun in [('cos',lambda q:np.cos(q*u)),('sin',lambda q:np.sin(q*u)),('tanh',lambda q:np.tanh(q*u)),('softsign',lambda q:u/(1+q*abs(u))),('rational',lambda q:u*u/(1+q*u*u))]:
  for base in [False,True]:
   f=lambda q:ev(np.c_[u,fun(q)] if base else fun(q))[0]; grid=np.linspace(.12,6,100); i=np.argmin([f(q) for q in grid]); b=minimize_scalar(f,bounds=(grid[max(i-1,0)],grid[min(i+1,99)]),method='bounded');print(label,base,b.x,b.fun)
from scipy.optimize import least_squares
f=lambda q:-np.expm1(-t[:,None]*q)
g=lambda q:((f(q)@ls(f(q),T[:n].T)-T[:n].T)).ravel()
q=least_squares(g,[.22,2.15],bounds=(.005,12)).x;print('time',q,np.std(g(q)),np.std(g([.22,2.15])));print('loadings',ls(f(q),T.T).T[:10]);
