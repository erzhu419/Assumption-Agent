import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement
D=dict(np.load('/tmp/graph_data.npz')); Y=D.pop('Y'); z=D['z']; n=52
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; print('time',t)
f=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-8)[0]
u=s['panel_b']['u'];print('b co',f(np.c_[np.ones(len(u)),u,u*np.roll(u,-1,axis=1)],s['panel_b']['v']).round(5))
np.set_printoptions(precision=4,suppress=True)
T=Y[:,1:]; print('y stats',np.quantile(Y[:,0],[0,.1,.5,.9,1])); print('y from traj',np.mean(abs(Y[n:,0]-np.c_[np.ones(len(T)),T][n:]@f(np.c_[np.ones(len(T)),T][:n],Y[:n,0]))))
def run(F):
 a=np.column_stack([np.ones(len(z)),F]); b=a@f(a[:n],Y[:n]); e=abs(b-Y);return np.r_[e[:n,0].mean(),e[n:,0].mean(),e[:n,1:].mean(),e[n:,1:].mean()]
def rate(p):return -np.expm1(-t[:,None]*p)
def res(p):
 a=rate(p);return (a@f(a,T[:n].T)-T[:n].T).ravel()
for p in [[.4,1.8,4],[.4,1.8,.15],[.4,1.8,12]]:
 q=least_squares(res,p,bounds=(.005,30),max_nfev=400); print('rates',q.x,np.linalg.norm(q.fun))
A=np.c_[z,D['c'],D['d']]; C=D['c']; E=D['d']; pairs=list(combinations_with_replacement(range(A.shape[1]),2)); P=np.column_stack([A[:,i]*A[:,j] for i,j in pairs]); F=np.c_[A,P]; names=[f'{k}{i}' for k in ['z','c','d'] for i in range(4)]; N=names+[names[i]+':'+names[j] for i,j in pairs];
print('main',run(A)); print('main+z*c/z*d',run(np.c_[A,P[:,[k for k,(i,j) in enumerate(pairs) if i<4]]]))
print('diag groups');
for k,l in [('z','z'),('z','c'),('z','d'),('c','c'),('c','d'),('d','d')]: print(k,l,run(np.c_[A,D[k]*D[l]]))
from scipy.linalg import svd
U,S,V=svd(T[:n],full_matrices=False); target=np.c_[Y[:,0],T@V[:3].T];
for j in range(4):
 y=target[:,j]; chosen=[]
 for it in range(14):
  base=np.c_[np.ones(len(z)),F[:,chosen]]; Q=np.linalg.qr(base[:n])[0]; r=y[:n]-Q@(Q.T@y[:n]); G=F[:n]-Q@(Q.T@F[:n]); score=abs(r@G)/np.maximum(np.linalg.norm(G,axis=0),1e-9);score[chosen]=-1; k=np.argmax(score);chosen.append(k)
  B=np.c_[np.ones(len(z)),F[:,chosen]]; e=abs(B@f(B[:n],y[:n])-y)
  print(j,it+1,N[k],round(e[:n].mean(),4),round(e[n:].mean(),4))
