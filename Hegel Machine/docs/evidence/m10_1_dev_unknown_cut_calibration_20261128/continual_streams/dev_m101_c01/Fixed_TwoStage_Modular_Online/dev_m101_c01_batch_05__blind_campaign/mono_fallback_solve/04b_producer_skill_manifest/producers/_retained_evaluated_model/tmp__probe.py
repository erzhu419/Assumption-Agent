import numpy as np,pickle
from itertools import combinations_with_replacement as cw
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x,y,z=(s['train'][k] for k in ('x','y','traj')); t=s['eval_times']; L=lambda X,Y:np.linalg.lstsq(X,Y,rcond=None)[0]
def phi(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in cw(range(4),k)])
def affine(p,key):
 u,v=s[p]['u'],s[p][key]; X=phi(u,1); ix=np.arange(len(u))
 for _ in range(8 if p=='panel_a' else 1):
  a=L(X[ix],v[ix]); ix=np.argsort(np.linalg.norm(v-X@a,axis=1))[:36]
 if p=='panel_a':
  ix=np.linalg.norm(v-X@a,axis=1)<.01; a=L(X[ix],v[ix])
 return a
A,B=affine('panel_a','v'),affine('panel_b','v'); A[1:]=np.diag(A[1:].diagonal()); print('A/B',A,B)
def timebasis(name,p):
 g=-np.expm1(-p[0]*t); h=-np.expm1(-p[-1]*t)
 if name=='exp': return np.c_[g,h]
 if name=='lin': return np.c_[g,t]
 if name=='sin': return np.c_[g,np.sin(p[-1]*t)]
 if name=='sat': return np.c_[g,t/(1+p[-1]*t)]
 if name=='tsat': return np.c_[g,t*g]
 if name=='dexp':return np.c_[g,t*np.exp(-p[-1]*t)]
 if name=='expsin':return np.c_[g,np.exp(-p[1]*t)*np.sin(p[-1]*t)]
 if name=='growth':return np.c_[g,np.expm1(p[-1]*t)]
 if name=='tanh':return np.c_[g,np.tanh(p[-1]*t)]
for name in ['exp','lin','sin','sat','tsat','dexp','expsin','growth','tanh']:
 def resid(p):
  F=timebasis(name,p);return (F@L(F,z.T)-z.T).ravel()
 n=3 if name=='expsin' else 2
 opt=min([least_squares(resid,p,bounds=(.00001,6),max_nfev=180) for p in ([a,b] if n==2 else [a,c,b] for a in [.5,1.5,3] for b in [.1,1,3] for c in ([.05,.5] if n==3 else [0]))],key=lambda q:np.linalg.norm(q.fun));print('time',name,opt.x,np.std(opt.fun))
q=phi(x,2); bz=L(q,np.c_[y,z]); u,d,v=np.linalg.svd(z,full_matrices=False); Y=np.c_[y,u[:,:2]*d[:2]]
for name,xx in [('raw',x),('A',phi(x,1)@A),('Ainv',(x-A[0])/A[1:].diagonal()),('B',phi(x,1)@B),('Binv',(x-B[0])@np.linalg.inv(B[1:])),('AinvB',phi((x-A[0])/A[1:].diagonal(),1)@B)]:
 print('coords',name)
 C=L(phi(xx,2),Y); print('quad',np.round(C[5:].T,3))
for key in ['v1','v2']:
 u,v=s['panel_c']['u'],s['panel_c'][key]; print(key,'quad',np.round(L(phi(u,2),v)[5:].T,3))
