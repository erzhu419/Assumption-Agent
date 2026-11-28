import pickle
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import svd
np.set_printoptions(precision=4,suppress=True,linewidth=150)
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']
def load(path):
 ns={}; exec(compile(open(P+path).read(),path,'exec'),ns); return ns['run_node']
a=load('_frozen_nodes/node_00.py')({'__input__':s}); b=load('_frozen_nodes/node_01.py')({'node_00':a})
n=len(s['train']['x']); A=a['calibrated_x'][:n]; B=b['x'][:n]
u=s['panel_c']['u']; v=s['panel_c']['v1']; w=s['panel_c']['v2']
i,j=np.triu_indices(4,1)
def quad(x): return np.c_[np.ones(len(x)),x*x,x[:,i]*x[:,j]]
q=np.linalg.lstsq(quad(u),v,rcond=None)[0]; C=quad(B)@q
print('panel C1 coeffs',q,sep='\n'); print('C1 fit rmse',np.mean((quad(u)@q-v)**2)**.5)
def vp(k,x=u):return np.c_[np.ones(len(x)),x,np.cos(x*k)]
def proj(k):
 d=vp(k); return (d@np.linalg.lstsq(d,w,rcond=None)[0]-w).ravel()
k=least_squares(proj,[1.5],bounds=(0.2,5)).x; W=vp(k,B)@np.linalg.lstsq(vp(k),w,rcond=None)[0]
print('panel C2 cosine frequency, rmse, coeffs',k,np.mean(proj(k)**2)**.5,np.linalg.lstsq(vp(k),w,rcond=None)[0],sep='\n')
for name,x in [('A',A),('B',B),('C1',C),('C2',W)]: print(name,'range',np.min(x,0),np.max(x,0))
y=s['train']['y']; t=s['train']['traj']; ts=s['eval_times']; U,S,V=svd(t,full_matrices=False); print('time profiles',V[:3],sep='\n')
F=np.c_[y,t]; print('first rows A,B,C1,C2,y,traj final',np.c_[A,B,C,W,y,t[:,-1]][:5],sep='\n')
folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(9).permutation(n),5))
def score(name,d):
 e=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); beta=np.linalg.lstsq(d[tr],F[tr],rcond=1e-9)[0]; r=abs(F[te]-d[te]@beta); e.append([r[:,0].mean(),r[:,1:].mean()])
 print(name,d.shape[1],np.asarray(e)[0],np.mean(e[1:],axis=0),np.max(e,axis=0))
def h(*args):return np.column_stack([np.ones(n)]+list(args))
def pair(x): return x[:,i]*x[:,j]
def cross(x,z): return (x[:,:,None]*z[:,None,:]).reshape(n,-1)
for name,blocks in [('A',[A]),('B',[B]),('C',[C]),('W',[W]),('AB',[A,B]),('AC',[A,C]),('BC',[B,C]),('CW',[C,W]),('ACW',[A,C,W]),('ABCW',[A,B,C,W]),('Aquad C',[A,A*A,pair(A),C]),('Aquad CW',[A,A*A,pair(A),C,W]),('Aquad BC',[A,A*A,pair(A),B,C]),('Bquad A',[A,B,B*B,pair(B)]),('Cquad A',[A,C,C*C,pair(C)]),('C trig A',[A,C,np.tanh(C),np.sin(C),np.cos(C)]),('A times C',[A,C,cross(A,C)]),('A times W',[A,C,W,cross(A,W)]),('Aquad C diag',[A,A*A,pair(A),C,A*C]),('Aquad CW diag',[A,A*A,pair(A),C,W,A*C,A*W]),('A cubic C',[A,A*A,pair(A),C,A**3]),('A trig C',[A,C,np.tanh(A),np.sin(A),np.cos(A)])]:score(name,h(*blocks))
for name,blocks in [('AC',[A,C]),('ACW',[A,C,W]),('Aquad CW',[A,A*A,pair(A),C,W])]:
 d=h(*blocks); beta=np.linalg.lstsq(d,F,rcond=None)[0]; print(name,'coefs y,first,last',beta[:,[0,1,-1]],sep='\n')
