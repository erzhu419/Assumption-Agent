import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times'];
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x: np.c_[np.ones(len(x)),x]
def poly(x,n): return np.column_stack([np.prod(x[:,i],axis=1) for d in range(n+1) for i in combinations_with_replacement(range(x.shape[1]),d)])
def score(F,V):
 P=F@ls(F[:52],V[:52]); return np.round([np.mean(abs(P[:52]-V[:52])),np.mean(abs(P[52:]-V[52:]))],5)
def base(r):return -np.expm1(-t[:,None]*np.asarray(r))
r=least_squares(lambda r:(base(r)@ls(base(r),T[:52].T)-T[:52].T).ravel(),[.5,1.8],bounds=(.01,10)).x; A=ls(base(r),T.T).T
print('rates',r,'time err',np.mean(abs(A@base(r).T-T))); print('y vs amplitude',score(aff(A),y),score(poly(A,2),y))
u,v=[s['panel_a'][k] for k in ('u','v')]; f=aff(u); keep=np.ones(len(u),bool)
for i in range(10):
 w=ls(f[keep],v[keep]); e=np.linalg.norm(f@w-v,axis=1); keep=e<max(.15,3*np.median(e));
print('a robust',keep.sum(),w.round(4),np.sort(e).round(4)); print('u/v clean rows',np.c_[u[keep],v[keep]][:5].round(5)); print('x std vs A',x.std(0),A.std(0)); z=(x-w[0])@np.linalg.inv(w[1:]);
for name,q in [('raw',x),('affine',z)]:
 for d in [1,2,3,4]: print(name,d,score(poly(q,d),y),score(poly(q,d),T))
 for h in [.7,1.3,2.15,3.]: print(name,h,score(np.c_[aff(q),np.sin(h*q),np.tanh(h*q)],y),score(np.c_[aff(q),np.sin(h*q),np.tanh(h*q)],T))
