import pickle, numpy as np
from itertools import combinations_with_replacement, combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
def load(i):
 ns={};exec(open(f'/holdout_task/_frozen_nodes/node_0{i}.py').read(),ns);return ns['run_node']
a=load(0)({'__input__':s}); u=a['calibrated_x']; z=load(1)({'node_00':a})['x']; n=len(s['train']['y']); Y=np.column_stack([s['train']['y'],s['train']['traj']]); t=s['eval_times']
def poly(x,p):
 d=x.shape[1];return np.column_stack([np.prod(x[:,k],axis=1) for j in range(p+1) for k in combinations_with_replacement(range(d),j)])
def loo(A,Y):
 c,_,rank,_=np.linalg.lstsq(A,Y,rcond=1e-11); h=np.sum(np.linalg.svd(A,full_matrices=False)[0][:,:rank]**2,axis=1); e=A@c-Y; return np.mean(np.abs(e),axis=0),np.mean(np.abs(e/np.maximum(1e-9,1-h[:,None])),axis=0)
p=s['panel_c']; q=p['u']; P=s['panel_b']; B=np.linalg.lstsq(poly(P['u'],1),P['v'],rcond=None)[0]; w=poly(q,1)@B
print('B panel MAE',np.mean(abs(poly(P['u'],1)@B-P['v']))); print('v1 versus B',np.mean(abs(p['v1']-w),axis=0)); print('v1 canonical',np.linalg.lstsq(poly(q,1),p['v1'],rcond=None)[0]); print('B canonical',B)
for k in ['v1','v2']:
 for base in [q,w]:
  for kind,F in [('quad',poly(base,2)),('cubics',np.c_[poly(base,2),base**3]),('squash',np.c_[poly(base,2),np.tanh(2*base),np.sin(3*base)]),('trig',np.c_[poly(base,1),np.sin(2*base),np.sin(4*base),np.cos(3*base)])]:
   print(k,'q' if base is q else 'w',kind,loo(F,p[k])[1])
print('time right singular vectors',np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2]);print('time mean',Y[:,1:].mean(axis=0)); print('scalar correlation with final',np.corrcoef(Y[:,0],Y[:,-1])[0,1]);
from scipy.linalg import lstsq
I=np.arange(n); folds=[(np.setdiff1d(I,h),h) for h in [I[-20:],I[:20],I[20:40],I[40:60]]]
def cv(A,reg=0,YY=Y):
 out=[]
 for j,k in folds:
  D=A[j]; V=A[k]; a=D.mean(0); scale=np.maximum(D.std(0),1e-10); D=(D-a)/scale; V=(V-a)/scale
  U,S,G=np.linalg.svd(D,full_matrices=False); c=(G.T*(S/(S**2+max(reg,1e-12))))@(U.T@(YY[j]-YY[j].mean(0))); err=np.mean(abs(V@c+YY[j].mean(0)-YY[k]),axis=0);out.append([err[0],err[1:].mean()]);
 return np.array(out)
for k in [2,3,4,5,6]:
 for r in [0.0001,0.01,0.1,1.]:
  A=np.c_[poly(u,k)[:,1:],poly(z,k)[:,1:]][:n]; E=cv(A,r);print('poly ridge',k,r,E[0],E.max(0))
for coords in [u,z,np.c_[u,z]]:
 for name,g in [('sin',lambda x,f:np.sin(x*f)),('tanh',lambda x,f:np.tanh(x*f)),('rat',lambda x,f:x/(1+f*x*x))]:
  for f in [1,2,3,4,5]:
   v=g(coords,f); D=np.c_[poly(u,2),v,(v[:,:,None]*u[:,None,:]).reshape(len(u),-1)][:n]; E=cv(D,0.1);print('add+pair',coords.shape[1],coords is u,name,f,E[0],E.max(0))
