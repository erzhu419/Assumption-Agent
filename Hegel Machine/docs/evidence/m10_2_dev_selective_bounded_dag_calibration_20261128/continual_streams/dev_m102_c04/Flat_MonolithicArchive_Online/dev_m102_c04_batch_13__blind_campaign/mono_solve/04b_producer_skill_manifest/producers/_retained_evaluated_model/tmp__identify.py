import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]; n=52
np.set_printoptions(precision=4,suppress=True)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(*a):return np.column_stack((np.ones(len(a[0])),*a))
def pairs(x):return np.array([x[:,i]*x[:,j] for i,j in it.combinations(range(4),2)]).T
u,v=(s['panel_a'][k] for k in ('u','v')); e=np.max(abs(u-v),axis=1); m=e<2; u,v=u[m],v[m];print('a mask',sum(m),np.sort(e)); print('a linear',ls(aff(u),v),np.std(v-aff(u)@ls(aff(u),v),0))
E=np.broadcast_to(np.eye(4),(len(u),4,4)); zobs=np.r_[t['x'],s['test']['x']]; cal=[]
for typ,fn in [('tanh',np.tanh),('sin',np.sin),('rad',lambda x:x/(1+abs(x))),('atan',np.arctan)]:
 for roll in (1,2,3):
  def design(k):return np.concatenate((E,E*u[:,:,None],fn(k[0]*u)[:,:,None],np.sin(k[1]*np.roll(u,roll,axis=1))[:,:,None]),2).reshape(-1,10)
  def fun(k):a=design(k);return a@ls(a,v.ravel())-v.ravel()
  k=least_squares(fun,[1.8,2.4],bounds=([.15,.15],[5,5])).x; a=design(k); w=ls(a,v.ravel());cal.append((np.std(fun(k)),typ,roll,k,w,fn))
for q in sorted(cal,key=lambda q:q[0])[:5]:print('cal',q[:5]);
err,typ,roll,k,w,fn=min(cal,key=lambda q:q[0]); z=least_squares if False else (zobs-w[:4])/w[4:8]
for _ in range(100): z=(zobs-w[:4]-w[8]*fn(k[0]*z)-w[9]*np.sin(k[1]*np.roll(z,roll,axis=1)))/w[4:8]
mods={}; u=s['panel_c']['u']
for key in ('v1','v2'):
 v=s['panel_c'][key]; rr=[]
 for name,fn in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('atan',np.arctan),('rat',lambda x:x/(1+abs(x))),('rat2',lambda x:x*x/(1+x*x)),('log',lambda x:np.log1p(x*x)),('relu',lambda x:np.maximum(x,0)),('exp',lambda x:np.exp(-x*x)),('cubic',lambda x:x**3)]:
  for kind in range(4):
   def des(x,a):return aff(*([x] if kind%2 else []),*([pairs(x)] if kind>=2 else []),fn(a*x))
   def fun(a):A=des(u,a);return np.mean((v-A@ls(A,v))**2)
   a=minimize_scalar(fun,bounds=(.1,4),method='bounded').x; A=des(u,a); W=ls(A,v); res=v-A@W; Q=np.linalg.svd(A,full_matrices=False)[0]; h=np.sum(Q[:,:np.linalg.matrix_rank(A)]**2,1); score=np.sqrt(np.mean((res/(1-h[:,None]))**2));rr.append((score,np.std(res),name,kind,a,des(z,a)@W));
 for q in sorted(rr,key=lambda q:q[0])[:12]:print(key,q[:5]);
 mods[key]=min(rr,key=lambda q:q[0])[-1]
p,r=mods.values(); b=aff(z)@ls(aff(s['panel_b']['u']),s['panel_b']['v']);
for name,L in [('z',[z]),('p',[p]),('r',[r]),('b',[b]),('pr',[p,r]),('zpr',[z,p,r]),('bpr',[b,p,r])]:
 for order in (1,2,3):
  h=np.column_stack(L); cols=[h];
  for k in range(2,order+1):cols.append(np.array([np.prod(h[:,i],axis=1) for i in it.combinations_with_replacement(range(h.shape[1]),k)]).T)
  X=aff(*cols); pred=X[n:72]@ls(X[:n],Y[:n]); e=abs(pred-Y[n:]); print(name,order,X.shape[1],np.mean(e[:,0]),np.mean(e[:,1:]));
np.savez('/tmp/identified.npz',z=z,p=p,r=r,b=b,Y=Y)
