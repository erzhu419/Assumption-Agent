import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})
z,q,y,T=a['z_tr'],a['z_te'],a['y'],a['traj']; u,v=(a['panel_b'][k] for k in ('u','v')); t=a['eval_times']
from scipy.optimize import least_squares
from itertools import combinations_with_replacement

def fit(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def basis(x,f): return np.column_stack((np.ones(len(x)),f))
def form(k,x,kind):
 s=x*k[0]
 fs={'sin':np.sin,'tanh':np.tanh,'rat':lambda s:s/(1+s*s),'atan':np.arctan}
 return fs[kind](s)+k[1]*x

def panel(kind,shared):
 def des(k,x): return basis(x,form(k,x,kind) if shared else np.column_stack((x,form([k[0],0],x,kind))))
 def resid(k):
  U=des(k,u); return (U@fit(U,v)-v).ravel()
 opts=[least_squares(resid,[k,-.4] if shared else [k],max_nfev=200) for k in (.7,1.6,2.8,4.)]; o=min(opts,key=lambda o:np.linalg.norm(o.fun))
 X=des(o.x,u); W=fit(X,v)
 print('panel',kind,shared,np.round(o.x,6),np.sqrt(np.mean(o.fun**2)), 'LOO',np.mean(abs(o.fun.reshape(v.shape)/(1-np.diag(X@np.linalg.pinv(X)))[:,None])))
 return o.x,np.sqrt(np.mean(o.fun**2)),des,W
opts=[panel(k,s) for k in ('sin','tanh','rat','atan') for s in (True,False)]
k,_,B,W=min(opts,key=lambda o:o[1]); G=B(k,z)@W; print('operator W',W)
yt=np.column_stack((y,T)); l=52
for name,F in [('B',basis(z,G)),('z+B',basis(z,np.c_[z,G]))]:
 R=yt-F@fit(F[:l],yt[:l]); print(name,'base MAE',np.mean(abs(R[l:]),axis=0),'fit y RMS',np.std(R[:l,0])); print('y residual largest', np.c_[z[:l],G[:l],y[:l],R[:l,0]][np.argsort(abs(R[:l,0]))[-5:]])

def products(x):return np.column_stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)])
P={
 'Bquad':products(G),'zquad':products(z),'Bsin':np.sin(2*G),'Btanh':np.tanh(2*G),'Bcube':G**3,
 'Bsq':G**2,'zsq':z**2,'zzB':np.einsum('ni,nj->nij',z,G).reshape(len(z),-1),'zBdiag':z*G,
 'BsinSum':np.sin(G.sum(axis=1))[:,None],'BpairSum':np.column_stack([np.sin(G[:,i]+sgn*G[:,j]) for i in range(4) for j in range(i) for sgn in (-1,1)]),
 'ztriple':np.column_stack([np.prod(z[:,[i for i in range(4) if i!=j]],axis=1) for j in range(4)]),
 'Btriple':np.column_stack([np.prod(G[:,[i for i in range(4) if i!=j]],axis=1) for j in range(4)])}
for name,X in [('B',G),('z+B',np.c_[z,G])]:
 base=basis(z,X); R=y[:l]-base[:l]@fit(base[:l],y[:l]); res=[]
 for nm,F in P.items():
  for nm2,J in [('',np.empty((len(z),0)))]+list(P.items()):
   if nm2 and nm2<=nm: continue
   H=np.c_[base,F,J]; H1=H[:l]; w=fit(H1,y[:l]); e=y[:l]-H1@w; h=np.diag(H1@np.linalg.pinv(H1));
   res.append((np.mean(abs(e/np.maximum(1-h,1e-7))),np.mean(abs(H[l:]@w-y[l:])),np.std(e),nm,nm2))
 print(name,'ranked inner LOOCV',sorted(res)[:12]);
 for nm,F in P.items():
  U=F[:l]-base[:l]@fit(base[:l],F[:l]); co=np.abs(U.T@R)/np.maximum(np.linalg.norm(U,axis=0),1e-15)/np.linalg.norm(R)
  print(nm,int(np.argmax(co)),round(max(co),3),end='; ')
 print()
