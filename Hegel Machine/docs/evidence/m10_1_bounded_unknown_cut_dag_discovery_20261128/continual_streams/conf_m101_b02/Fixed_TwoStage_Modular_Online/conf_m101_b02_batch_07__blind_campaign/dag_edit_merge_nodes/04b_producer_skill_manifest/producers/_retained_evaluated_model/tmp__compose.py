import sys, pickle, itertools as it, numpy as np
sys.path.insert(0,'/holdout_task')
from node_00 import _calibrate,_cross,_aff,_ls,_kernel
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; aw,B,C,H,O=_calibrate(s); n=len(x)
folds=[np.arange(n-20,n),*np.array_split(np.arange(n),4)]; R=[]
def scores(name,D):
 e=[]; regs=[1e-7,.0001,.003,.03,.3,2]
 for v in folds:
  i=np.setdiff1d(np.arange(n),v); X=D[i]; mu=X.mean(0); sd=np.maximum(X.std(0),.02); X=(X-mu)/sd; Z=(D[v]-mu)/sd; u,g,h=np.linalg.svd(X,full_matrices=False); A=(Z@h.T)[:,:,None]*(u.T@(Y[i]-Y[i].mean(0)))[None,:,:]; P=np.einsum('vko,kr->rvo',A,g[:,None]/(g[:,None]**2+np.array(regs))); E=abs(P+Y[i].mean(0)-Y[v]); e.append(np.c_[E[:,:,0].mean(1),E[:,:,1:].mean((1,2))])
 e=np.asarray(e)
 for k,r in enumerate(regs): R.append((name,r,e[:,k]))

def inter(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def pw(z,d):return np.column_stack([np.prod(z[:,j],axis=1) for j in it.combinations_with_replacement(range(4),d)])
def pair(z,js):return np.column_stack([z[:,i]*z[:,j] for i,j in js])
for ci,z in enumerate([x,np.linalg.solve(aw[1:].T,(x-aw[0]).T).T,x@aw[1:]+aw[0]]):
 b=B(z); c=C(z); h=H(z); o=O(z); root=np.sqrt(1+.916*z*z)-1
 for fi,g in enumerate([z,b,o,h,c,z+b,z-b,z+o,z-o,z+c,z+h,z-h,b+h,b+c]):
  F=np.c_[z,g] if fi else z
  Q=np.c_[F,_cross(z),np.sqrt(1+.916*z*z)-1]
  for nm,D in {'plain':F,'compose1':np.c_[Q,B(g),C(g),H(g)],'compose2':np.c_[Q,O(g),_cross(g),np.sqrt(1+.916*g*g)-1],'samecross':np.c_[Q,_cross(g),z*g,g*g],'xcross':np.c_[Q,inter(z,g)],'paircompose':np.c_[F,C(g),H(g)],'full':np.c_[Q,b,c,h,inter(z,g),C(g),H(g),B(g)]}.items():scores((ci,fi,nm),D)
 for bi,add in enumerate([np.c_[z,z*z],np.c_[z,z*z,z**3],np.c_[z,z*z,z**3,z**4],np.c_[z,o,root],np.c_[z,o,root,z**3],np.c_[z,b,h],np.c_[z,o,root,z*np.exp(-z*z),np.sin(2.8*z)],np.c_[z,np.tanh(1.4*z),np.cos(1.3*z),np.sqrt(1+.916*z*z)-1]]):
  for fi,g in enumerate([z,o,b,h,z+o,z-o]):
   ps=list(it.combinations(range(4),2)); tri=np.column_stack([np.prod(z[:,j],axis=1) for j in it.combinations(range(4),3)]); p=pair(z,ps); j=pair(g,ps)
   for nm,D in {'pair':np.c_[add,p], 'cubic':np.c_[add,p,tri,pw(z,3)],'triple':np.c_[add,p,tri], 'warp':np.c_[add,p,j], 'smallwarp':np.c_[add,_cross(z),_cross(g)],'mixedwarp':np.c_[add,p,inter(z,g)],'qwarp':np.c_[add,p,pw(g,2)],'squaredpairs':np.c_[add,p,p*p],'nested':np.c_[add,p,H(g),C(g)]}.items():scores((ci,bi,fi,nm),D)
for j in [0,1]:
 for mode in ['worst','public']:
  rr=sorted(R,key=lambda r:np.max(r[2][:,j]) if mode=='worst' else r[2][0,j]); print(j,mode,*[(a,b,c.round(5).tolist()) for a,b,c in rr[:7]],sep='\n')
