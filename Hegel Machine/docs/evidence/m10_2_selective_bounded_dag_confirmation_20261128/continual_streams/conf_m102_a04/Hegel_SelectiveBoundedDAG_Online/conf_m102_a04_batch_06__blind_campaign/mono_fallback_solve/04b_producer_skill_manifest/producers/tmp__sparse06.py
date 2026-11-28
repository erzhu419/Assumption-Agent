import numpy as np,pickle
from itertools import combinations_with_replacement as comb, combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(suppress=True,precision=4,linewidth=160)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def one(x):return np.c_[np.ones(len(x)),x]
ix=list(comb(range(4),2))
def P(x):return np.column_stack([one(x)]+[x[:,i]*x[:,j] for i,j in ix])
def D(x,q):return np.c_[one(x),x*x,x*x/(1+q*x*x)]
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.linalg.norm(v,axis=1)<6
F=lambda x:np.c_[P(x),x**3]; a=ls(F(u[keep]),v[keep]); print('A',a)
p=s['panel_b']; b=ls(one(p['u']),p['v']); p=s['panel_c']; u=p['u']; c=ls(P(u),p['v1']); print('C1',c)
q=least_squares(lambda q:(D(u,q)@ls(D(u,q),p['v2'])-p['v2']).ravel(),[.25],gtol=1e-13).x; d=ls(D(u,q),p['v2']); print('C2',q,d)
tr=s['train']; x=tr['x']; z=np.array([least_squares(lambda zz:(F(zz[None])@a-xx).ravel(),xx,gtol=1e-12).x for xx in x]); w=one(z)@b
Y=np.c_[tr['y'],tr['traj']]; _,_,vt=np.linalg.svd(tr['traj'][:52],full_matrices=False); target=np.c_[tr['y'],tr['traj']@vt[:3].T]
np.savez('/tmp/feat06.npz',z=z,w=w,a=a,b=b,c=c,d=d,q=q,Y=Y,target=target)
def candidates(blocks,extra):
 n=len(z); feats={}
 for k,h in blocks.items():
  for j,col in enumerate(h.T): feats[f'{k}{j}']=col
 source=list(feats.items());
 for (i,g),(j,h) in comb(source,2):
  feats[f'{i}*{j}']=g*h
 for i,g in source:
  for tag,h in [('sin',np.sin(g)),('sin125',np.sin(1.25*g)),('tanh',np.tanh(g)),('tanh125',np.tanh(1.25*g)),('sqrat',g*g/(1+.22*g*g)),('rat',g/(1+.22*g*g)),('cube',g**3),('abs',abs(g))]: feats[f'{tag}({i})']=h
 if extra:
  for k,h in blocks.items():
   for i,j in combinations(range(h.shape[1]),2):
    for sign in (1,-1):
     g=h[:,i]+sign*h[:,j]; name=f'{k}{i}{sign:+}{k}{j}'
     feats['sin'+name]=np.sin(1.25*g); feats['tanh'+name]=np.tanh(1.25*g); feats['rat'+name]=g/(1+.22*g*g); feats['sqrat'+name]=g*g/(1+.22*g*g)
 return np.column_stack(list(feats.values())),list(feats)
def omp(A,names,yt):
 train=np.arange(52); val=np.arange(52,72); A=A-A[train].mean(0); A/=np.maximum(A[train].std(0),1e-7); B=A[train]; mean=yt[train].mean(0); T=yt[train]-mean; S=np.maximum(T.std(0),.01); T=T/S; R=T.copy(); selected=[]; Q=np.empty((len(B),0)); best=(99,None)
 for step in range(35):
  scores=np.linalg.norm(B.T@R,axis=1)/np.maximum(np.sqrt(np.maximum(np.sum(B*B,0)-np.sum((Q.T@B)**2,0),0)),.001); scores[selected]=-1
  j=np.argmax(scores); selected.append(j); Q=np.linalg.qr(B[:,selected])[0]; coef=ls(B[:,selected],T); R=T-B[:,selected]@coef; pred=(A[val][:,selected]@coef)*S+mean; err=abs(pred-yt[val]); metric=err.mean();
  if metric<best[0]:best=(metric,(step+1,err.mean(0),[names[i] for i in selected],np.mean(abs(R)*S,axis=0)))
 return best
for hname,h in [('z',z),('w',w)]:
 c1=P(h)@c; c2=D(h,q)@d
 for bn,blocks in [('plain',{'z':z,'w':w,'c':c1,'d':c2}),('cd',{'c':c1,'d':c2}),('with linear after',{'z':z,'w':w,'c':c1,'d':c2,'e':one(c1)@b,'f':one(c2)@b})]:
  A,names=candidates(blocks,True)
  for tag,t in [('all',target),('y',target[:,:1]),('traj',Y[:,1:])]:print(hname,bn,tag,omp(A,names,t),flush=True)
