import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; g=np.load('/tmp/feat06.npz'); z,w,Y=[g[k] for k in ('z','w','Y')]; dd=np.load('/tmp/d06.npz'); c,d=dd['c'],dd['d']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; one=lambda x:np.c_[np.ones(len(x)),x]; t=s['eval_times']; print('t,telemetry,summary:',t,s['telemetry_vec'],s['train']['summary']); print('SVD',np.linalg.svd(Y,compute_uv=False)); print('SVD centered',np.linalg.svd(Y-Y.mean(0),compute_uv=False)); print('traj SVD',np.linalg.svd(Y[:,1:],compute_uv=False)); print('linear y from time',abs(one(Y[:,1:])@ls(one(Y[:52,1:]),Y[:52,0])-Y[:,0]).mean());
_,sv,vt=np.linalg.svd(Y[:52,1:],full_matrices=False)
def run(blocks, nr):
 T=np.c_[Y[:,0],Y[:,1:]@vt[:nr].T]; M=T[:52].mean(0); S=T[:52].std(0); T=(T-M)/S
 F={}; add=lambda n,x:F.update({n:x});
 for k,h in blocks.items():
  for i in range(4):
   add(f'{k}{i}',h[:,i]);
   for j in range(i,4):add(f'{k}{i}*{k}{j}',h[:,i]*h[:,j]);
  for i,j in combinations(range(4),2):
   for sig in [1,-1]:
    x=h[:,i]+sig*h[:,j]; n=f'{k}{i}{sig:+}{k}{j}';
    for q in [.12,.22,.32,.5,1.]:add(f'rat{q}:{n}',x/(1+q*x*x));add(f'sq{q}:{n}',x*x/(1+q*x*x))
    for q in [1.,1.25,1.6,2.4]:add(f'sin{q}:{n}',np.sin(q*x));add(f'cos{q}:{n}',np.cos(q*x));add(f'tanh{q}:{n}',np.tanh(q*x))
 for k,l in combinations(blocks,2):
  for i in range(4):
   for j in range(4):add(f'{k}{i}*{l}{j}',blocks[k][:,i]*blocks[l][:,j]);
 A=np.column_stack(list(F.values())); A-=A[:52].mean(0); A/=A[:52].std(0).clip(.00001); B=A[:52]; Q=np.empty((52,0)); ii=[]; R=T[:52]; best=(99,None); names=list(F)
 for step in range(43):
  den=np.sqrt(np.maximum((B*B).sum(0)-((Q.T@B)**2).sum(0),.000001)); sc=np.linalg.norm(B.T@R,axis=1)/den; sc[ii]=-1; ii.append(sc.argmax()); H=B[:,ii]; Q=np.linalg.qr(H)[0]; C=ls(H,T[:52]); R=T[:52]-H@C; P=(A[:,ii]@C)*S+M; pred=np.c_[P[:,0],P[:,1:]@vt[:nr]]; er=abs(pred-Y); a,b=er[52:,0].mean(),er[52:,1:].mean();
  if max(a,b)<best[0]:best=(max(a,b),(step+1,a,b,er[:52].mean(),[names[i] for i in ii],C*S));
 return best
for r in [2,3,4]:
 m=run({'z':z,'c':c,'d':d},r); print('sparse',r,m[0],m[1][:-1]); print(np.round(m[1][-1][:12],3))
