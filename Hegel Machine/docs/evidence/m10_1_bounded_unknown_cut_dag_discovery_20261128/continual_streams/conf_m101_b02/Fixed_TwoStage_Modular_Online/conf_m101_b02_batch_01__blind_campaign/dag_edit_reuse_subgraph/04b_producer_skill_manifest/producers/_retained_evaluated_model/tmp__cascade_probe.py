import pickle,runpy,itertools as it
import numpy as np
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/node_00.py'); sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; tr=s['train']; n=len(tr['y']); off,A=f['_calibrate'](s['panel_a']); x=tr['x']; z=np.linalg.solve(A.T,(x-off).T).T
b,c=s['panel_b'],s['panel_c']; norms=lambda q:(q-q.mean(0))/np.maximum(q.std(0),1e-8)
def maps(bmode):
 pf=f['_panel_fit']; bt=(lambda u:(np.sin(2.815*u),)) if bmode==0 else (lambda u:(np.tanh(3.51*u),u/(1+(2.434*u)**2)))
 return [lambda q:pf(b['u'],b['v'],q,bt),lambda q:pf(c['u'],c['v1'],q,lambda u:(np.sin(1.5*u),u*abs(u))),lambda q:pf(c['u'],c['v2'],q,lambda u:(u*u,np.cos(2.5*u)))]
Y=np.c_[tr['y'],tr['traj']]; print('trajectory intercept and contrasts:',np.round(Y[:,1:].mean(0),4),np.round(Y[:,1:].std(0),4)); r=np.random.default_rng(38); folds=[np.arange(n-20,n),*np.array_split(r.permutation(n),5)]; trains=[np.setdiff1d(np.arange(n),v) for v in folds]
def score(g,fd,cd):
 e=[]
 for t,v in zip(trains,folds):
  ix=np.r_[t,v]; out=sink({'node_00':dict(g=g[ix],fd=fd[ix],cd=cd[ix],y=Y[t,0],traj=Y[t,1:])}); R=abs(Y[v]-np.c_[out['y_hat'],out['state_pred']]);e.append([R[:,0].mean(),R[:,1:].mean()]);
 return np.array(e)
def kernel(q,w):
 a,u=np.linalg.eigh(np.exp(-w*cdist(q,q,'sqeuclidean')/q.shape[1])); return u*np.sqrt(np.maximum(a,0))
results=[]
for mode in (0,1):
 fun=maps(mode); direct=[f(z) for f in fun]; F=direct[0]; C=np.c_[*direct[1:]]; base=np.c_[np.ones(n),z,F,C];
 def record(name,g):
  e=score(g,F,C); results.append((np.max(e),max(e[0]),name,e[0],e.max(0)))
 for zm,z0 in [('inverse',z),('raw',x),('forward',x@A+off)]:
  D=[p(z0) for p in fun]; L=np.c_[np.ones(n),z0,*D]; record((mode,zm,'direct'),L)
  for depth in (1,2):
   for ids in it.product(range(3), repeat=depth+1):
    if len(set(ids))<2:continue
    h=z0
    for i in ids:h=fun[i](h)
    for g in [h,np.c_[D,h].reshape(n,-1) if False else np.c_[*D,h]]:
     record((mode,zm,ids,g.shape[1],0),np.c_[L,norms(g)])
     if zm=='inverse':
      for w in (.12,.35,.7):record((mode,zm,ids,g.shape[1],w),np.c_[L,kernel(norms(g),w)])
  if zm=='inverse':
   H=np.c_[*[fun[j](D[i]) for i,j in it.product(range(3),repeat=2) if i!=j]]
   for w in (.12,.35,.7,1.4):
    for a in (.25,.5,1,2):record((mode,'joint',w,a),np.c_[L,kernel(np.c_[norms(np.c_[*D]),a*norms(H)],w)])
   for basis in (np.c_[z,*D],np.c_[*D],np.c_[z,*D,H]):
    for w in (.08,.16,.3,.5,.8,1.2):record((mode,'rbf',basis.shape[1],w),np.c_[L,kernel(norms(basis),w)])
for idx in (0,1):
 print('rank',idx)
 for entry in sorted(results,key=lambda x:x[idx])[:12]:print(entry)
