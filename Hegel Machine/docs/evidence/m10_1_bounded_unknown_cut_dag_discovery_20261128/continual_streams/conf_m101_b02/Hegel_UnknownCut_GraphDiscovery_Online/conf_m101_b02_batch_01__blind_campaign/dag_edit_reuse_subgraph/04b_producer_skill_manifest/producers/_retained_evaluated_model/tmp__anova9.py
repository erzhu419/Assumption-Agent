import sys,pickle,numpy as np
from scipy.linalg import solve
from itertools import combinations
sys.path.insert(0,'/holdout_task')
import node_00,node_01
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; g,_=node_01._features(s,node_00.run_node({'__input__':s})); Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y); g=[u[:n] for u in g]; z=g[0]; ids=np.arange(n); folds=[ids[-i:] for i in (18,20,24)]+list(np.array_split(np.random.default_rng(81).permutation(n),4))
def ker(x,tr,h):
 x=(x-x[tr].mean(0))/np.maximum(x[tr].std(0),1e-9); p=x@x.T/x.shape[1]; d=np.maximum(p.diagonal()[:,None]+p.diagonal()-2*p,0); return np.exp(-d/(2*h*h)),p
for source in ['z','direct','route','joint','jointsplit']:
 for h in [.35,.65,1.1,2,4]:
  K=[]
  for va in folds:
   tr=np.setdiff1d(ids,va); a=[ker(z[:,i:i+1],tr,h)[0] for i in range(4)]; S=sum(a)/4; I=sum(a[i]*a[j] for i,j in combinations(range(4),2))/6; T=sum(a[i]*a[j]*a[k] for i,j,k in combinations(range(4),3))/4
   groups={'z':[z],'direct':[z,g[1],g[2],g[4]],'route':[z,g[1],g[3],g[5]],'joint':g,'jointsplit':g}[source]
   B,P=ker(np.concatenate(groups,axis=1),tr,h)
   if source=='jointsplit': B=sum(ker(u,tr,h)[0] for u in groups)/len(groups)
   K.append((va,tr,1+P,B,S,I,T))
  ans=[]
  for w in [0,.03,.3,3,30]:
   for j in [.03,.3,3]:
    for ts in [0,.3,3]:
     for l in [1e-6,1e-4,.003,.03]:
      es=[]
      for va,tr,P,B,S,I,T in K:
       Q=P+B+w*S+j*I+ts*T; pred=Q[np.ix_(va,tr)]@solve(Q[np.ix_(tr,tr)]+l*np.eye(len(tr)),Y[tr],assume_a='pos'); E=abs(pred-Y[va]); es.append([E[:,0].mean(),E[:,1:].mean()]);
      ans.append((np.max(es),w,j,ts,l,np.round(np.array(es)[:3],4),np.round(np.max(es,0),4)))
  best=min(ans,key=lambda v:v[0]); print(source,h,best[:5],best[5].tolist(),best[6].tolist(),flush=True)
