import runpy,pickle,numpy as np
from scipy.spatial.distance import cdist
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=[tr[k] for k in ['x','y','traj']]; h,hs=m['_designs'](s,x); z,b,c1,c2,b1,b2,c,d=m['_maps'](s,x); n=len(x); rng=np.random.default_rng(1)
splits=[np.arange(52,72)]+[rng.choice(n,20,replace=False) for _ in range(30)]+[np.argsort(q)[-20:] for q in [np.linalg.norm(z,axis=1),*z.T,*(-z.T)]]
def split_stats(e):
 e=np.asarray(e); return np.round([e[0],e[1:31].mean(),e[1:31].max(),e[31:].mean(),e[31:].max()],6)
def ridge(H,ix,ts,alpha):
 q=H[ix]; u=q.mean(0); w=q.std(0); w[w<1e-9]=1.; q=(q-u)/w; yt=y[ix]; yc=yt-yt.mean(); return yt.mean()+((H[ts]-u)/w)@np.linalg.solve(q.T@q+alpha*np.eye(q.shape[1]),q.T@yc)
for reg in [.0001,.001,.003,.01,.03,.1,.3,1.,3.]:
 P=[]
 for ts in splits:
  ix=np.setdiff1d(np.arange(n),ts); p=np.array([ridge(q[:,1:],ix,ts,reg) for q in hs]); P.append([*np.mean(abs(p-y[ts]),axis=1),np.mean(abs(p.mean(0)-y[ts]))])
 print('ridge',reg,*[split_stats(e) for e in np.array(P).T],sep='  ')
for rank in [1,2,3,4,5]:
 E=[]
 for ts in splits:
  ix=np.setdiff1d(np.arange(n),ts); beta=m['_fit'](h[ix],T[ix]); _,_,v=np.linalg.svd(h[ix]@beta,full_matrices=False); vv=v[:rank]; pred=h[ts]@beta@vv.T@vv; E.append(np.mean(abs(pred-T[ts])))
 print('traj_rank',rank,split_stats(E))
for scale in [.5,1.,1.5,2.5,4.]:
 W=[z,c1,c2]; D=[cdist(q/q.std(0),q/q.std(0),'sqeuclidean') for q in W]; K=sum((1+np.sqrt(5*d)/scale+5*d/(3*scale**2))*np.exp(-np.sqrt(5*d)/scale) for d in D)/len(D)
 for reg in [.001,.01,.05,.2]:
  E=[]
  for ts in splits:
   ix=np.setdiff1d(np.arange(n),ts); a=y[ix].mean(); p=a+K[np.ix_(ts,ix)]@np.linalg.solve(K[np.ix_(ix,ix)]+reg*np.eye(len(ix)),y[ix]-a); q=np.mean([ridge(u[:,1:],ix,ts,.03) for u in hs],axis=0); E.append([np.mean(abs(p-y[ts])),np.mean(abs(.25*p+.75*q-y[ts]))]);
  print('GP',scale,reg,*[split_stats(e) for e in np.array(E).T]);
