from operators import *
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; inv,B,C,D=operators(s); z=inv(s['train']['x']); c=C(z); d=D(z); T=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; n=52
q=lambda g:-np.expm1(-t[:,None]*g); g=least_squares(lambda g:(q(g)@fit(q(g),T[:n].T)-T[:n].T).ravel(),[.4,1.8]).x; a=fit(q(g),T.T).T; Y=np.c_[a,y,T]; U=np.c_[c,d]; nam=[f'{k}{i}' for k in 'cd' for i in range(4)]; pairs=[(i,j) for i in range(8) for j in range(i,8)]; names=[nam[i]+'*'+nam[j] for i,j in pairs]; F=np.stack([U[:,i]*U[:,j] for i,j in pairs],1); H=one(U); R=Y-H@fit(H[:n],Y[:n]); G=F-H@fit(H[:n],F[:n]); sig=np.linalg.norm(G[:n],axis=0); W=G/sig
print('quad coefficients amp0,amp1,y:'); V=fit(W[:n],R[:n]);
for j in range(3):print(j,[(names[i],round(V[i,j]/sig[i],4)) for i in np.argsort(-abs(V[:,j]))[:16]])
for alpha in [0,1e-5,.0001,.001,.003,.01,.03,.1]:
 V=np.linalg.solve(W[:n].T@W[:n]+alpha*np.eye(W.shape[1]),W[:n].T@R[:n]); e=R-W@V; print('ridge',alpha,np.mean(abs(e[n:,:3]),0),np.mean(abs(e[n:,3:])))
for j in range(3):
 active=[()]; seen=set(active)
 for k in range(1,7):
  scores=[]
  for curr in active:
   for i in range(W.shape[1]):
    if i in curr:continue
    ids=tuple(sorted((*curr,i)))
    if ids in seen:continue
    seen.add(ids); X=W[:,ids]; co=fit(X[:n],R[:n,j]);e=R[:,j]-X@co; scores.append((np.linalg.norm(e[:n]),ids,np.mean(abs(e[n:])),co/sig[list(ids)]))
  scores.sort(key=lambda v:v[0]); active=[v[1] for v in scores[:50]]
  print('beam',j,k,[(round(sc,5),[names[i] for i in ids],round(er,5),co.round(4)) for sc,ids,er,co in scores[:2]])
print('temporal rates',g)
