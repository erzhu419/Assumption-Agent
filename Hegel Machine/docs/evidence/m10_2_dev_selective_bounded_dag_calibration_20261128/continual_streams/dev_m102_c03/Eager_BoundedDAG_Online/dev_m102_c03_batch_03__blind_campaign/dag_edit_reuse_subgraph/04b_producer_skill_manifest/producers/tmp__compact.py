import pickle,runpy,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cr
m=runpy.run_path('/holdout_task/node_01.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; z=m['_warp'](m['_latent'](x,s['panel_a']),s['panel_c']); T=tr['traj']; y=tr['y']; t=s['eval_times']; fit=m['_fit']; poly=m['_poly']; q=lambda r:-np.expm1(-t[:,None]*r); n=len(x); rng=np.random.default_rng(1); splits=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(rng.permutation(n),4)]
D=[]
for i,j in splits:
 def e(r):
  Q=q(r);return (Q@fit(Q,T[i].T)-T[i].T).ravel()
 Q=q(least_squares(e,[.408,1.815],bounds=([.1,1],[.9,4])).x);D.append((i,j,Q,np.c_[y[i],fit(Q,T[i].T).T]))
R=[]
for a in (.5,1.,1.5,2.,2.2,2.5,3.,4.):
 w=np.tanh(a*z)
 for mode,F in [('ct',np.c_[poly(z,3),poly(w,2)]),('mix',np.c_[poly(z,2),poly(w,2),z*w]),('wt',np.c_[poly(z,1),poly(w,3)]),('w',poly(w,3)),('mix3',np.c_[poly(z,3),poly(w,2),z*w]),('w1',np.c_[poly(z,3),w])]:
  for alpha in (.0003,.003,.015,.06,.2):
   for prune in (0,12,20,28):
    if prune>=F.shape[1]:continue
    S=[]
    for i,j,Q,Y in D:
     M=F[i].mean(0); sd=np.maximum(F[i].std(0),1e-8); A=(F[i]-M)/sd; B=(F[j]-M)/sd; mu=Y.mean(0); G=A.T@A+alpha*np.eye(A.shape[1]); P=A.T@(Y-mu); C=np.linalg.solve(G,P)
     if prune:
      V=np.linalg.inv(G); pred=[]
      for c in range(3):
       ix=np.argsort(abs(C[:,c])/np.sqrt(V.diagonal()))[-prune:];pred.append(B[:,ix]@np.linalg.solve(G[np.ix_(ix,ix)],P[ix,c])+mu[c]);
      Z=np.column_stack(pred)
     else: Z=B@C+mu
     S.append([np.mean(abs(Z[:,0]-y[j])),np.mean(abs(Z[:,1:]@Q.T-T[j]))])
    S=np.array(S);R.append((S.max(),S[0].max(),a,mode,alpha,prune,S[0].round(5),S.mean(0).round(5)))
for p in sorted(R,key=lambda r:r[0])[:13]:print(p)
print('best qualifying');print(*sorted([r for r in R if r[1]<.038],key=lambda r:r[0])[:12],sep='\n')
