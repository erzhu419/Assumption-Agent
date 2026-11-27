exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
for tag,X in [('joint',x),('frozen',xx)]:
 r=X@b[3:]; y=target-b[2]-r; N=len(y); lab=ch['lab_id']; folds=[lab==g for g in np.unique(lab)]+[abs(X[:,j])>=np.sort(abs(X[:,j]))[-2] for j in range(4)]; allk=np.ones(N,bool)
 Q=np.c_[X, hold, past, ch['strain_0'],past.mean(1),past.std(1),past.max(1)-past.min(1)]; Q[:,4:]=(Q[:,4:]-Q[:,4:].mean(0))/Q[:,4:].std(0)
 Qnames=['x0','x1','x2','x3','hold',*['p'+str(j) for j in range(6)],'strain0','past mean','past sd','past range']; print(tag,'ratio correlations', sorted(zip(Qnames, np.round(np.corrcoef(np.c_[Q,(y/r)].T)[-1,:-1],3)),key=lambda z:-abs(z[1]))[:8]);
 for grp in ([0,1,2,3],list(range(4,12)),[12,13,14],list(range(15))):
  Z=Q[:,grp]
  for mode in ('linear','diagonal square','diagonal cube','quadratic'):
   W=Z if mode=='linear' else Z**2 if mode=='diagonal square' else abs(Z)**3 if mode=='diagonal cube' else np.array([Z[:,i]*Z[:,j] for i,j in combinations_with_replacement(range(Z.shape[1]),2)]).T
   A=r[:,None]*np.c_[np.ones(N),W]
   for p in (.003,.03):
    mu=np.linalg.norm(A,axis=0); B=A/np.maximum(mu,1e-8); P=p*np.eye(B.shape[1]); P[0,0]=0
    def fit(k): return np.linalg.solve(B[k].T@B[k]+P,B[k].T@y[k]);
    e=[B[v]@fit(~v)-y[v] for v in folds]; print('ridge',grp[0],grp[-1],mode,p,np.round([np.sqrt(np.mean((B@fit(allk)-y)**2)), np.sqrt(np.mean(np.concatenate(e[:4])**2)),np.sqrt(np.mean(np.concatenate(e[4:])**2)),max(np.sqrt(np.mean(a*a)) for a in e)],5))
 rng=np.random.default_rng(13); starts=rng.normal(size=(12,5))*.25
 for typ in ('square','cube','bounded','shifted'):
  def predict(p):
   a=X@p[:4]; c=p[4]; u=a*a if typ=='square' else abs(a)**3 if typ=='cube' else a*a/(1+a*a) if typ=='bounded' else (a+c)**2; return r*(u+(0 if typ=='shifted' else c))
  def fit(k):
   def fun(p): return np.r_[ (predict(p)-y)[k], .04*p ]
   z=[least_squares(fun,p,max_nfev=250) for p in starts]; return min(z,key=lambda w:w.cost).x
  p=fit(allk); e=[(predict(fit(~v))-y)[v] for v in folds]; print('projection',typ,np.round(p,4),np.round([np.sqrt(np.mean((predict(p)-y)**2)), np.sqrt(np.mean(np.concatenate(e[:4])**2)),np.sqrt(np.mean(np.concatenate(e[4:])**2)),max(np.sqrt(np.mean(a*a)) for a in e)],5))
