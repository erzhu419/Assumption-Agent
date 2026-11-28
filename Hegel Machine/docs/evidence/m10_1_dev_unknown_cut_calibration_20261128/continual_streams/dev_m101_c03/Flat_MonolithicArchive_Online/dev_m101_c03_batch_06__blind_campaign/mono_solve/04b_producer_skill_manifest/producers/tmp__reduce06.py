exec(open('/tmp/validate06.py').read().split('for model in')[0])
from itertools import combinations_with_replacement
z,a,b=_geometry(s,tr['x']); dot=np.sum(a*b,axis=1); cross=a[:,0]*b[:,1]
pairs={(0,1),(0,3),(1,2),(2,3)}
base=np.c_[np.ones(n),b,dot,cross,z]
K=np.stack([np.prod(z[:,k],axis=1) for k in sorted(pairs)],1)
C=np.stack([np.prod(z[:,k],axis=1) for k in combinations_with_replacement(range(4),3) if any(set(p)<=set(k) for p in pairs)],1)
for name,P,cut in [('full',H,21),('pairquad',np.c_[base,K,H[:,21:]],15),('paircubic',np.c_[H[:,:21],C],21),('paired',np.c_[base,K,C],15),('noz',np.c_[base[:,:7],K,C],11)]:
 for shared in (False,True):
  for time in ('project','joint'):
   results=[]
   for j in splits.values():
    i=np.setdiff1d(np.arange(n),j); coef=_ls(P[i],y[i]); yh=P[j]@coef
    Q=np.c_[P[:,:cut],P[:,cut:]@coef[cut:]] if shared else P
    L=_ls(Q[i],S[i]); F=_time_fit(t,S[i],Q[i]@L)
    if time=='project': sh=Q[j]@L@np.linalg.pinv(F.T)@F.T
    else:
     from scipy.optimize import least_squares
     # The endpoint fixes the shared cubic direction. Profile the linear spatial
     # coefficients and temporal rates against the denoised training trajectories.
     B=lambda k:-np.expm1(-t[:,None]*k)
     p=least_squares(lambda k:(L@np.linalg.pinv(B(k).T)@B(k).T-L).ravel(),[.4,1.8,5.3],bounds=([.08,.8,3],[.8,3,10])).x
     F=B(p);sh=Q[j]@L@np.linalg.pinv(F.T)@F.T
    e=abs(sh-S[j]);results.append([abs(yh-y[j]).mean(),e.mean(),e.mean(0).max(),e[:,-1].mean()])
   m=np.array(results);print(name,shared,time,'56/16',m[0].round(6),'mean',m.mean(0).round(6),'worst',m.max(0).round(6))
