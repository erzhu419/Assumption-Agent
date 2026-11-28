exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
p=s['panel_c']; u=p['u']; v1=p['v1']; v2=p['v2']; w1=old.fit(old.AF(u),v1); c1=old.AF(z)@w1
print('C1 AF',w1);print('C2 quadratic',old.fit(poly(u,2),v2));print('C2 quad using v1',old.fit(poly(v1,2),v2));print('C2 resid using v1',np.std(poly(v1,2)@old.fit(poly(v1,2),v2)-v2))
c2=poly(z,2)@old.fit(poly(u,2),v2)
from scipy.linalg import solve
from scipy.spatial.distance import cdist
P={'z':z,'b':b,'c1':c1,'c2':c2,'z+b':np.c_[z,b],'z+c1':np.c_[z,c1],'z+c2':np.c_[z,c2],'b+c2':np.c_[b,c2],'z+b+c1':np.c_[z,b,c1],'z+b+c2':np.c_[z,b,c2],'z+b+c1+c2':np.c_[z,b,c1,c2]}
for key,A in P.items():
 print(key,'linear',assess(D(A)),'quadratic',assess(poly(A,2)) if A.shape[1]<=8 else '')
 if A.shape[1]==4: print(' cubic',assess(poly(A,3)))
 res=[]
 for trf in [lambda a:a,lambda a:np.arcsinh(3*a)]:
  for standard in [False,True]:
   for deg in [1,2,3]:
    for decay in [.05,.3,1.5]:
     for ridge in [1e-6,.002,.02]:
      scores=[]
      for i,j in sl:
       x=trf(A);x=(x-x[i].mean(0))/(x[i].std(0) if standard else .5);d=x.shape[1]; k=(1+x@x.T/d)**deg+np.exp(-decay*cdist(x,x,'sqeuclidean')/d)
       e=k[np.ix_(j,i)]@solve(k[np.ix_(i,i)]+ridge*np.eye(len(i)),T[i],assume_a='pos')-T[j];scores.append([abs(e[:,0]).mean(),abs(e[:,1:]@f.T).mean()])
      scores=np.array(scores);res.append((scores.mean(0),scores[0],scores.max(0),(trf.__code__.co_firstlineno,standard,deg,decay,ridge)))
 for h in [0,1]:
  for r in sorted(res,key=lambda r:r[0][h])[:2]: print(' head',h,r)
q=least_squares(lambda q:(-np.expm1(-s['eval_times'][:,None]*q)@old.fit(-np.expm1(-s['eval_times'][:,None]*q),t['traj'].T)-t['traj'].T).ravel(),[1.8,.42],bounds=([1,.15],[3,.85]));print('rates',q.x,'svd',np.linalg.svd(t['traj'],compute_uv=False)); print('corr',np.corrcoef(np.c_[z[:n],b[:n],c1[:n],c2[:n],T].T)[-3:])
