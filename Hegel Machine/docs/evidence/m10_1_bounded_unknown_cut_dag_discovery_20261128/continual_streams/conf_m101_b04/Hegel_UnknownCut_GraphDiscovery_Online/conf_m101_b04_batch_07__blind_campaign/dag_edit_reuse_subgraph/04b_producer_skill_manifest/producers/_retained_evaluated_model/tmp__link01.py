exec(open('/tmp/stress01.py').read().split('for te in folds:')[0])
from scipy.interpolate import BSpline
from scipy.special import expit
folds=[np.arange(52,72),*folds]
N=len(y); signs=np.array([-1.,1.,1.,1.]); g=b@signs
pairs=[(i,j) for i in range(4) for j in range(i+1,4)]; Q=np.column_stack([b[:,i]*b[:,j] for i,j in pairs]); sq=b*b
print('linear univariate link',np.corrcoef(g,y)[0,1]); print('scalar extrema ordering',np.c_[g,y][np.argsort(g)][::4])
def evaluate(tr,te,case):
 kind,quad,amp,reg=case
 base=np.c_[np.ones(N),b]; beta=np.linalg.lstsq(base[tr],y[tr],rcond=None)[0]; h=(base@beta-np.mean(y[tr]))/np.std(y[tr]); L=[]
 if kind in (1,3): L.extend([np.tanh(h),np.exp(-h*h),h*h/(1+h*h)])
 if kind in (2,3): L.extend([np.sin(h),np.cos(h),np.sin(2*h),np.cos(2*h)])
 if quad in (1,2): L.extend(Q.T if quad==2 else [Q[:,4]])
 if quad==2: L.extend(sq.T)
 if amp==1: L.extend([x[:,0]**2,x[:,0]**4])
 if amp==2: L.extend([x[:,0]**2/(1+x[:,0]**2),x[:,0]**2])
 G=b/np.std(b[tr],0)/2; K=np.exp(-.5*cdist(G,G,'sqeuclidean')) if reg>0 else np.zeros((N,N))
 if L:
  D=np.array(L).T; D=D/np.maximum(np.std(D[tr],0),.01); K=K+.1*D@D.T
 G=K[np.ix_(tr,tr)]+max(reg,.001)*np.eye(len(tr)); A=base[tr]; P=solve(G,np.c_[y[tr],A],assume_a='pos'); c=np.linalg.solve(A.T@P[:,1:],A.T@P[:,0]); return base[te]@c+K[np.ix_(te,tr)]@(P[:,0]-P[:,1:]@c)
cases=list(itertools.product(range(4),range(3),range(3),[0,.01,.03,.08])); all_idx=np.arange(N)
res=[]
for case in cases:
 m=[np.mean(abs(evaluate(np.setdiff1d(all_idx,te),te,case)-y[te])) for te in folds];res.append((max(m),np.mean(m),case,np.round(m,5)))
for r in sorted(res,key=lambda z:z[:2])[:10]: print(r)
for te in folds:
 tr=np.setdiff1d(all_idx,te); bx=np.max(abs(x[tr]),1); bb=np.linalg.norm(b[tr],axis=1); inner=[*np.array_split(tr,3),tr[np.argsort(bx)[-12:]],tr[np.argsort(bb)[-12:]]]; ranked=[]
 for case in [q for q in cases if q[3] in (.01,.03,.08)]:
  m=[np.mean(abs(evaluate(np.setdiff1d(tr,iv),iv,case)-y[iv])) for iv in inner]; ranked.append((max(m)*.5+np.mean(m)*.5,case))
 selected=min(ranked)[1]; print('nested',selected,np.mean(abs(evaluate(tr,te,selected)-y[te])))
