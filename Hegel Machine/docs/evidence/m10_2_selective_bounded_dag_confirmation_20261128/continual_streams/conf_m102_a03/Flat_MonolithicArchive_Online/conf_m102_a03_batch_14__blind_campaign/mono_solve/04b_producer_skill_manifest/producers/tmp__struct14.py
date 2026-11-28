import sys,pickle,numpy as np,itertools
sys.path.insert(0,'/holdout_task')
from batch_solver import _features,_fit,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; n=len(y); h,z,b,d,q=_features(s,tr['x']); p=q.sum(1); X=(d[:,:,None]*q[:,None,:]).reshape(n,16); ar=np.arange(n); rr=np.random.default_rng(34); folds=[(ar[:52],ar[52:])]+[(i[:52],i[52:]) for i in [rr.permutation(n) for _ in range(12)]]
def cv(A,B=y):
 e=np.array([np.mean(abs(A[v]@_fit(A[t],B[t])-B[v])) for t,v in folds]); return np.array([e[0],e.mean(),max(e),np.mean(abs(A[:52]@_fit(A[:52],B[:52])-B[:52]))])
print('baseline',cv(np.c_[h,X]),cv(h,T)); print('coef',_fit(np.c_[h,X][:52],y[:52]).round(4)); print('coef all',_fit(np.c_[h,X],y).round(4)); print('state',_fit(h,T).round(4)); print('D3 delta',np.max(abs(np.load('/tmp/spaces.npz')['D3'][:n]-d)))
results=[]
def ck(k,*cols):
 A=np.column_stack((h,*cols));results.append((cv(A),k))
for lag in range(4):
 v=d*np.roll(q,lag,axis=1)
 for k,V in [('all',v),('sum',v.sum(1))]:ck(('cyclic',lag,k),V)
for k,V in [('D',d),('Z',z),('B',b),('Q',q)]:
 for j in range(4):
  for l in range(j,4):
   for t in [.1,.25,.5,1,2,4]:
    for m,U in [('sum',V[:,j]+V[:,l]),('diff',V[:,j]-V[:,l]),('prod',V[:,j]*V[:,l]),('norm',np.hypot(V[:,j],V[:,l]))]:
     g=1/(1+t*U**2)
     ck((k,j,l,t,m,'p'),p*g);ck((k,j,l,t,m,'all'),h[:,1:]*g[:,None]);ck((k,j,l,t,m,'D'),p*g,d)
     ck((k,j,l,t,m,'b'),p*g,b*g[:,None]);ck((k,j,l,t,m,'X'),p*g,X)
for k in range(1,6):
 R=[]
 for idx in itertools.combinations(range(16),k):
  a=np.c_[h,X[:,idx]]; t=folds[0][0]; r=np.mean((a[t]@_fit(a[t],y[t])-y[t])**2);R.append((r,idx))
 for _,i in sorted(R)[:5]:ck(('sparse',i),X[:,i])
print('mean',sorted(results,key=lambda a:a[0][1])[:15]);print('train',sorted(results,key=lambda a:a[0][3])[:12]);print('fixed',sorted(results,key=lambda a:a[0][0])[:10]);
