import sys,pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from node_00 import _calibrate
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=len(x)
aw,B,C,H,O=_calibrate(s)
folds=[np.arange(n-20,n),*np.array_split(np.arange(n),4)]; Y=np.c_[y,t]; rows=[]
def evaluate(K):
 e=[]
 for v in folds:
  i=np.setdiff1d(np.arange(n),v);d=K[np.ix_(i,i)]; l=K[np.ix_(v,i)]; w,u=np.linalg.eigh(d);a=(l@u)[:,:,None]*(u.T@Y[i])[None,:,:];p=np.einsum('vkm,kr->rvm',a,1/(w[:,None]+[1e-6,1e-5,.0001,.001,.01,.1]));r=abs(p-Y[v]);e.append(np.c_[r[:,:,0].mean(1),r[:,:,1:].mean((1,2))])
 return np.array(e).transpose(1,0,2)
for ci,z in enumerate([x,np.linalg.solve(aw[1:].T,(x-aw[0]).T).T,x@aw[1:]+aw[0]]):
 b=B(z); o=O(z); c=C(z); h=H(z)
 for gi,g in enumerate([z,np.c_[z,o],np.c_[z,b],np.c_[z,o,h],np.c_[z,np.tanh(2*z)],np.c_[z,z**3],np.c_[z,b,c,h],np.c_[z,z*np.exp(-z*z)],np.c_[z,np.sin(2.8*z)]]):
  g=(g-g.mean(0))/g.std(0); dots=[g[:,j::4]@g[:,j::4].T/g[:,j::4].shape[1] for j in range(4)] if gi!=6 else [g@g.T/g.shape[1]]*4
  dists=[np.diag(d)[:,None]+np.diag(d)[None,:]-2*d for d in dots]; dz=z/z.std(0); lin=dz@dz.T/4; gz=g@g.T/g.shape[1]
  for kind,l in it.product(['poly','rbf','matern'],[.5,1,2,4,8]):
   ds=[(1+d/l)**3-1 for d in dots] if kind=='poly' else [np.exp(-d/(2*l*l)) for d in dists] if kind=='rbf' else [(1+np.sqrt(np.maximum(d,0))/l)*np.exp(-np.sqrt(np.maximum(d,0))/l) for d in dists]
   A=sum(ds)/4; P=sum(ds[j]*ds[k] for j,k in it.combinations(range(4),2))/6; Trip=sum(ds[i]*ds[j]*ds[k] for i,j,k in it.combinations(range(4),3))/4
   for pi,K in enumerate([A,.1*P+A,P+A,A+P+.2*Trip,A+.2*lin*A,A+lin*lin*.1,A+.1*gz**2,A+.1*gz**3,A+.1*lin*gz+.02*lin*lin*gz,A+.05*P+.02*Trip]):
    for ri,e in enumerate(evaluate(1+lin+K)):rows.append(((ci,gi,kind,l,pi,ri),e))
for j in range(2):
 for w in [0,.5,1]:
  print(j,w,*[(a,b.round(5).tolist()) for a,b in sorted(rows,key=lambda q:w*q[1][:,j].max()+(1-w)*q[1][0,j])[:5]],sep='\n')
