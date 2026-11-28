import pickle
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('solver','/holdout_task/batch_solver.py');m=module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];y=s['train']['y'];a=np.arange(len(x));r=np.sum(x*x,1)
pa=s['panel_a']; u,v=pa['u'],pa['v']; idx=np.arange(len(u))
for _ in range(5):
 uc=u[idx]-u[idx].mean(0); scale=(uc*(v[idx]-v[idx].mean(0))).sum(0)/(uc*uc).sum(0)
 bias=(v[idx]-u[idx]*scale).mean(0);idx=np.argsort(np.linalg.norm(u*scale+bias-v,axis=1))[:-3]
pb=s['panel_b']; w=np.linalg.lstsq(np.c_[np.ones(len(pb['u'])),pb['u']],pb['v'],rcond=None)[0]; b=(x-bias)/scale@w[1:]+w[0]
F=m._quadratic(x); P=F@F.T; D=cdist(x,x,'sqeuclidean'); delta=(b[:,None,:]-b[None,:,:])**2
G={}
for width in (2.,4.,8.):G[width]=np.exp(-delta/np.maximum(b.var(0),1e-12)/width).mean(2)
base=[(P+np.exp(-D/(2*l*l)),nu) for l,nu in ((2.,.0001),(1.4,.001))]
def predict(k,i,j):
 K,nu=k;return K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+nu*np.eye(len(i)),y[i],assume_a='pos')
def ref(i,j):return .75*predict(base[0],i,j)+.25*predict(base[1],i,j)
choices=[(amp,width,(base[0][0]+amp*G[width],base[0][1])) for amp in (.035,.075,.15,.3) for width in (2.,4.,8.)]
def tune(i):
 top=i[np.argsort(r[i])[-max(8,len(i)//6):]]; groups=list(np.array_split(i,5))+[top]
 def score(item):
  err=[]
  for h in groups:
   l=np.setdiff1d(i,h);err.append(np.mean(abs(predict(item[2],l,h)-y[h])))
  return np.mean(err[:5])+.7*max(err)
 return min(choices,key=score)
res=[]
for j in list(np.array_split(a,6))+[np.argsort(r)[-12:],np.argsort(r)[:12],a[56:]]:
 i=np.setdiff1d(a,j);amp,width,k=tune(i);u=ref(i,j);v=predict(k,i,j);p=predict(next(c[2] for c in choices if c[:2]==(.15,4.)),i,j)
 row=[np.mean(abs(h-y[j])) for h in (u,v,.5*(u+v),p,.5*(u+p))];res.append(row);print('fold',j[0],amp,width,np.round(row,6))
print('columns: reference, nested panel, nested blend, fixed panel, fixed blend')
print('block means',np.mean(res[:6],0),'worst',np.max(res,0),'tail',res[-1]);print('full-data choice',tune(a)[:2])
