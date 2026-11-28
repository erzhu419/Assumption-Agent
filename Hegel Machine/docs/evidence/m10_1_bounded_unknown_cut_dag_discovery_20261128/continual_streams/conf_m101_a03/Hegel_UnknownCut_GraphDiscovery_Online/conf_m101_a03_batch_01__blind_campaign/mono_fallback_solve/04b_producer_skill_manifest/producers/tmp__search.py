import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; y=d['y']; t=d['traj']; x=d['x']
def a(x): return np.c_[np.ones(len(x)),x]
def fit(f,y): return np.linalg.lstsq(f,y,rcond=None)[0]
p=s['panel_a']; f=a(p['u']); r=np.linalg.norm(p['v']-f@fit(f,p['v']),axis=1); k=r<4*np.median(r); A=fit(f[k],p['v'][k]); u=(x-A[0])/np.diag(A[1:]); p=s['panel_b']; B=fit(a(p['u']),p['v']); z=a(u)@B
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=18).split(x)]
def score(f): return np.array([np.mean(abs(f[j]@fit(f[i],y[i])-y[j])) for i,j in splits])
def poly(x,n):
 idx=[j for k in range(1,n+1) for j in cwr(range(x.shape[1]),k)]; return a(np.column_stack([x[:,j].prod(1) for j in idx])),['1']+[''.join(map(str,j)) for j in idx]
from itertools import combinations,product
np.set_printoptions(precision=4,suppress=True)
base=np.c_[a(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]]; Q,_=poly(z,2)
print('pooled coefficients',fit(base,y),'errors',score(base))
r=y-base@fit(base,y)
for basis in (base,Q):
 fs=[]; ns=[]
 def add(n,f): ns.append(n);fs.append(f)
 w=z.T; L=[(str(i),w[i]) for i in range(4)]+[(f'{i}{sg}{j}',w[i]+sg*w[j]) for i,j in combinations(range(4),2) for sg in (1,-1)]
 L+= [(f'{i}*{j}',w[i]*w[j]) for i,j in cwr(range(4),2)]
 L+=[('pool',base[:,-1]),('state',fit(a(z),t)[1:,-1]@w)]
 for n,g in L:
  add(n+'^3',g**3)
  for k in [.5,1,1.4,1.7,2,2.5,3]:
   for m,f in [('sin',np.sin(k*g)),('cos',np.cos(k*g)),('tanh',np.tanh(k*g)),('rat',g/(1+k*g*g)),('sat',1/(1+k*g*g))]:
    add(f'{m}{k}({n})',f)
 for i,j,k in product(range(4), repeat=3):
  for b in (.2,.5,1,1.7):
   add(f'{i}*{j}/(1+{b}*{k}^2)',w[i]*w[j]/(1+b*w[k]**2));add(f'{i}*{j}sin({b}*{k})',w[i]*w[j]*np.sin(b*w[k]))
 F=np.array(fs).T; i,j=splits[0]; F=F-basis@fit(basis[i],F[i]); rr=y-basis@fit(basis[i],y[i]); c=(F[i].T@rr[i])/np.maximum(1e-20,(F[i]**2).sum(0)); er=np.mean(abs(F[j]*c-rr[j,None]),0)
 for k in np.argsort(er)[:10]: print(basis.shape[1],ns[k],c[k],score(np.c_[basis,F[:,k]]))
