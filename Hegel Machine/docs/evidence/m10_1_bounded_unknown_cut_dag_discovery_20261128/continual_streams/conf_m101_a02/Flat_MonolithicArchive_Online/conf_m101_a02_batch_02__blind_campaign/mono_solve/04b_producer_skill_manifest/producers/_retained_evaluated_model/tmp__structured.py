import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import null_space
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; n=len(x)
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(z,d):return np.column_stack([np.prod(z[:,i],axis=1) if i else np.ones(len(z)) for k in range(d+1) for i in combinations_with_replacement(range(z.shape[1]),k)])
a=s['panel_a']; u,v=a['u'],a['v']; A=aff(u); g=np.ones(len(u),bool)
for i in range(4):
 c=fit(A[g],v[g]); e=np.linalg.norm(v-A@c,axis=1);g=e<4*np.median(e)
a=(x-c[0])/np.diag(c[1:]); b=s['panel_b']; B=fit(aff(b['u']),b['v']); z=aff(a)@B
np.set_printoptions(precision=4,suppress=True)
def metrics(D,target=y):
 c=fit(D[:52],target[:52]); e=D@c-target
 return np.mean(abs(e[:52])),np.mean(abs(e[52:])),np.max(abs(e[52:])),c
from itertools import combinations,product
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
rng=np.random.default_rng(785)
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]; folds=np.array_split(rng.permutation(n),5); splits += [(np.setdiff1d(np.arange(n),j),j) for j in folds]
def score(fn):
 e=[np.mean(abs(fn(i,j)-y[j])) for i,j in splits];return np.round([*e[:2],np.mean(e[2:]),max(e[2:])],6)
def evald(D):return score(lambda i,j:D[j]@fit(D[i],y[i]))
b=z.copy(); w=b.T; L=aff(b); P=w[0]*w[2]+w[1]*w[3]; Q=w[0]*w[2]-w[1]*w[3]; T=np.column_stack([np.prod(w[list(k)],axis=0) for k in combinations(range(4),3)]); F=np.c_[L,P]; freq=s['telemetry_vec'][0]
print('linear coefficients',fit(F[:52],y[:52])); print('distinct triples',evald(np.c_[F,T]),fit(np.c_[F,T][:52],y[:52])); print('pair modulation',evald(np.c_[F,P[:,None]*b]),evald(np.c_[F,Q[:,None]*b]))
terms=[]
for k in range(1,5):
 for inds in combinations(range(4),k):
  for tail in product((-1,1),repeat=k-1):
   signs=(1,)+tail; terms.append((str((inds,signs)),np.array(signs)@w[list(inds)]))
for i,j in combinations(range(4),2):terms.append((f'{i}{j}',w[i]*w[j]))
for name,col in [('pair+',P),('pair-',Q)]+[(f'tri{i}',T[:,i]) for i in range(4)]:terms.append((name,col))
D=[]; names=[]
for name,col in terms:
 for k in [1.,freq]:
  funcs={'sin':np.sin(k*col),'cos':np.cos(k*col),'tanh':np.tanh(k*col),'rational':col/(1+k*col**2),'inv':1/(1+k*col**2)}
  for kind,val in funcs.items():
   for label,mult in [('one',1),('p',P),('q',Q)]+[(str(l),w[l]) for l in range(4)]:
    names.append((name,kind,k,label));D.append(val*mult)
for signs in product((-1,0,1),repeat=4):
 if not any(signs) or next(s for s in signs if s)!=1:continue
 v=T@signs
 for k in [1.,freq]:
  names.append((signs,'sintri',k));D.append(np.sin(k*v))
D=np.asarray(D).T
for tag,base in [('pair',F),('pairtrip',np.c_[F,T]),('quad',poly(b,2))]:
 def model(i,j,m=1):
  H=D-base@fit(base[i],D[i]); norm=np.linalg.norm(H[i],axis=0); norm=np.where(norm>1e-9,norm,np.inf); hn=H/norm; picked=[]; X=base; e=y[i]-X[i]@fit(X[i],y[i]);
  for _ in range(m):
   ix=np.argmax(abs(hn[i].T@e));picked.append(ix);X=np.c_[base,H[:,picked]];e=y[i]-X[i]@fit(X[i],y[i]);
  return X[j]@fit(X[i],y[i]),[names[k] for k in picked]
 for m in [1,2,3,5,8]:print('grammar',tag,m,score(lambda i,j:model(i,j,m)[0]),model(*splits[0],m)[1]);
for name,u in [('a',a),('b',b)]:
 dist=cdist(u,u,'sqeuclidean'); dot=u@u.T; lin=aff(u)
 for family in ['poly','rbf']:
  for scale in ([.25,.5,1,2] if family=='poly' else [.15,.35,.7,1.3]):
   for ridge in [.0001,.003,.03]:
    K=(1+scale*dot)**4 if family=='poly' else np.exp(-scale*dist)
    def predict(i,j):
     C=K[np.ix_(i,i)]+ridge*np.eye(len(i)); X=np.linalg.solve(C,np.c_[y[i],lin[i]]); beta=np.linalg.solve(lin[i].T@X[:,1:],lin[i].T@X[:,0]); return lin[j]@beta+K[np.ix_(j,i)]@(X[:,0]-X[:,1:]@beta)
    v=score(predict)
    if max(v)<.049:print('kernel',name,family,scale,ridge,v)
