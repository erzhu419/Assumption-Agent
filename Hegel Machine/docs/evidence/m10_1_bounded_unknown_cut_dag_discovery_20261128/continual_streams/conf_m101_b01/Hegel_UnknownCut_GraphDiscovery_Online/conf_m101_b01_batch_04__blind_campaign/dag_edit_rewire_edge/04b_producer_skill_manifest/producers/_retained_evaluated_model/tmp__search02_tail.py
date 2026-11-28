from scipy.optimize import minimize_scalar
L=poly(q,1)
for kind in ['sin','sin+cos','tanh','rat']:
 def extra(A,f):
  if kind=='sin':return np.sin(f*A)
  if kind=='sin+cos':return np.c_[np.sin(f*A),np.cos(f*A)]
  if kind=='tanh':return np.tanh(f*A)
  return A/(1+f*A*A)
 obj=lambda f:np.mean(loo(np.c_[L,extra(q,f)],p['v1'])[1])
 fs=np.arange(.2,9,.05); f=min(fs,key=obj);opt=minimize_scalar(obj,bounds=(max(.05,f-.15),f+.15),method='bounded');print('panel1',kind,opt.x,opt.fun)
for f in [np.arctanh,lambda x:x,np.arcsin]:
 A=poly(q,2); b=f(p['v2']);print('panel2',f,loo(A,b)[1])
A=np.c_[u,z]; sets={}
def add(group,X,names):sets[group]=(X,names)
add('aff',z,[f'z{i}' for i in range(4)])
for deg in [2,3]:
 idx=list(combinations_with_replacement(range(4),deg));add(f'u{deg}',np.column_stack([np.prod(u[:,k],1) for k in idx]),[f'u{k}' for k in idx]);add(f'z{deg}',np.column_stack([np.prod(z[:,k],1) for k in idx]),[f'z{k}' for k in idx]);
pairs=np.array([A[:,i]*A[:,j] for i in range(8) for j in range(i,8)]).T;pn=[f'{i}*{j}' for i in range(8) for j in range(i,8)]
dirs=np.c_[A,*[A[:,i]+sg*A[:,j] for i in range(8) for j in range(i+1,8) for sg in [-1,1]]];dn=[str(i) for i in range(8)]+[f'{i}{sg:+}({j})' for i in range(8) for j in range(i+1,8) for sg in [-1,1]]
for f in [0.5,1,1.3,1.5,1.7,2,2.2,2.5,3,3.5,4]:
 for name,g in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda x:x/(1+x*x))]:
  tag=f'{name}{f}'; add(tag,g(f*dirs),[f'{tag}({d})' for d in dn]);add(tag+'p',g(f*pairs),[f'{tag}({d})' for d in pn]);
  if name in ['sin','tanh','rat']:
   D=(g(f*z)[:,:,None]*A[:,None,:]).reshape(len(A),-1);add(tag+'m',D,[f'{tag}(z{i})*{j}' for i in range(4) for j in range(8)])
for f in [2.5,3,3.5,4,4.5]:
 C=lambda x:np.c_[poly(x,1),np.sin(f*x),np.cos(f*x)]; add(f'panel1_{f}',C(z)@np.linalg.lstsq(C(q),p['v1'],rcond=None)[0],[f'v1_{f}_{i}' for i in range(4)])
add('panel2',poly(z,2)@np.linalg.lstsq(poly(q,2),p['v2'],rcond=None)[0],[f'v2_{i}' for i in range(4)])
D=np.concatenate([v[0] for v in sets.values()],axis=1)[:n]; names=sum([v[1] for v in sets.values()],[])
# Orthogonal matching pursuit with training-only atom selection.
def omp(D,b,tr,te,maxk=16,base=()):
 R=D[tr].copy();V=D[te]; R-=R.mean(0); G=R.copy(); norm=np.linalg.norm(R,axis=0); y=b[tr]-b[tr].mean(); use=[]; results=[]
 for step in range(maxk+len(base)):
  i=base[step] if step<len(base) else np.argmax(np.abs(y@R)/np.maximum(np.linalg.norm(R,axis=0),1e-10));v=R[:,i].copy();v/=np.linalg.norm(v);R-=v[:,None]*(v@R);y-=v*(v@y);use.append(i)
  F=np.c_[np.ones(len(tr)),D[np.ix_(tr,use)]]; c=np.linalg.lstsq(F,b[tr],rcond=None)[0]; pred=np.c_[np.ones(len(te)),V[:,use]]@c;results.append((np.mean(abs(pred-b[te])),[(names[j],round(a,4)) for j,a in zip(use,c[1:])],np.mean(abs(y))))
 return results
for b in [Y[:,0],Y[:,-1]]:
 for base in [(),tuple(range(4))]:
  out=[omp(D,b,j,k,13,base) for j,k in folds];print('OMP',b is Y[:,0],len(base));print(np.array([[r[0] for r in row] for row in out]).round(4)); print('structure',out[0][9]); print('all',omp(D,b,I,I,12,base)[9]);
 for name, (F,_) in sets.items():
  if F.shape[1] <= 10 or name.startswith(('tanh','rat','sin')) and name.endswith('m'):
   e=cv(np.c_[z,F][:n],0.035,b[:,None]);
   if e[:,0].max()<.14: print('group',name,e[:,0])
