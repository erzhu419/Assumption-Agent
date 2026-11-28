import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; t=np.asarray(s['eval_times']); rng=np.random.default_rng(91)
def fit(F,y):return np.linalg.lstsq(F,y,rcond=None)[0]
def design(u):return np.c_[np.ones(len(u)),u]
p=s['panel_a'];u=p['u'];v=p['v'];ok=np.linalg.norm(v,axis=1)<7;u=u[ok];v=v[ok];ids=[ix for k in range(4) for ix in it.combinations_with_replacement(range(4),k)]
def poly(u,ids):return np.column_stack([np.prod(u[:,ix],axis=1) for ix in ids])
w=fit(poly(u,ids),v); support=np.max(abs(w),axis=1)>.01; ids=[ix for ix,k in zip(ids,support) if k];w=fit(poly(u,ids),v);print('A',list(zip(ids,w.round(4).tolist())), 'rms',np.std(poly(u,ids)@w-v));
def inv(x):return least_squares(lambda z:(poly(z.reshape(-1,4),ids)@w-x).ravel(),x.ravel(),jac_sparsity=np.kron(np.eye(len(x)),np.ones((4,4))),gtol=1e-11).x.reshape(-1,4)
z=inv(x); p=s['panel_b'];B=fit(design(p['u']),p['v']); p=s['panel_c'];u=p['u'];pairs=[(0,1),(1,2),(2,3),(0,3)];
def v1(z):return poly(z,[()]+pairs)@W1
W1=fit(poly(u,[()]+pairs),p['v1']);
for typ in ['exp','cos','rat','tanh','power']:
 def g(u,k):return {'exp':lambda:np.expm1(-k*u*u),'cos':lambda:np.cos(k*u)-1,'rat':lambda:1/(1+k*u*u)-1,'tanh':lambda:np.tanh(k*u*u),'power':lambda:abs(u)**k}[typ]()
 def e(k):
  f=design(g(u,k));return np.mean((f@fit(f,p['v2'])-p['v2'])**2)
 o=minimize_scalar(e,bounds=(.01,4),method='bounded');print('C2',typ,o.x,np.sqrt(o.fun));
 if typ=='exp': k2=o.x;W2=fit(design(g(u,k2)),p['v2']);
def v2(z):return design(np.expm1(-k2*z*z))@W2
coords={'raw':x,'A':z,'AB':design(z)@B,'AiB':(z-B[0])@np.linalg.inv(B[1:]),'B':design(x)@B,'iB':(x-B[0])@np.linalg.inv(B[1:])};print('coord ranges',[(k,np.round(v.min(0),2),np.round(v.max(0),2)) for k,v in coords.items()]); folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(rng.permutation(72),4)]
def lib(Z,kind):
 args=[(str(j),a) for j,a in enumerate(Z.T)]; L=args.copy(); L.extend([('*'.join(map(str,ix)),np.prod(Z[:,ix],axis=1)) for d in [2,3] for ix in it.combinations_with_replacement(range(4),d)]); L+= [(f'c{i}',a) for i,a in enumerate(np.c_[v1(Z),v2(Z)].T)]
 if kind==0:return L
 for i,j in it.combinations(range(4),2):
  u,v=Z[:,i],Z[:,j]; args += [(f'{i}+{j}',u+v),(f'{i}-{j}',u-v)];L+= [(f'{i}{j}/norm',u*v/(1+u*u+v*v)),(f'{i}{j}/sum',u*v/(1+(u+v)**2)),(f'{i}{j}/diff',u*v/(1+(u-v)**2))];
  for k in (.3,.6,1.,1.5):
   L += [(f'{i}/(1+{k}*{j}^2)',u/(1+k*v*v)),(f'{j}/(1+{k}*{i}^2)',v/(1+k*u*u))]
 args += [(f'c{i}',a) for i,a in enumerate(np.c_[v1(Z),v2(Z)].T)]
 for nm,a in args:
  for k in (.5,1,1.5,2,3):
   L.extend([(f'{f.__name__}({k}*{nm})',f(k*a)) for f in [np.sin,np.cos,np.tanh,np.arctan]])
  L.extend([(f'sat{nm}',a/(1+abs(a))),(f'abs{nm}',abs(a))])
 return L
def metrics(A,method,reg):
 errs=[]
 for a,b in folds:
  F=(A-A[a].mean(0))/np.maximum(A[a].std(0),1e-9); U=Y[a]-Y[a].mean(0)
  if method=='ridge':p=F[b]@F[a].T@np.linalg.solve(F[a]@F[a].T+reg*np.eye(len(a)),U)
  else:
   p=[]
   for y in U.T:
    rem=y.copy(); cols=[]
    for _ in range(reg):
     k=np.argmax(abs(F[a].T@rem));cols.append(k);w=fit(F[a][:,cols],y);rem=y-F[a][:,cols]@w
    p.append(F[b][:,cols]@w)
   p=np.asarray(p).T
  e=abs(p+Y[a].mean(0)-Y[b]); errs.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.array(errs)
if __name__=='__main__':
 for k,Z in coords.items():
  for kind in (0,1):
   terms=lib(Z,kind); A=np.column_stack([v for _,v in terms]); R=[]
   for m,vals in [('ridge',[.01,.1,1,5]),('omp',[4,7,10,15])]:
    for r in vals:
     e=metrics(A,m,r);R.append((e.mean(),m,r,e[0].round(4),e[1:].max(0).round(4)))
   print(k,kind,sorted(R,key=lambda z:z[0])[:2],flush=True)
