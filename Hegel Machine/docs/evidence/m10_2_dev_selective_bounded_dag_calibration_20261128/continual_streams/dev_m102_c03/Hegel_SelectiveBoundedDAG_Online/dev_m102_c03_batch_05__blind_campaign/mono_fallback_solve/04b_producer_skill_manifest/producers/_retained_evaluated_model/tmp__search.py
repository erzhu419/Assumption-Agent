from operators import *
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; inv,B,C,D=operators(s); z=inv(s['train']['x']); c=C(z); d=D(z); T=s['train']['traj'];y=s['train']['y'];t=s['eval_times'];n=52
q=lambda g:-np.expm1(-t[:,None]*g); g=least_squares(lambda g:(q(g)@fit(q(g),T[:n].T)-T[:n].T).ravel(),[.4,1.8]).x; a=fit(q(g),T.T).T; Y=np.c_[a,y]; h=one(np.c_[c,d]); P=np.linalg.pinv(h[:n]); res=Y-h@(P@Y[:n]); rng=np.random.default_rng(1)
print('res SVD',np.linalg.svd(res,compute_uv=False)); print('input',c.min(0),c.max(0),d.min(0),d.max(0));print('base',np.mean(abs(res[n:]),0))
features=[];names=[]
def add(nm,x):features.append(x);names.append(nm)
for k,u in [('c',c),('d',d),('z',z),('bz',B(z)),('edge',z*np.roll(z,-1,axis=1)),('cos',np.cos(2.4*z))]:
 for i in range(4):
  v=u[:,i]
  for name,f in [('sq',lambda v:v*v),('cu',lambda v:v**3),('abs',abs),('tanh',np.tanh),('sin',np.sin),('rat',lambda v:v/(1+v*v)),('atan',np.arctan)]:add(f'{k}{i}{name}',f(v))
 for i,j in combinations(range(4),2):
  for m in [-2,-1,-.5,.5,1,2]:
   v=u[:,i]+m*u[:,j]
   for name,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda v:v/(1+v*v)),('sq',lambda v:v*v),('cu',lambda v:v**3)]:add(f'{k}{i}+{m:g}*{j}{name}',f(v))
  add(f'{k}{i}*{j}',u[:,i]*u[:,j]);add(f'{k}{i}/(1+{j}^2)',u[:,i]/(1+u[:,j]**2));add(f'{k}{j}/(1+{i}^2)',u[:,j]/(1+u[:,i]**2))
for i in range(4):
 for j in range(4):
  for name,f in [('mul',lambda a,b:a*b),('rat',lambda a,b:a/(1+b*b)),('ratr',lambda a,b:b/(1+a*a)),('tansum',lambda a,b:np.tanh(a+b)),('tandif',lambda a,b:np.tanh(a-b))]:add(f'c{i}d{j}{name}',f(c[:,i],d[:,j]))
F=np.array(features).T; F-=h@(P@F[:n]); F/=np.maximum(1e-7,np.linalg.norm(F[:n],axis=0));
for j in range(3):
 rr=res[:,j]; ids=[];print('TOP',j,[(names[i],round(sv,3)) for i,sv in sorted(enumerate(abs(F[:n].T@rr[:n])),key=lambda i:-i[1])[:10]])
 for k in range(7):
  ids.append(np.argmax(abs(F[:n].T@rr[:n]))); X=F[:,ids];rr=res[:,j]-X@fit(X[:n],res[:n,j]);print('omp',names[ids[-1]],np.mean(abs(rr[:n])).round(4),np.mean(abs(rr[n:])).round(4))
for nm,Z in [('Cquad',c),('Dquad',d),('CDquad',np.c_[c,d]),('zquad',z),('C+Dcube',np.c_[c**3,d**3]),('sepsat',np.c_[np.tanh(c),np.tanh(d),np.sin(c),np.sin(d)])]:
 X=np.c_[h,Z,*[Z[:,i]*Z[:,j] for i in range(Z.shape[1]) for j in range(i,Z.shape[1])]] if 'quad' in nm else np.c_[h,Z]; rr=Y-X@fit(X[:n],Y[:n]); print(nm,np.mean(abs(rr[:n]),0),np.mean(abs(rr[n:]),0))
for j in [0,1,2]:
 out=[]
 for nm,u in [('c',c),('d',d),('cd',np.c_[c,d]),('z',z),('bz',B(z)),('edge',z*np.roll(z,-1,axis=1)),('cos',np.cos(2.4*z))]:
  U=one(u)
  for fn,f in [('tanh',np.tanh),('sin',np.sin),('rat',lambda v:v/(1+v*v)),('atan',np.arctan),('gauss',lambda v:np.exp(-v*v)),('sq',lambda v:v*v),('cu',lambda v:v**3)]:
   def rr(w,full=False):
    X=np.c_[h,f(U@w)];r=Y[:,j]-X@fit(X[:n],Y[:n,j]);return r if full else r[:n]
   M=[]
   for i in range(7):
    w0=rng.normal(0,.6,U.shape[1]);w=least_squares(rr,w0,max_nfev=80).x;r=rr(w,True);M.append((np.std(r[:n]),np.mean(abs(r[n:])),w))
   best=min(M,key=lambda x:x[0]);out.append((*best[:2],nm,fn,best[2].round(3)))
 print('RIDGE',j,sorted(out,key=lambda x:x[0])[:12])
