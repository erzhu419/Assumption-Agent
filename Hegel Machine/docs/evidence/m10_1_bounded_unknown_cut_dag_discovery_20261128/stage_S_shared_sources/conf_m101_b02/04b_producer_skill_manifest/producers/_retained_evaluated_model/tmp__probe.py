import pickle,itertools as it,numpy as np
from scipy.optimize import minimize, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=150)
def aug(x): return np.c_[np.ones(len(x)),x]
def fit(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,c],axis=1) for k in range(1,d+1) for c in it.combinations_with_replacement(range(x.shape[1]),k)])
def report(X,Y):
 b=fit(X,Y);e=Y-X@b;h=np.sum(np.linalg.qr(X,mode='reduced')[0]**2,axis=1);return np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-h[:,None]))**2))
def cal(p,k='v'):
 d=s[p];u=d['u'];v=d[k];H=aug(u);m=np.ones(len(u),bool)
 if p=='panel_a': m=np.linalg.norm(v-H@fit(H,v),axis=1)<3
 p=fit(H[m],v[m]);return lambda x:aug(x)@p
A=cal('panel_a');B=cal('panel_b'); u=s['panel_c']['u']; fns={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'asinh':np.arcsinh,'softsign':lambda z:z/np.sqrt(1+z*z)}
operators={}
for k in ('v1','v2'):
 v=s['panel_c'][k]; print('\n',k,'quadratic coefficients\n',fit(poly(u,2),v)); print('quad',report(poly(u,2),v))
 best=[]
 for tag,f in fns.items():
  for add in (False,True):
   def gram(x,a,f=f,add=add):return aug(np.c_[x,f(x*a)] if add else f(x*a))
   def loss(a):return report(gram(u,a),v)[0]**2
   r=minimize_scalar(loss,bounds=(.04,5),method='bounded'); q=minimize(loss,np.repeat(r.x,4),method='L-BFGS-B',bounds=[(.04,6)]*4,options={'ftol':1e-15,'gtol':1e-10});g=gram(u,q.x);stats=report(g,v); print(tag,add,'shared',r.x,report(gram(u,r.x),v),'per-axis',q.x,stats)
   best.append((stats[1],tag,add,lambda x,g=gram,b=fit(g,v),a=q.x:g(x,a)@b))
 if k=='v2':
  for fn in (lambda x:x, np.tanh,np.sin):
   for z in (.25,.5,1.,1.5):
    if fn(np.array([1.]))[0]==1:continue
    g=poly(fn(u*z),2);print('quad nonlinear input',fn.__name__,z,report(g,v))
  for c in (1.2,1.5,2,3,5):
   a=poly(u,2);b=fit(a,np.arctanh(v/c));print('tanh quadratic out',c,np.sqrt(np.mean((c*np.tanh(a@b)-v)**2)))
  operators[k]=lambda x,b=fit(poly(u,2),v):poly(x,2)@b
 else:
  _,tag,add,operators[k]=min(best,key=lambda a:a[0]);print('winner',tag,add)
x=s['train']['x'];z=s['train']['traj'];y=s['train']['y'];t=s['eval_times'];V=np.linalg.svd(z[:52],full_matrices=False)[2][:2];Y=np.c_[y,z@V.T]; a=A(x);b=B(a);C=operators['v1'];D=operators['v2']; results=[]
features={'x':x,'A':a,'B':B(x),'BA':b,'AB':A(B(x)),'Ai':(x-A(np.zeros_like(x)))/(A(np.ones_like(x))-A(np.zeros_like(x)))}
for key,F in features.items():
 for tag,H in {'C':C(F),'D':D(F),'CD':np.c_[C(F),D(F)],'CF':np.c_[C(F),F],'CDF':np.c_[C(F),D(F),F]}.items():
  for ii,G in enumerate([aug(H),poly(H,2),np.c_[aug(H),np.tanh(H),np.sin(2*H),H**2,H**3]]):
   for l in [.0001,.005,.05,.3,1]:
    tr=G[:52]; e=(G[52:]@np.linalg.solve(tr.T@tr+l*np.eye(G.shape[1]),tr.T@Y[:52]))-Y[52:];m=np.mean(abs(e),0);results.append((m,key,tag,ii,l))
for j in range(3):
 print('best for',j);print(*sorted(results,key=lambda r:r[0][j])[:10],sep='\n')
